# qdax_binpack/behaviours.py

import jumanji
# Using original Jumanji Observation types for the existing descriptor function
from jumanji.environments.packing.bin_pack.types import Observation as JumanjiObservation
from jumanji.environments.packing.bin_pack.types import Item, item_volume

import chex
import jax
import jax.numpy as jnp

from typing import Sequence, Tuple, Union, Optional
from qdax.custom_types import Descriptor, Genotype
from qdax.core.neuroevolution.buffers.buffer import QDTransition


def _calculate_normalized_rank(chosen_value: chex.Numeric,
                               all_values: chex.Array,
                               valid_mask: chex.Array) -> chex.Numeric:
    """
    Calculates the normalized rank of a chosen value within a set of valid values.
    Rank is [0, 1], where 0 means smallest and 1 means largest.
    Handles cases with few valid values.
    (Original function - unchanged)
    """
    values_for_ranking = jnp.where(valid_mask, all_values, -jnp.inf)
    num_valid = jnp.sum(valid_mask)
    count_smaller = jnp.sum((values_for_ranking < chosen_value) & valid_mask)
    score = jax.lax.cond(
        num_valid > 1,
        lambda: count_smaller / (num_valid - 1.0),
        lambda: 0.5,
    )
    return jnp.clip(score, 0.0, 1.0)


# NEW: Updated function to handle both standard and extended (rotated) observations
def _get_prioritization_scores_single_step(
    obs: JumanjiObservation,
    flat_action: chex.Numeric,
    num_item_choices_from_spec: int, # This is the total number of item choices (e.g., O*I or I)
) -> Tuple[chex.Numeric, chex.Numeric]:
    """
    Calculates item and EMS prioritization scores for a single timestep.
    Handles both standard and extended (with rotations) observations.
    """
    # item_idx_chosen is an index into the flattened item/orientation array
    item_idx_chosen = flat_action % num_item_choices_from_spec
    ems_idx_chosen = flat_action // num_item_choices_from_spec

    # Conditionally flatten item-related parts of the Jumanji observation if they are multi-dimensional.
    # This is a robust way to detect the extended environment with rotations.
    is_extended_obs = obs.items.x_len.ndim > 1

    def get_volumes_and_mask_for_extended():
        # For extended env, obs.items leaves and obs.items_mask have shape (O, I).
        # We flatten them to work with the flat item_idx_chosen.
        flat_items = jax.tree_util.tree_map(lambda x: x.flatten(), obs.items)
        flat_mask = obs.items_mask.flatten()
        return item_volume(flat_items), flat_mask

    def get_volumes_and_mask_for_standard():
        # For standard env, shapes are already 1D: (I,).
        return item_volume(obs.items), obs.items_mask

    all_item_volumes, valid_item_selection_mask = jax.lax.cond(
        is_extended_obs,
        get_volumes_and_mask_for_extended,
        get_volumes_and_mask_for_standard,
    )
    
    # By this point, all_item_volumes and valid_item_selection_mask are 1D arrays,
    # and item_idx_chosen is a valid scalar index for them.
    chosen_item_volume = all_item_volumes[item_idx_chosen]
    item_prioritization_score = _calculate_normalized_rank(
        chosen_item_volume, all_item_volumes, valid_item_selection_mask
    )

    # EMS part is unchanged as it's not affected by item rotations.
    all_ems_volumes = obs.ems.volume()
    chosen_ems_volume = all_ems_volumes[ems_idx_chosen]
    valid_ems_selection_mask = obs.ems_mask
    ems_prioritization_score = _calculate_normalized_rank(
        chosen_ems_volume, all_ems_volumes, valid_ems_selection_mask
    )
    return item_prioritization_score, ems_prioritization_score

def binpack_descriptor_extraction( # Original name and signature restored
    data: QDTransition, # data.obs is expected to be JumanjiObservation
    mask: jnp.ndarray,
    num_item_choices_from_spec: int,
) -> Descriptor:
    """
    Computes descriptors based on episode activity:
    1. Average prioritization of large items.
    2. Average prioritization of large EMSs.
    (This function's body is unchanged as the logic is now handled in its helper)
    """
    vmapped_scores_over_time = jax.vmap(
        _get_prioritization_scores_single_step,
        in_axes=(0, 0, None),
        out_axes=0
    )
    vmapped_scores_over_batch_time = jax.vmap(
        vmapped_scores_over_time,
        in_axes=(0, 0, None),
        out_axes=0
    )
    item_scores_all, ems_scores_all = vmapped_scores_over_batch_time(
        data.obs, data.actions, num_item_choices_from_spec
    )
    sum_mask = jnp.sum(mask, axis=-1)
    safe_sum_mask = jnp.where(sum_mask == 0, 1.0, sum_mask)
    mean_item_prioritization = jnp.sum(item_scores_all * mask, axis=-1) / safe_sum_mask
    mean_ems_prioritization = jnp.sum(ems_scores_all * mask, axis=-1) / safe_sum_mask
    mean_item_prioritization = jnp.where(sum_mask == 0, 0.5, mean_item_prioritization)
    mean_ems_prioritization = jnp.where(sum_mask == 0, 0.5, mean_ems_prioritization)
    descriptors = jnp.stack([mean_item_prioritization, mean_ems_prioritization], axis=-1)
    return descriptors

# --- NEW: Descriptor Function for Heuristic Genomes (Unchanged) ---
# ... (The rest of the file is unchanged)
# Based on the new feature_stack order in `heuristic_policies.py` (13 features total)
# Interaction features: indices 0-4
# Item features: indices 5-7
# EMS features: indices 8-12

ITEM_FEATURE_GENOME_INDICES_SLICE = slice(5, 8)  # Corresponds to item_vol, item_max_dim, item_min_dim weights
EMS_FEATURE_GENOME_INDICES_SLICE = slice(8, 13)   # Corresponds to ems_vol, ems_x1,y1,z1, ems_min_dim weights

def compute_heuristic_genome_descriptors(
    policy_params: Genotype,
) -> Descriptor:
    """
    Computes descriptors based on the L1 norm of heuristic genome components.
    The policy_params (genotypes) are expected to be a PyTree containing
    a 'genome' leaf, which is an array of shape (batch_size, genome_length).
    genome_length should be NUM_EXPANDED_HEURISTIC_FEATURES (13).
    """
    actual_genomes = policy_params['genome']
    
    item_feature_weights = actual_genomes[:, ITEM_FEATURE_GENOME_INDICES_SLICE]
    desc1_item_l1_norm = jnp.sum(jnp.abs(item_feature_weights), axis=-1)

    ems_feature_weights = actual_genomes[:, EMS_FEATURE_GENOME_INDICES_SLICE]
    desc2_ems_l1_norm = jnp.sum(jnp.abs(ems_feature_weights), axis=-1)

    num_item_weights = ITEM_FEATURE_GENOME_INDICES_SLICE.stop - ITEM_FEATURE_GENOME_INDICES_SLICE.start
    max_l1_item = 1.0 * num_item_weights

    num_ems_weights = EMS_FEATURE_GENOME_INDICES_SLICE.stop - EMS_FEATURE_GENOME_INDICES_SLICE.start
    max_l1_ems = 1.0 * num_ems_weights
    
    normalized_desc1 = desc1_item_l1_norm / jnp.maximum(max_l1_item, 1e-6)
    normalized_desc2 = desc2_ems_l1_norm / jnp.maximum(max_l1_ems, 1e-6)
    
    final_descriptors = jnp.stack([
        jnp.clip(normalized_desc1, 0.0, 1.0),
        jnp.clip(normalized_desc2, 0.0, 1.0)
    ], axis=-1)

    return final_descriptors