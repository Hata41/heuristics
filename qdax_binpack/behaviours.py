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

from typing import List


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

def compute_heuristic_genome_descriptors(
    policy_params: Genotype,
    descriptor_slices: List[slice],
) -> Descriptor:
    """
    Computes descriptors based on the L1 norm of specified heuristic genome components.

    This function is a generalized version that accepts a list of slices. Each slice
    defines a group of features in the genome. For each group, it calculates the
    normalized L1 norm of the corresponding weights, creating one descriptor per slice.

    Args:
        policy_params: The genotypes, a PyTree expected to contain a 'genome' leaf
            of shape (batch_size, genome_length).
        descriptor_slices: A list of slice objects. The length of this list determines
            the number of descriptors. Each slice specifies the indices of the genome
            weights that contribute to that descriptor.

    Returns:
        A descriptor array of shape (batch_size, num_descriptors), where
        num_descriptors is len(descriptor_slices). Each descriptor value is
        clipped to the range [0.0, 1.0].
    """
    actual_genomes = policy_params['genome']
    
    all_descriptors = []

    # Iterate over the list of slices provided
    for feature_slice in descriptor_slices:
        # 1. Extract the weights for the current feature group
        feature_weights = actual_genomes[:, feature_slice]

        # 2. Calculate the L1 norm for this group
        l1_norm = jnp.sum(jnp.abs(feature_weights), axis=-1)

        # 3. Calculate the maximum possible L1 norm for normalization
        # Assumes weights are in [-1, 1], so max absolute value is 1.0
        num_weights_in_group = feature_slice.stop - feature_slice.start
        max_l1_for_group = 1.0 * num_weights_in_group

        # 4. Normalize the descriptor
        normalized_desc = l1_norm / jnp.maximum(max_l1_for_group, 1e-6)

        # 5. Clip the result and add to our list
        clipped_desc = jnp.clip(normalized_desc, 0.0, 1.0)
        all_descriptors.append(clipped_desc)

    # Stack all computed descriptors into a final array
    final_descriptors = jnp.stack(all_descriptors, axis=-1)

    return final_descriptors