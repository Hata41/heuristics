import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple, Tuple

# To make this file self-contained for now, let's use the full ArrayObservation from your repo.
# This would normally be an import.
class ArrayObservation(NamedTuple):
    ems: chex.Array
    ems_mask: chex.Array
    items: chex.Array
    items_mask: chex.Array
    items_placed: chex.Array
    action_mask: chex.Array

# --- 1. Featurizer (Expanded, Simplified Normalization) ---
class HeuristicFeatures(NamedTuple):
    # Interaction / Fit Features
    delta_x: chex.Array
    delta_y: chex.Array
    delta_z: chex.Array
    vol_diff: chex.Array
    can_fit: chex.Array

    # Item-Specific Features
    item_vol: chex.Array # Using raw volume, assuming it's already effectively normalized
    item_max_dim: chex.Array
    item_min_dim: chex.Array

    # EMS-Specific Features
    ems_vol: chex.Array # Using raw volume
    ems_coord_x1: chex.Array # Using raw coordinates
    ems_coord_y1: chex.Array
    ems_coord_z1: chex.Array
    ems_min_dim: chex.Array


def compute_heuristic_features(obs: ArrayObservation) -> HeuristicFeatures: # Removed container_dims
    """
    Computes features. Assumes item/EMS dimensions and coordinates are
    effectively pre-normalized if the container is unit size [1,1,1].
    """
    # EMS dimensions and volume
    ems_x_len = obs.ems[:, 1] - obs.ems[:, 0]
    ems_y_len = obs.ems[:, 3] - obs.ems[:, 2]
    ems_z_len = obs.ems[:, 5] - obs.ems[:, 4]
    ems_volume = ems_x_len * ems_y_len * ems_z_len
    ems_dims = jnp.stack([ems_x_len, ems_y_len, ems_z_len], axis=-1)
    ems_min_dim = jnp.min(ems_dims, axis=-1)

    # Item dimensions and volume
    item_x_len = obs.items[:, 0]
    item_y_len = obs.items[:, 1]
    item_z_len = obs.items[:, 2]
    item_volume = item_x_len * item_y_len * item_z_len
    item_dims = obs.items
    item_max_dim = jnp.max(item_dims, axis=-1)
    item_min_dim = jnp.min(item_dims, axis=-1)

    # Expand for broadcasting
    ems_x_len_exp, ems_y_len_exp, ems_z_len_exp = ems_x_len[:, None], ems_y_len[:, None], ems_z_len[:, None]
    ems_volume_exp = ems_volume[:, None]
    ems_min_dim_exp = ems_min_dim[:, None]

    item_x_len_exp, item_y_len_exp, item_z_len_exp = item_x_len[None, :], item_y_len[None, :], item_z_len[None, :]
    item_volume_exp = item_volume[None, :]
    item_max_dim_exp = item_max_dim[None, :]
    item_min_dim_exp = item_min_dim[None, :]

    # --- Interaction / Fit Features ---
    delta_x = ems_x_len_exp - item_x_len_exp
    delta_y = ems_y_len_exp - item_y_len_exp
    delta_z = ems_z_len_exp - item_z_len_exp
    vol_diff = ems_volume_exp - item_volume_exp
    fits_x = delta_x >= 0
    fits_y = delta_y >= 0
    fits_z = delta_z >= 0
    can_fit = fits_x & fits_y & fits_z

    # --- Item-Specific Features (Using raw values, assuming pre-normalized context) ---
    # These are (1, I) and will be broadcast later or used in feature_stack
    _item_vol = item_volume_exp
    _item_max_dim = item_max_dim_exp
    _item_min_dim = item_min_dim_exp

    # --- EMS-Specific Features (Using raw values) ---
    # These are (E, 1)
    _ems_vol = ems_volume_exp
    _ems_min_dim = ems_min_dim_exp
    _ems_coord_x1 = obs.ems[:, 0:1]
    _ems_coord_y1 = obs.ems[:, 2:3]
    _ems_coord_z1 = obs.ems[:, 4:5]

    return HeuristicFeatures(
        delta_x=delta_x, delta_y=delta_y, delta_z=delta_z, vol_diff=vol_diff, can_fit=can_fit.astype(jnp.float32),
        item_vol=_item_vol, item_max_dim=_item_max_dim, item_min_dim=_item_min_dim,
        ems_vol=_ems_vol, ems_coord_x1=_ems_coord_x1, ems_coord_y1=_ems_coord_y1,
        ems_coord_z1=_ems_coord_z1, ems_min_dim=_ems_min_dim
    )

# --- 2. Heuristic Policy Application (Adjusted for new feature names if any) ---
# Number of features in HeuristicFeatures (still 13)
# Interaction: delta_x, delta_y, delta_z, vol_diff, can_fit (5)
# Item: item_vol, item_max_dim, item_min_dim (3)
# EMS: ems_vol, ems_coord_x1, ems_coord_y1, ems_coord_z1, ems_min_dim (5)
NUM_EXPANDED_HEURISTIC_FEATURES = 13 # Remains 13

def apply_expanded_linear_heuristic(
    features: HeuristicFeatures,
    genome: chex.Array,
    obs_action_mask: chex.Array
) -> chex.Array:
    chex.assert_shape(genome, (NUM_EXPANDED_HEURISTIC_FEATURES,))

    num_ems = features.delta_x.shape[0]
    num_items = features.delta_x.shape[1]

    # Broadcast item-specific and ems-specific features to (E, I) shape for stacking
    f_item_vol = jnp.broadcast_to(features.item_vol, (num_ems, num_items))
    f_item_max_d = jnp.broadcast_to(features.item_max_dim, (num_ems, num_items))
    f_item_min_d = jnp.broadcast_to(features.item_min_dim, (num_ems, num_items))

    f_ems_vol = jnp.broadcast_to(features.ems_vol, (num_ems, num_items))
    f_ems_x1 = jnp.broadcast_to(features.ems_coord_x1, (num_ems, num_items))
    f_ems_y1 = jnp.broadcast_to(features.ems_coord_y1, (num_ems, num_items))
    f_ems_z1 = jnp.broadcast_to(features.ems_coord_z1, (num_ems, num_items))
    f_ems_min_d = jnp.broadcast_to(features.ems_min_dim, (num_ems, num_items))

    # Order of features in stack must match genome weights:
    feature_stack = jnp.stack([
        features.delta_x, features.delta_y, features.delta_z, features.vol_diff, features.can_fit,
        f_item_vol, f_item_max_d, f_item_min_d,
        f_ems_vol, f_ems_x1, f_ems_y1, f_ems_z1, f_ems_min_d
    ], axis=-1)

    scores = jnp.einsum("eif,f->ei", feature_stack, genome)
    masked_scores = jnp.where(obs_action_mask, scores, -jnp.inf)
    return masked_scores

# --- 3. Heuristic Policy Module (Updated to use new functions) ---
class HeuristicPolicy(NamedTuple):
    featurizer_fn: callable = compute_heuristic_features
    heuristic_application_fn: callable = apply_expanded_linear_heuristic
    num_genome_features: int = NUM_EXPANDED_HEURISTIC_FEATURES

    def init_genome(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.uniform(key, shape=(self.num_genome_features,), minval=-1.0, maxval=1.0)

    # Removed container_dims from apply method
    def apply(self, genome: chex.Array, obs: ArrayObservation) -> chex.Array:
        features = self.featurizer_fn(obs) # Pass only obs
        pair_scores = self.heuristic_application_fn(features, genome, obs.action_mask)
        
        original_shape = pair_scores.shape
        num_flat_actions = original_shape[0] * original_shape[1]
        flattened_scores = pair_scores.reshape(num_flat_actions)
        action_probabilities_flat = jax.nn.softmax(flattened_scores, axis=-1)
        return action_probabilities_flat

# --- QDax Policy Adapter --- (Remove container_dims from apply)
class QDaxHeuristicPolicyAdapter:
    def __init__(self, heuristic_policy_instance: HeuristicPolicy, obs_to_array_converter: callable = None):
        self.heuristic_policy = heuristic_policy_instance
        self.obs_to_array_converter = obs_to_array_converter

    def init(self, key: chex.PRNGKey, obs_for_shape: ArrayObservation = None) -> chex.ArrayTree:
        initial_genome = self.heuristic_policy.init_genome(key)
        return {'genome': initial_genome}

    # Removed container_dims from apply method
    def apply(self, policy_params: chex.ArrayTree, observation: ArrayObservation) -> chex.Array:
        if self.obs_to_array_converter:
            array_obs = self.obs_to_array_converter(observation)
        else:
            array_obs = observation
        genome = policy_params['genome']
        return self.heuristic_policy.apply(genome, array_obs) # Call without container_dims

# --- Example Usage (Updated for new features, no container_dims) ---
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    
    dummy_num_ems = 5
    dummy_num_items = 10
    # dummy_container_dims = jnp.array([1.0, 1.0, 1.0]) # Assuming unit container

    # ... (dummy observation creation - same as before)
    dummy_ems_data = jax.random.uniform(key, (dummy_num_ems, 6)) # Values between 0 and 1
    # Ensure x1 < x2 etc. and within [0,1] for a unit container context
    dummy_ems_data = dummy_ems_data.at[:, 0].set(jnp.clip(dummy_ems_data[:, 0] * 0.5, 0, 0.4)) # x1
    dummy_ems_data = dummy_ems_data.at[:, 1].set(jnp.clip(dummy_ems_data[:, 0] + dummy_ems_data[:, 1] * 0.5, 0.1, 0.5)) # x2
    dummy_ems_data = dummy_ems_data.at[:, 2].set(jnp.clip(dummy_ems_data[:, 2] * 0.5, 0, 0.4)) # y1
    dummy_ems_data = dummy_ems_data.at[:, 3].set(jnp.clip(dummy_ems_data[:, 2] + dummy_ems_data[:, 3] * 0.5, 0.1, 0.5)) # y2
    dummy_ems_data = dummy_ems_data.at[:, 4].set(jnp.clip(dummy_ems_data[:, 4] * 0.5, 0, 0.4)) # z1
    dummy_ems_data = dummy_ems_data.at[:, 5].set(jnp.clip(dummy_ems_data[:, 4] + dummy_ems_data[:, 5] * 0.5, 0.1, 0.5)) # z2


    key, subkey = jax.random.split(key); dummy_ems_mask = jax.random.choice(subkey, jnp.array([True, False]), (dummy_num_ems,))
    
    key, subkey = jax.random.split(key); dummy_items_data = jax.random.uniform(subkey, (dummy_num_items, 3)) * 0.5 # Item dims smaller than 0.5
    dummy_items_data = jnp.clip(dummy_items_data, 0.01, 0.5) # Ensure positive small dimensions


    key, subkey = jax.random.split(key); dummy_items_mask = jax.random.choice(subkey, jnp.array([True, False]), (dummy_num_items,))
    key, subkey = jax.random.split(key); dummy_items_placed = jax.random.choice(subkey, jnp.array([True, False]), (dummy_num_items,))
    dummy_items_placed &= dummy_items_mask
    _ems_mask_exp = dummy_ems_mask[:, None]; _item_valid_mask_exp = (dummy_items_mask & ~dummy_items_placed)[None, :]
    pre_action_mask = _ems_mask_exp & _item_valid_mask_exp
    key, subkey = jax.random.split(key); dummy_action_mask = jax.random.choice(subkey, jnp.array([True, False]), (dummy_num_ems, dummy_num_items,))
    dummy_action_mask &= pre_action_mask
    dummy_obs = ArrayObservation(
        ems=dummy_ems_data, ems_mask=dummy_ems_mask, items=dummy_items_data,
        items_mask=dummy_items_mask, items_placed=dummy_items_placed, action_mask=dummy_action_mask
    )

    print("Dummy Observation (sample):")
    print(f"  EMS data sample (first EMS): {dummy_obs.ems[0]}")
    print(f"  Item data sample (first item): {dummy_obs.items[0]}")
    print(f"  Action Mask sum: {dummy_obs.action_mask.sum()}")

    # Test Featurizer (no container_dims)
    features = compute_heuristic_features(dummy_obs)
    print("\nComputed Features (delta_x sample):")
    print(features.delta_x[:2, :3])
    print(f"  Item Volume (sample): {features.item_vol[0, :3]}") # Check raw values
    print(f"  Item Max Dim (sample): {features.item_max_dim[0, :3]}")


    heuristic_policy = HeuristicPolicy()
    key, genome_key = jax.random.split(key)
    genome = heuristic_policy.init_genome(genome_key)
    print(f"\nSample Genome ({genome.shape}): {genome[:5]}...")

    flat_action_probs = heuristic_policy.apply(genome, dummy_obs) # No container_dims
    print(f"\nFlat Action Probabilities ({flat_action_probs.shape}):")
    print(f"Sum of probabilities: {flat_action_probs.sum():.4f}")

    selected_flat_action = jnp.argmax(flat_action_probs)
    print(f"Selected flat action index: {selected_flat_action}")
    
    print("\n--- Further Considerations for QDax Integration ---")
    # ... (same considerations)