import jax
import jax.numpy as jnp
import chex
from typing import NamedTuple, Tuple, Callable

# This remains your primary observation format after conversion from Jumanji
class ArrayObservation(NamedTuple):
    ems: chex.Array
    ems_mask: chex.Array
    items: chex.Array
    items_mask: chex.Array
    items_placed: chex.Array
    action_mask: chex.Array

# --- 1. Expanded Featurizer (V2) ---

class ExpandedHeuristicFeatures(NamedTuple):
    # --- Interaction / Fit Features (E, O, I) ---
    delta_x: chex.Array
    delta_y: chex.Array
    delta_z: chex.Array
    vol_diff: chex.Array
    can_fit: chex.Array  # Binary: 1.0 if it fits, 0.0 otherwise
    
    # --- Alignment & Wasted Space (E, O, I) ---
    # How well item dims align with EMS dims. Range [0, 1]. 1 is perfect alignment.
    alignment_score: chex.Array 
    # Ratio of wasted volume in the EMS if this item is placed. Range [0, 1].
    wasted_vol_ratio: chex.Array

    # --- Item-Specific Features (Broadcasted to (E, O, I)) ---
    item_vol: chex.Array
    item_max_dim: chex.Array
    item_min_dim: chex.Array
    # Ratio of largest face area to item volume (rewards flatter items)
    item_flatness_ratio: chex.Array 
    
    # --- EMS-Specific Features (Broadcasted to (E, O, I)) ---
    ems_vol: chex.Array
    ems_min_dim: chex.Array
    # Features for full-support constraint
    ems_z1: chex.Array  # Lower is more stable
    ems_contact_area: chex.Array # Larger is more stable
    is_on_floor: chex.Array # Binary: 1.0 if EMS is on the container floor


def compute_expanded_heuristic_features(obs: ArrayObservation) -> ExpandedHeuristicFeatures:
    """
    Computes an expanded set of features for heuristic-based bin packing.
    This version is designed for ExtendedBinPack with rotations.
    Assumes obs.items has shape (O, I, 3) where O is orientations and I is items.
    """
    # --- EMS Pre-computation (E-dim) ---
    ems_x_len = obs.ems[:, 1] - obs.ems[:, 0]
    ems_y_len = obs.ems[:, 3] - obs.ems[:, 2]
    ems_z_len = obs.ems[:, 5] - obs.ems[:, 4]
    ems_volume = ems_x_len * ems_y_len * ems_z_len
    ems_dims = jnp.stack([ems_x_len, ems_y_len, ems_z_len], axis=-1)
    ems_min_dim = jnp.min(ems_dims, axis=-1)
    ems_contact_area = ems_x_len * ems_y_len
    ems_z1 = obs.ems[:, 4]
    is_on_floor = (ems_z1 == 0).astype(jnp.float32)

    # --- Item Pre-computation (O, I-dims) ---
    # obs.items has shape (O, I, 3)
    item_dims = obs.items
    item_x_len, item_y_len, item_z_len = item_dims[..., 0], item_dims[..., 1], item_dims[..., 2]
    item_volume = item_x_len * item_y_len * item_z_len
    item_max_dim = jnp.max(item_dims, axis=-1)
    item_min_dim = jnp.min(item_dims, axis=-1)
    face_xy = item_x_len * item_y_len
    face_xz = item_x_len * item_z_len
    face_yz = item_y_len * item_z_len
    item_max_face_area = jnp.maximum(jnp.maximum(face_xy, face_xz), face_yz)
    item_flatness_ratio = item_max_face_area / jnp.maximum(item_volume, 1e-6)

    # --- Expand dims for broadcasting: (E, O, I) ---
    # EMS features from (E,) -> (E, 1, 1)
    ems_x_len_exp = ems_x_len[:, None, None]
    ems_y_len_exp = ems_y_len[:, None, None]
    ems_z_len_exp = ems_z_len[:, None, None]
    ems_vol_exp = ems_volume[:, None, None]

    # Item features from (O, I) -> (1, O, I)
    item_x_len_exp = item_x_len[None, ...]
    item_y_len_exp = item_y_len[None, ...]
    item_z_len_exp = item_z_len[None, ...]
    item_vol_exp = item_volume[None, ...]

    # --- Feature Calculation ---
    # Interaction
    delta_x = ems_x_len_exp - item_x_len_exp
    delta_y = ems_y_len_exp - item_y_len_exp
    delta_z = ems_z_len_exp - item_z_len_exp
    vol_diff = ems_vol_exp - item_vol_exp
    can_fit = (delta_x >= 0) & (delta_y >= 0) & (delta_z >= 0)
    
    # Alignment
    sorted_ems_dims = jnp.sort(ems_dims, axis=-1) # (E, 3)
    sorted_item_dims = jnp.sort(item_dims, axis=-1) # (O, I, 3)
    # alignment = 1 - normalized_l1_distance_between_sorted_dims
    # We compare the ratio vectors [d1/d_max, d2/d_max, d3/d_max]
    norm_ems_dims = sorted_ems_dims / jnp.maximum(sorted_ems_dims[:, -1:], 1e-6)
    norm_item_dims = sorted_item_dims / jnp.maximum(sorted_item_dims[..., -1:], 1e-6)
    alignment_dist = jnp.sum(jnp.abs(norm_ems_dims[:, None, None, :] - norm_item_dims[None, ...]), axis=-1)
    alignment_score = jnp.clip(1.0 - alignment_dist / 2.0, 0.0, 1.0) # Max dist is 2.0

    # Wasted space
    wasted_vol_ratio = jnp.clip(vol_diff / jnp.maximum(ems_vol_exp, 1e-6), 0.0, 1.0)

    # --- Broadcasting for stacking ---
    num_ems, num_orientations, num_items = vol_diff.shape
    def broadcast(arr, is_ems_feature):
        if is_ems_feature: # Shape (E,) -> (E, O, I)
            return jnp.broadcast_to(arr[:, None, None], (num_ems, num_orientations, num_items))
        else: # Shape (O, I) -> (E, O, I)
            return jnp.broadcast_to(arr[None, ...], (num_ems, num_orientations, num_items))

    return ExpandedHeuristicFeatures(
        delta_x=delta_x, delta_y=delta_y, delta_z=delta_z, vol_diff=vol_diff,
        can_fit=can_fit.astype(jnp.float32),
        alignment_score=alignment_score,
        wasted_vol_ratio=wasted_vol_ratio,
        item_vol=broadcast(item_volume, False),
        item_max_dim=broadcast(item_max_dim, False),
        item_min_dim=broadcast(item_min_dim, False),
        item_flatness_ratio=broadcast(item_flatness_ratio, False),
        ems_vol=broadcast(ems_volume, True),
        ems_min_dim=broadcast(ems_min_dim, True),
        ems_z1=broadcast(ems_z1, True),
        ems_contact_area=broadcast(ems_contact_area, True),
        is_on_floor=broadcast(is_on_floor, True),
    )

# --- 2. Heuristic Policy Application (V2) ---
# Number of features in ExpandedHeuristicFeatures
NUM_EXPANDED_HEURISTIC_FEATURES_V2 = 16

def apply_expanded_linear_heuristic_v2(
    features: ExpandedHeuristicFeatures,
    genome: chex.Array,
    obs: ArrayObservation # For masks
) -> chex.Array:
    chex.assert_shape(genome, (NUM_EXPANDED_HEURISTIC_FEATURES_V2,))

    # The order here MUST match the order in ExpandedHeuristicFeatures and the genome
    feature_stack = jnp.stack([
        features.delta_x, features.delta_y, features.delta_z, features.vol_diff, features.can_fit,
        features.alignment_score, features.wasted_vol_ratio,
        features.item_vol, features.item_max_dim, features.item_min_dim, features.item_flatness_ratio,
        features.ems_vol, features.ems_min_dim, features.ems_z1, features.ems_contact_area, features.is_on_floor
    ], axis=-1)

    # Calculate scores for all (E, O, I) pairs
    # This is the core, translatable scoring logic
    scores = jnp.einsum("eoif,f->eoi", feature_stack, genome)

    # In training, we must apply masks BEFORE flattening for softmax
    # The heuristic only considers valid items and valid EMSs
    # The action_mask checks for geometric fit, which `can_fit` also does.
    # We use the env's action_mask as the ground truth for what is possible.
    
    # Get masks with correct dimensions for broadcasting
    # obs.items_mask shape (O, I) -> (1, O, I)
    # obs.ems_mask shape (E,) -> (E, 1, 1)
    # obs.action_mask shape (E, O, I) -> from extended env
    
    # We only score valid items in valid EMSs. The action_mask should already handle this.
    # If an item or EMS is invalid, its action mask row/column will be all False.
    masked_scores = jnp.where(obs.action_mask, scores, -jnp.inf)
    
    return masked_scores

# --- 3. Heuristic Policy Module  ---
class HeuristicPolicy(NamedTuple):
    featurizer_fn: Callable = compute_expanded_heuristic_features
    heuristic_application_fn: Callable = apply_expanded_linear_heuristic_v2
    num_genome_features: int = NUM_EXPANDED_HEURISTIC_FEATURES_V2

    def init_genome(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.uniform(key, shape=(self.num_genome_features,), minval=-1.0, maxval=1.0)

    def apply(self, genome: chex.Array, obs: ArrayObservation) -> chex.Array:
        # 1. Compute features for all pairs
        features = self.featurizer_fn(obs)
        
        # 2. Compute masked scores for all pairs
        pair_scores = self.heuristic_application_fn(features, genome, obs) # Shape (E, O, I)

        # 3. Reshape and flatten for the QDax runner script
        # The runner script expects a flat vector of probabilities for the action (ems, item_with_orientation)
        # TOTAL_ITEM_CHOICES in the script is (Num_Orientations * Num_Items)
        num_ems, num_orientations, num_items = pair_scores.shape
        
        # Reshape to (E, O * I)
        scores_for_runner = pair_scores.reshape(num_ems, num_orientations * num_items)
        
        # Flatten to (E * O * I)
        flattened_scores = scores_for_runner.reshape(-1)
        
        # Softmax over all possible flat actions
        action_probabilities_flat = jax.nn.softmax(flattened_scores, axis=-1)
        
        return action_probabilities_flat

# --- 4. QDax Policy Adapter (V2) ---
# This remains the same, but it will wrap the new HeuristicPolicy
class QDaxHeuristicPolicyAdapter:
    def __init__(self, heuristic_policy_instance: HeuristicPolicy, obs_to_array_converter: Callable):
        self.heuristic_policy = heuristic_policy_instance
        self.obs_to_array_converter = obs_to_array_converter

    def init(self, key: chex.PRNGKey) -> chex.ArrayTree:
        initial_genome = self.heuristic_policy.init_genome(key)
        return {'genome': initial_genome}

    def apply(self, policy_params: chex.ArrayTree, jumanji_observation) -> chex.Array:
        # Convert Jumanji observation to the ArrayObservation format our heuristic expects
        array_obs = self.obs_to_array_converter(jumanji_observation)
        genome = policy_params['genome']
        return self.heuristic_policy.apply(genome, array_obs)