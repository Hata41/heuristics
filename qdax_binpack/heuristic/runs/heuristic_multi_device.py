import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import jumanji
from typing import Tuple, Type, Dict, Any, List
import jax
import jax.numpy as jnp
import functools
import os
import time

from qdax.core.distributed_map_elites import DistributedMAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import ExtraScores, Fitness, RNGKey, Descriptor, Genotype
from qdax.utils.metrics import default_qd_metrics, CSVLogger

"""New"""
from flax import linen as nn
from qdax_binpack.behaviours import binpack_descriptor_extraction, compute_heuristic_genome_descriptors
from tqdm import tqdm

# Import from your new heuristic_policies.py
from qdax_binpack.heuristic.utils.heuristic_policies import HeuristicPolicy, QDaxHeuristicPolicyAdapter, ArrayObservation
from qdax_binpack.jumanji_conversion import observation_to_arrays as jumanji_obs_to_array_obs


# Import the user's custom scoring function utility
try:
    from qdax_binpack.qdax_jumanji_utils import jumanji_scoring_function_eval_multiple_envs
except ImportError:
    print("WARNING: qdax_jumanji_utils.py not found. Please ensure it's in your PYTHONPATH.")
    print("Using a placeholder function for jumanji_scoring_function_eval_multiple_envs.")
    def jumanji_scoring_function_eval_multiple_envs(genotypes, key, env, n_eval_envs, episode_length, play_step_fn, descriptor_extractor):
        print("Placeholder jumanji_scoring_function_eval_multiple_envs called")
        a_leaf = jax.tree_util.tree_leaves(genotypes['genome'] if isinstance(genotypes, dict) else genotypes)[0] # Adjusted for genome
        num_genotypes = a_leaf.shape[0]
        fake_fitnesses = jnp.zeros(num_genotypes)
        fake_descriptors = jnp.zeros((num_genotypes, 2))
        fake_extra_scores = {}
        return fake_fitnesses, fake_descriptors, fake_extra_scores


# Setup JAX devices
try:
    devices = jax.devices('gpu')
except RuntimeError:
    print("GPU not found, using CPU.")
    devices = jax.devices('cpu')
num_devices = len(devices)
print(f'Detected the following {num_devices} device(s): {devices}')


# Choose Descriptors : 

# descriptor_type = "genome"
descriptor_type = "proritization"


## Define Hyperparameters
seed = 0
episode_length = 20 # Max steps for a full episode
# For QDax loop (can be different from single episode test)
qdax_batch_size_per_device = 1 # QDax emitter batch size per device
qdax_total_batch_size = qdax_batch_size_per_device * num_devices
num_total_iterations = 1000 # Target total algorithm iterations for QDax
log_period = 1 # Iterations per compiled update_fn call
num_update_calls = num_total_iterations // log_period

iso_sigma = 0.005 # For QDax emitter
line_sigma = 0.05  # For QDax emitter
N_EVAL_ENVS = 10   # For QDax scoring function

## Instantiate the Extended Jumanji environment 
env = jumanji.make('Extended_BinPack-v0')

key = jax.random.key(seed)
key, subkey = jax.random.split(key)
action_spec_val = env.action_spec()

# NEW: Handle different action spaces between BinPack-v2 and Extended_BinPack-v0
is_extended_env_with_rotation = len(action_spec_val.num_values) == 3

if is_extended_env_with_rotation:
    print(f"Running with Extended_BinPack-v0 (with rotations). Action spec: {action_spec_val.num_values}")
    NUM_ORIENTATIONS = action_spec_val.num_values[0].item()
    NUM_ITEMS_PER_ORIENTATION = action_spec_val.num_values[2].item()
    # The total number of "item" choices for the policy's flattened output
    TOTAL_ITEM_CHOICES = NUM_ORIENTATIONS * NUM_ITEMS_PER_ORIENTATION
else:
    print(f"Running with BinPack-v2. Action spec: {action_spec_val.num_values}")
    # Original logic for BinPack-v2 or ExtendedBinPack without rotation
    TOTAL_ITEM_CHOICES = action_spec_val.num_values[1].item()

# --- Instantiate Heuristic Policy for QDax ---
heuristic_policy_logic = HeuristicPolicy() # Default featurizer and application logic
# The adapter's obs_to_array_converter will handle the Jumanji Obs -> ArrayObs conversion
policy_network = QDaxHeuristicPolicyAdapter(
    heuristic_policy_instance=heuristic_policy_logic,
    obs_to_array_converter=jumanji_obs_to_array_obs
)

## play_step_fn (remains largely the same, but uses the new policy_network)
def play_step_fn(env_state, timestep, policy_params, key): # policy_params will be {'genome': genome_array}
    # policy_network.apply will handle observation conversion if configured
    # The input 'timestep.observation' is a Jumanji Observation
    flat_action_probs = policy_network.apply(policy_params, timestep.observation)

    # Action selection (e.g., argmax or sampling)
    flat_action_idx = jnp.argmax(flat_action_probs, axis=-1) # Deterministic

    # NEW: Action construction depends on the environment type
    if is_extended_env_with_rotation:
        # Unpack for 3-component action: (orientation, ems, item)
        # flat_action_probs are for a grid of (ems, item_with_orientation)
        chosen_ems_idx = flat_action_idx // TOTAL_ITEM_CHOICES
        chosen_item_and_orientation_idx = flat_action_idx % TOTAL_ITEM_CHOICES
        
        # Decompose the item_with_orientation index
        orientation_idx = chosen_item_and_orientation_idx // NUM_ITEMS_PER_ORIENTATION
        item_idx = chosen_item_and_orientation_idx % NUM_ITEMS_PER_ORIENTATION

        env_action = jnp.array([orientation_idx, chosen_ems_idx, item_idx], dtype=jnp.int32)
    else:
        # Original logic for 2-component action: (ems, item)
        chosen_ems_idx = flat_action_idx // TOTAL_ITEM_CHOICES
        chosen_item_idx = flat_action_idx % TOTAL_ITEM_CHOICES
        env_action = jnp.array([chosen_ems_idx, chosen_item_idx], dtype=jnp.int32)
        
    next_env_state, next_timestep = env.step(env_state, env_action)
    transition = QDTransition(
        obs=timestep.observation, next_obs=next_timestep.observation, rewards=next_timestep.reward,
        dones=jnp.where(next_timestep.last(), 1.0, 0.0), actions=flat_action_idx,
        truncations=jnp.where(next_timestep.last() & (next_timestep.discount > 0), 1.0, 0.0),
        state_desc=None, next_state_desc=None,
    )
    # IMPORTANT: play_step_fn must manage its key if it uses it for random ops internally.
    # For now, assuming action selection is deterministic (argmax) or key is handled if sampling.
    return next_env_state, next_timestep, policy_params, key, transition


# --- Single Episode Test Run with Heuristic Policy ---
print("\n--- Running Single Episode Test with Heuristic Policy ---")
key_test, key_init_genome, key_env_reset, key_episode = jax.random.split(key, 4)

# 1. Initialize one genome (heuristic parameters)
#    For the adapter, obs_for_shape is optional in init. Let's get a sample Jumanji obs.
single_genome_params = policy_network.init(key_init_genome) # Returns {'genome': genome_array}
print(f"Initialized single heuristic genome params: {jax.tree_map(lambda x: x.shape, single_genome_params)}")
print(f"Genome sample: {single_genome_params['genome'][:4]}...")

## Init population for QDax
key, subkey = jax.random.split(key)
# This will be an array of keys for each genome in the total QDax batch
population_keys = jax.random.split(subkey, num=qdax_total_batch_size)

# For init, QDaxHeuristicPolicyAdapter.init does not need obs_for_shape
# We are vmapping the init function which takes only a key.
init_variables_flat_tree = jax.vmap(policy_network.init)(population_keys)
# init_variables_flat_tree is now a PyTree, e.g., {'genome': array_of_genomes_shape (total_batch, genome_dim)}

# Reshape for distributed MAP-Elites: PyTree leaves to (num_devices, batch_size_per_device, ...)
init_variables_tree = jax.tree_util.tree_map(
    lambda x: jnp.reshape(x, (num_devices, qdax_batch_size_per_device) + x.shape[1:]),
    init_variables_flat_tree
)
print(f"Shape of initial heuristic genomes for QDax (e.g., 'genome' leaf): {jax.tree_map(lambda x: x.shape, init_variables_tree)['genome']}")


if descriptor_type != "genome":
    print("Using binpack_descriptor_extraction for QDax scoring function")

    ## Descriptor extraction & Scoring for QDax
    # The descriptor function now needs the total number of item choices
    descriptor_extraction_fn = functools.partial(binpack_descriptor_extraction, 
                                                num_item_choices_from_spec=TOTAL_ITEM_CHOICES)
    # The scoring_fn_dist is already set up to take policy_params (which will be {'genome': ...})
    scoring_fn_dist_qdax = functools.partial(
        jumanji_scoring_function_eval_multiple_envs, 
        env=env, 
        n_eval_envs=N_EVAL_ENVS,
        episode_length=episode_length, 
        play_step_fn=play_step_fn, 
        descriptor_extractor=descriptor_extraction_fn,
    )
    # Wrapped scoring function for DistributedMAPElites (expects 3 outputs)
    def wrapped_scoring_fn_qdax(genotypes: Genotype, key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores]:
        # 'genotypes' here will be the PyTree from the repertoire, e.g., {'genome': genome_batch}
        fitnesses, descriptors, extra_scores = scoring_fn_dist_qdax(genotypes, key)
        return fitnesses.reshape(-1, 1), descriptors, extra_scores

else:
    print("Using genome based descriptors for QDax scoring function")

    ##### Genome based descriptors
    def dummy_episode_descriptor_extractor(transitions, mask):
        batch_size = jax.tree_leaves(transitions)[0].shape[0]
        return jnp.zeros((batch_size, 2)) # Return dummy shape

    scoring_fn_dist_qdax = functools.partial(
        jumanji_scoring_function_eval_multiple_envs,
        env=env,
        n_eval_envs=N_EVAL_ENVS,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
        descriptor_extractor=dummy_episode_descriptor_extractor, # THIS IS IMPORTANT
    )

    def wrapped_scoring_fn_qdax(genotypes: Genotype, key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores]:
        # 1. Get fitnesses (descriptors from dummy_episode_descriptor_extractor are ignored)
        fitnesses, _ignored_episode_descriptors, extra_scores = scoring_fn_dist_qdax(genotypes, key)
        
        # 2. Compute descriptors based on the genome
        genome_based_descriptors = compute_heuristic_genome_descriptors(genotypes) # Pass the PyTree genotypes
        
        return fitnesses.reshape(-1, 1), genome_based_descriptors, extra_scores


## Emitter & Algorithm for QDax
variation_fn = functools.partial(isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma)
mixing_emitter = MixingEmitter(
    mutation_fn=None, variation_fn=variation_fn, variation_percentage=1.0,
    batch_size=qdax_batch_size_per_device # This is per device for DistributedMAPElites
)

qd_offset = 0.0
metrics_function = functools.partial(default_qd_metrics, qd_offset=qd_offset)
algo_instance = DistributedMAPElites(
    scoring_function=wrapped_scoring_fn_qdax, # Use the QDax specific wrapped one
    emitter=mixing_emitter,
    metrics_function=metrics_function
)

## Centroids & Distributed Init for QDax
key, cvt_key = jax.random.split(key)
centroids = compute_cvt_centroids(num_descriptors=2, num_init_cvt_samples=10000, num_centroids=64, minval=0.0, maxval=1.0, key=cvt_key)
key, init_keys_subkey = jax.random.split(key)
distributed_init_keys = jax.random.split(init_keys_subkey, num=num_devices)
distributed_init_keys = jnp.stack(distributed_init_keys)

# init_variables_tree is already shaped (num_devices, batch_size_per_device, genome_dim)
repertoire, emitter_state, init_metrics_per_device = algo_instance.get_distributed_init_fn(
    centroids=centroids, devices=devices
)(genotypes=init_variables_tree, key=distributed_init_keys) # Pass the PyTree of genomes

## Prepare for Metrics Collection (QDax loop)
all_metrics: Dict[str, List[Any]] = {"qd_score": [], "max_fitness": [], "coverage": []}
logged_init_metrics = {"time": 0.0, "loop": 0, "iteration": 0}
for metric_key, metric_values_all_devices in init_metrics_per_device.items():
    value_from_first_device = jax.tree_util.tree_map(lambda x: x[0], metric_values_all_devices)
    logged_init_metrics[metric_key] = value_from_first_device
    if metric_key in all_metrics: all_metrics[metric_key].append(value_from_first_device)

csv_logger = CSVLogger("distributed_mapelites_heuristic_binpack_logs.csv", header=list(logged_init_metrics.keys()) + ["num_evaluations"])
if "num_evaluations" not in logged_init_metrics: logged_init_metrics["num_evaluations"] = qdax_total_batch_size * N_EVAL_ENVS
csv_logger.log(logged_init_metrics)

## Get pmapped update function for QDax
update_fn = algo_instance.get_distributed_update_fn(num_iterations=log_period, devices=devices)

## Run the QDax optimization loop
print(f"Starting QDax optimization: {num_update_calls} update calls, with {log_period} iterations per call.")
actual_evals_done_total = logged_init_metrics.get("num_evaluations",0)

for i in tqdm(range(num_update_calls), desc="QDax Training Progress"):
    start_time = time.time()
    key, loop_key_subkey = jax.random.split(key)
    distributed_loop_keys = jax.random.split(loop_key_subkey, num_devices)
    distributed_loop_keys = jnp.stack(distributed_loop_keys)

    repertoire, emitter_state, metrics_from_update_per_device = update_fn(
        repertoire, emitter_state, distributed_loop_keys
    )
    
    current_metrics_first_device = jax.tree_util.tree_map(lambda x: x[0], metrics_from_update_per_device)
    timelapse = time.time() - start_time
    current_qd_iteration = (i + 1) * log_period
    
    actual_evals_this_loop = log_period * qdax_total_batch_size * N_EVAL_ENVS
    actual_evals_done_total += actual_evals_this_loop

    logged_metrics_csv = {"time": timelapse, "loop": i + 1, "iteration": current_qd_iteration, "num_evaluations": actual_evals_done_total}
    
    for metric_key, value_array_log_period in current_metrics_first_device.items():
        logged_metrics_csv[metric_key] = value_array_log_period[-1] if value_array_log_period.ndim > 0 and len(value_array_log_period) > 0 else value_array_log_period
        if metric_key in all_metrics: all_metrics[metric_key].extend(list(value_array_log_period))
            
    csv_logger.log(logged_metrics_csv)

print("QDax Training finished.")

# Convert lists in all_metrics to JAX arrays for plotting
for k in list(all_metrics.keys()):
    if isinstance(all_metrics[k], list): all_metrics[k] = jnp.array(all_metrics[k])
    if not all_metrics[k].shape : all_metrics[k] = jnp.array([all_metrics[k]])

## Plotting QDax Results
final_repertoire = jax.tree_util.tree_map(lambda x: x[0], repertoire)
from qdax.utils.plotting import plot_map_elites_results
from matplotlib.pyplot import savefig
os.makedirs("qdax_binpack_heuristic_distributed", exist_ok=True)

actual_iterations_run_qdax = num_update_calls * log_period
plot_x_axis_iterations_qdax = jnp.arange(0, actual_iterations_run_qdax + 1)
expected_metric_array_len_qdax = 1 + actual_iterations_run_qdax

metrics_for_plot_filtered_qdax = {}
for k, v_array in all_metrics.items():
    if hasattr(v_array, '__len__') and len(v_array) == expected_metric_array_len_qdax:
        metrics_for_plot_filtered_qdax[k] = v_array
    elif not hasattr(v_array, '__len__') and expected_metric_array_len_qdax == 1:
        metrics_for_plot_filtered_qdax[k] = jnp.array([v_array])
    else:
        print(f"Plotting Warning (QDax): Metric '{k}' length {len(v_array) if hasattr(v_array, '__len__') else 'scalar'} != expected {expected_metric_array_len_qdax}. Skipping.")

for essential_key in ["qd_score", "max_fitness", "coverage"]:
    if essential_key not in metrics_for_plot_filtered_qdax:
        print(f"Warning (QDax): Essential metric '{essential_key}' missing for plotting. Using zeros.")
        if expected_metric_array_len_qdax > 0: metrics_for_plot_filtered_qdax[essential_key] = jnp.zeros(expected_metric_array_len_qdax)

if expected_metric_array_len_qdax > 0 and \
    all(k in metrics_for_plot_filtered_qdax for k in ["qd_score", "max_fitness", "coverage"]):
    fig, axes = plot_map_elites_results(
        env_steps=plot_x_axis_iterations_qdax,
        metrics=metrics_for_plot_filtered_qdax,
        repertoire=final_repertoire,
        min_descriptor=0.0,
        max_descriptor=1.0,
        x_label="QD Algorithm Iterations (0 = init)"
    )
    plot_filename = "qdax_binpack_heuristic_distributed/repertoire_plot_heuristic_distributed.png"
    savefig(plot_filename)
    print(f"QDax Plot saved to {plot_filename}")
else:
    print("Skipping QDax plotting due to missing essential metrics or zero iterations run.")