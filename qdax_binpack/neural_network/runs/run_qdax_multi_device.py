
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"



import jumanji
from typing import Tuple, Type, Dict, Any, List # Added List

import jax
import jax.numpy as jnp
import functools
import os
import time

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.distributed_map_elites import DistributedMAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import ExtraScores, Fitness, RNGKey, Descriptor, Genotype
from qdax.utils.metrics import default_qd_metrics, CSVLogger

"""New"""
from flax import linen as nn
from qdax_binpack.neural_network.utils.nets import BinPackActor, BinPackTorso, BPActorHead, Obs_to_Arrays
from qdax_binpack.behaviours import binpack_descriptor_extraction
from tqdm import tqdm

# Import the user's custom scoring function utility
try:
    from qdax_binpack.qdax_jumanji_utils import jumanji_scoring_function_eval_multiple_envs
except ImportError:
    print("WARNING: qdax_jumanji_utils.py not found. Please ensure it's in your PYTHONPATH.")
    print("Using a placeholder function for jumanji_scoring_function_eval_multiple_envs.")
    def jumanji_scoring_function_eval_multiple_envs(genotypes, key, env, n_eval_envs, episode_length, play_step_fn, descriptor_extractor):
        print("Placeholder jumanji_scoring_function_eval_multiple_envs called")
        # Determine num_genotypes correctly for batched PyTrees
        a_leaf = jax.tree_util.tree_leaves(genotypes)[0]
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

## Define Hyperparameters
seed = 0
episode_length = 20
batch_size_per_device = 1
total_batch_size = batch_size_per_device * num_devices
num_total_iterations = 100 # Target total algorithm iterations
log_period = 1 # Iterations per compiled update_fn call
num_update_calls = num_total_iterations // log_period # Number of times pmapped update_fn is called
iso_sigma = 0.005
line_sigma = 0.05
N_EVAL_ENVS = 1

## Instantiate the Jumanji environment & Policy
# env = jumanji.make('BinPack-v2')
env = jumanji.make('Extended_BinPack-v0') # Use the extended environment

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
    print(f"Running with BinPack-v2 or Extended_BinPack-v0 (no rotations). Action spec: {action_spec_val.num_values}")
    TOTAL_ITEM_CHOICES = action_spec_val.num_values[1].item()

transformer_num_heads, num_transformer_layers, qkv_features = 1, 1, 2
attention_kwargs = dict(num_heads=transformer_num_heads, qkv_features=qkv_features, kernel_init=nn.initializers.ones, bias_init=nn.initializers.zeros)
policy_network = BinPackActor(torso=BinPackTorso(num_transformer_layers=num_transformer_layers, attention_kwargs=attention_kwargs), input_layer=Obs_to_Arrays(), action_head=BPActorHead())

## play_step_fn
def play_step_fn(env_state, timestep, policy_params, key):
    network_input = timestep.observation
    proba_action_flat = policy_network.apply(policy_params, network_input)
    flat_action_idx = jnp.argmax(proba_action_flat, axis=-1)

    # NEW: Action construction depends on the environment type
    if is_extended_env_with_rotation:
        # Unpack for 3-component action: (orientation, ems, item)
        chosen_ems_idx = flat_action_idx // TOTAL_ITEM_CHOICES
        chosen_item_and_orientation_idx = flat_action_idx % TOTAL_ITEM_CHOICES
        
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
    return next_env_state, next_timestep, policy_params, key, transition

## Init population
key, subkey = jax.random.split(key)
population_keys = jax.random.split(subkey, num=total_batch_size)
obs_spec = env.observation_spec()
single_fake_obs = obs_spec.generate_value()
fake_batch_for_init = jax.tree_util.tree_map(lambda x: x[None, ...], single_fake_obs)
init_variables_flat = jax.vmap(policy_network.init, in_axes=(0, None))(population_keys, fake_batch_for_init)
init_variables = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (num_devices, batch_size_per_device) + x.shape[1:]), init_variables_flat)

## Descriptor extraction & Scoring
# The descriptor function now needs the total number of item choices
descriptor_extraction_fn = functools.partial(binpack_descriptor_extraction, num_item_choices_from_spec=TOTAL_ITEM_CHOICES)
scoring_fn_dist = functools.partial(
    jumanji_scoring_function_eval_multiple_envs, env=env, n_eval_envs=N_EVAL_ENVS,
    episode_length=episode_length, play_step_fn=play_step_fn, descriptor_extractor=descriptor_extraction_fn,
)
def wrapped_scoring_fn(genotypes: Genotype, key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores]:
    fitnesses, descriptors, extra_scores = scoring_fn_dist(genotypes, key)
    return fitnesses.reshape(-1, 1), descriptors, extra_scores

## Emitter & Algorithm
variation_fn = functools.partial(isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma)
mixing_emitter = MixingEmitter(mutation_fn=None, variation_fn=variation_fn, variation_percentage=1.0, batch_size=batch_size_per_device)
qd_offset = 0.0
metrics_function = functools.partial(default_qd_metrics, qd_offset=qd_offset)
algo_instance = DistributedMAPElites(scoring_function=wrapped_scoring_fn, emitter=mixing_emitter, metrics_function=metrics_function)

## Centroids & Distributed Init
key, cvt_key = jax.random.split(key)
centroids = compute_cvt_centroids(num_descriptors=2, num_init_cvt_samples=10000, num_centroids=64, minval=0.0, maxval=1.0, key=cvt_key)
key, init_keys_subkey = jax.random.split(key)
distributed_init_keys = jax.random.split(init_keys_subkey, num=num_devices)
distributed_init_keys = jnp.stack(distributed_init_keys)
repertoire, emitter_state, init_metrics_per_device = algo_instance.get_distributed_init_fn(
    centroids=centroids, devices=devices
)(genotypes=init_variables, key=distributed_init_keys)

## Prepare for Metrics Collection
# Initialize all_metrics with empty lists for keys expected from default_qd_metrics
all_metrics: Dict[str, List[Any]] = {
    "qd_score": [], "max_fitness": [], "coverage": []
    # Add other keys here if your metrics_function returns more that you want to plot
}

# Populate initial metrics (from device 0)
logged_init_metrics = {"time": 0.0, "loop": 0, "iteration": 0}
for metric_key, metric_values_all_devices in init_metrics_per_device.items():
    value_from_first_device = metric_values_all_devices[0] # Get value from first device
    logged_init_metrics[metric_key] = value_from_first_device
    if metric_key in all_metrics:
        all_metrics[metric_key].append(value_from_first_device)

csv_logger = CSVLogger("distributed_mapelites_binpack_logs.csv", header=list(logged_init_metrics.keys()) + ["num_evaluations"]) # Adjusted header
if "num_evaluations" not in logged_init_metrics: logged_init_metrics["num_evaluations"] = total_batch_size * N_EVAL_ENVS # Evals for init
csv_logger.log(logged_init_metrics)

## Get pmapped update function
update_fn = algo_instance.get_distributed_update_fn(num_iterations=log_period, devices=devices)

## Run the optimization loop
print(f"Starting {num_update_calls} update calls, with {log_period} iterations per call.")
actual_evals_done_total = logged_init_metrics.get("num_evaluations",0) # Start with init evals

for i in tqdm(range(num_update_calls), desc="QD Training Progress"):
    start_time = time.time()
    key, loop_key_subkey = jax.random.split(key)
    distributed_loop_keys = jax.random.split(loop_key_subkey, num=num_devices)
    distributed_loop_keys = jnp.stack(distributed_loop_keys)

    repertoire, emitter_state, metrics_from_update_per_device = update_fn(
        repertoire, emitter_state, distributed_loop_keys
    )
    
    current_metrics_first_device = jax.tree_util.tree_map(lambda x: x[0], metrics_from_update_per_device)
    timelapse = time.time() - start_time
    current_qd_iteration = (i + 1) * log_period # QD iteration number at the end of this loop
    
    actual_evals_this_loop = log_period * total_batch_size * N_EVAL_ENVS
    actual_evals_done_total += actual_evals_this_loop

    logged_metrics_csv = {"time": timelapse, "loop": i + 1, "iteration": current_qd_iteration, "num_evaluations": actual_evals_done_total}
    
    for metric_key, value_array_log_period in current_metrics_first_device.items():
        logged_metrics_csv[metric_key] = value_array_log_period[-1] if value_array_log_period.ndim > 0 and len(value_array_log_period) > 0 else value_array_log_period
        if metric_key in all_metrics:
            all_metrics[metric_key].extend(list(value_array_log_period))
            
    csv_logger.log(logged_metrics_csv)

print("Training finished.")

# Convert lists in all_metrics to JAX arrays for plotting
for k in list(all_metrics.keys()): # Iterate over copy of keys if modifying dict
    if isinstance(all_metrics[k], list):
        all_metrics[k] = jnp.array(all_metrics[k])
    if not all_metrics[k].shape: # Ensure scalar metrics are at least 1D for plotting consistency
        all_metrics[k] = jnp.array([all_metrics[k]])


## Plotting
final_repertoire = jax.tree_util.tree_map(lambda x: x[0], repertoire)
from qdax.utils.plotting import plot_map_elites_results
from matplotlib.pyplot import savefig
os.makedirs("qdax_binpack_distributed", exist_ok=True)

actual_iterations_run = num_update_calls * log_period
plot_x_axis_iterations = jnp.arange(0, actual_iterations_run + 1)
expected_metric_array_len = 1 + actual_iterations_run

metrics_for_plot_filtered = {}
for k, v_array in all_metrics.items():
    if hasattr(v_array, '__len__') and len(v_array) == expected_metric_array_len:
        metrics_for_plot_filtered[k] = v_array
    elif not hasattr(v_array, '__len__') and expected_metric_array_len == 1:
        metrics_for_plot_filtered[k] = jnp.array([v_array])
    else:
        print(f"Plotting Warning: Metric '{k}' length {len(v_array) if hasattr(v_array, '__len__') else 'scalar'} != expected {expected_metric_array_len}. Skipping.")

for essential_key in ["qd_score", "max_fitness", "coverage"]:
    if essential_key not in metrics_for_plot_filtered:
        print(f"Warning: Essential metric '{essential_key}' missing or mismatched for plotting. Plot may be incomplete or use zeros.")
        if expected_metric_array_len > 0 :
             metrics_for_plot_filtered[essential_key] = jnp.zeros(expected_metric_array_len)


if expected_metric_array_len > 0 and \
   all(k in metrics_for_plot_filtered for k in ["qd_score", "max_fitness", "coverage"]):
    fig, axes = plot_map_elites_results(
        env_steps=plot_x_axis_iterations,
        metrics=metrics_for_plot_filtered,
        repertoire=final_repertoire,
        min_descriptor=0.0,
        max_descriptor=1.0,
        x_label="QD Algorithm Iterations (0 = init)"
    )
    plot_filename = "qdax_binpack_distributed/repertoire_plot_distributed.png"
    savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
else:
    print("Skipping plotting due to missing essential metrics or zero iterations run.")