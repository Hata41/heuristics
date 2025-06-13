import jumanji
from typing import Tuple, Type

import jax
import jax.numpy as jnp
import functools

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import ExtraScores, Fitness, RNGKey, Descriptor, Genotype
from qdax.utils.metrics import default_ga_metrics, default_qd_metrics

"""New"""
from flax import linen as nn
from qdax_binpack.neural_network.utils.nets import BinPackActor, BinPackTorso, BPActorHead, Obs_to_Arrays
from qdax_binpack.behaviours import binpack_descriptor_extraction
from qdax_binpack.qdax_jumanji_utils import jumanji_scoring_function_eval_multiple_envs
from tqdm import tqdm

## Define Hyperparameters
seed = 0
episode_length = 20
population_size = 100 # Increased for a better initial population
batch_size = 32 # Emitter batch size
num_iterations = 50
iso_sigma = 0.005
line_sigma = 0.05
N_EVAL_ENVS = 5 # Number of evaluations per policy

## Instantiate the Jumanji environment
# env = jumanji.make('BinPack-v2')
env = jumanji.make('Extended_BinPack-v0')

key = jax.random.key(seed)
action_spec_val = env.action_spec()

# NEW: Handle different action spaces
is_extended_env_with_rotation = len(action_spec_val.num_values) == 3

if is_extended_env_with_rotation:
    print(f"Running with Extended_BinPack-v0 (with rotations). Action spec: {action_spec_val.num_values}")
    NUM_ORIENTATIONS = action_spec_val.num_values[0].item()
    NUM_ITEMS_PER_ORIENTATION = action_spec_val.num_values[2].item()
    TOTAL_ITEM_CHOICES = NUM_ORIENTATIONS * NUM_ITEMS_PER_ORIENTATION
else:
    print(f"Running with BinPack-v2 or Extended_BinPack-v0 (no rotations). Action spec: {action_spec_val.num_values}")
    TOTAL_ITEM_CHOICES = action_spec_val.num_values[1].item()

## Define Policy Network
transformer_num_heads = 1
num_transformer_layers = 1
qkv_features = 2
attention_kwargs = dict(
    num_heads=transformer_num_heads,
    qkv_features=qkv_features,
    kernel_init=nn.initializers.ones,
    bias_init=nn.initializers.zeros,
)
policy_network = BinPackActor(
    torso=BinPackTorso(num_transformer_layers=num_transformer_layers, attention_kwargs=attention_kwargs),
    input_layer=Obs_to_Arrays(),
    action_head=BPActorHead(),
)

## Define the function to play a step in the environment
def play_step_fn(env_state, timestep, policy_params, key):
    network_input = timestep.observation
    proba_action_flat = policy_network.apply(policy_params, network_input)
    flat_action_idx = jnp.argmax(proba_action_flat, axis=-1)

    # NEW: Action construction depends on the environment type
    if is_extended_env_with_rotation:
        chosen_ems_idx = flat_action_idx // TOTAL_ITEM_CHOICES
        chosen_item_and_orientation_idx = flat_action_idx % TOTAL_ITEM_CHOICES
        orientation_idx = chosen_item_and_orientation_idx // NUM_ITEMS_PER_ORIENTATION
        item_idx = chosen_item_and_orientation_idx % NUM_ITEMS_PER_ORIENTATION
        env_action = jnp.array([orientation_idx, chosen_ems_idx, item_idx], dtype=jnp.int32)
    else:
        chosen_ems_idx = flat_action_idx // TOTAL_ITEM_CHOICES
        chosen_item_idx = flat_action_idx % TOTAL_ITEM_CHOICES
        env_action = jnp.array([chosen_ems_idx, chosen_item_idx], dtype=jnp.int32)

    next_state, next_timestep = env.step(env_state, env_action)
    transition = QDTransition(
        obs=timestep.observation,
        next_obs=next_timestep.observation,
        rewards=next_timestep.reward,
        dones=jnp.where(next_timestep.last(), 1.0, 0.0),
        actions=flat_action_idx,
        truncations=jnp.where(next_timestep.last() & (next_timestep.discount > 0), 1.0, 0.0),
        state_desc=None,
        next_state_desc=None,
    )
    return next_state, next_timestep, policy_params, key, transition

## Initialize a population of policies
key, subkey = jax.random.split(key)
population_keys = jax.random.split(subkey, num=population_size)

obs_spec = env.observation_spec()
single_fake_obs = obs_spec.generate_value()
fake_batch_for_init = jax.tree_util.tree_map(lambda x: x[None, ...], single_fake_obs)

init_variables = jax.vmap(policy_network.init, in_axes=(0, None))(
    population_keys, fake_batch_for_init
)

## Define descriptor extraction and scoring function
descriptor_extraction_fn = functools.partial(
    binpack_descriptor_extraction,
    num_item_choices_from_spec=TOTAL_ITEM_CHOICES, # Pass the correct total
)

# Use the robust jumanji_scoring_function_eval_multiple_envs
scoring_fn = functools.partial(
    jumanji_scoring_function_eval_multiple_envs,
    env=env,
    n_eval_envs=N_EVAL_ENVS,
    episode_length=episode_length,
    play_step_fn=play_step_fn,
    descriptor_extractor=descriptor_extraction_fn,
)

# Wrapper for MAP-Elites which expects (fitness, descriptor, extra_scores)
def map_elites_scoring_function(
    genotypes: Genotype, key: RNGKey
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    fitnesses, descriptors, extra_scores = scoring_fn(genotypes, key)
    return fitnesses.reshape(-1, 1), descriptors, extra_scores

## Define emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)
mixing_emitter = MixingEmitter(
    mutation_fn=None, variation_fn=variation_fn, variation_percentage=1.0, batch_size=batch_size
)

## Define and instantiate MAP-Elites
metrics_function = functools.partial(default_qd_metrics, qd_offset=0)
algo_instance = MAPElites(
    scoring_function=map_elites_scoring_function,
    emitter=mixing_emitter,
    metrics_function=metrics_function,
)

# Compute the centroids for the repertoire
key, cvt_key = jax.random.split(key)
centroids = compute_cvt_centroids(
    num_descriptors=2,
    num_init_cvt_samples=10000,
    num_centroids=64,
    minval=0.0,
    maxval=1.0,
    key=cvt_key,
)

# Initialize the repertoire and emitter state
key, subkey = jax.random.split(key)
repertoire, emitter_state, init_metrics = algo_instance.init(init_variables, centroids, subkey)
print("Initial QD Score: ", init_metrics["qd_score"])
print("Initial Max Fitness: ", init_metrics["max_fitness"])

## Run the optimization loop
print(f"Starting MAP-Elites optimization for {num_iterations} iterations...")
all_metrics = {k: [v] for k, v in init_metrics.items()}

pbar = tqdm(range(num_iterations))
for i in pbar:
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, metrics = algo_instance.update(
        repertoire, emitter_state, subkey
    )
    # Store metrics
    for k, v in metrics.items():
        all_metrics[k].append(v)
    pbar.set_description(f"Iter {i+1}/{num_iterations} | QD Score: {metrics['qd_score']:.2f} | Max Fitness: {metrics['max_fitness']:.2f}")

print("MAP-Elites training finished.")

# Prepare metrics for plotting
metrics_for_plot = {k: jnp.array(v) for k, v in all_metrics.items()}

## Plotting
from qdax.utils.plotting import plot_map_elites_results
from matplotlib.pyplot import savefig
import os

os.makedirs("qdax_binpack", exist_ok=True)

# create the x-axis array (iterations)
iterations_axis = jnp.arange(num_iterations + 1) # +1 for the init step

# create the plots and the grid
fig, axes = plot_map_elites_results(
    env_steps=iterations_axis,
    metrics=metrics_for_plot,
    repertoire=repertoire,
    min_descriptor=0.0,
    max_descriptor=1.0,
    x_label="QD Algorithm Iterations"
)
plot_filename = "qdax_binpack/repertoire_plot_single_device.png"
savefig(plot_filename)
print(f"Plot saved to {plot_filename}")