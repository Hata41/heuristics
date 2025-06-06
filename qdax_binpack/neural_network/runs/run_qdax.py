import jumanji
from typing import Tuple, Type

import jax
import jax.numpy as jnp
import functools

from qdax.baselines.genetic_algorithm import GeneticAlgorithm
from qdax.core.map_elites import MAPElites
from qdax.core.distributed_map_elites import DistributedMAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids, compute_cvt_centroids
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.jumanji_envs import jumanji_scoring_function
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.custom_types import ExtraScores, Fitness, RNGKey, Descriptor
from qdax.utils.metrics import default_ga_metrics, default_qd_metrics

"""New"""
from flax import linen as nn
from qdax_binpack.neural_network.utils.nets import BinPackActor, BinPackTorso, BPActorHead, Obs_to_Arrays
from behaviours import binpack_descriptor_extraction
from tqdm import tqdm

## Define Hyperparameters

seed = 0
episode_length = 20
population_size = 2
batch_size = population_size

num_iterations = 15

iso_sigma = 0.005
line_sigma = 0.05


## Instantiate the Jumanji environment 

# Instantiate a Jumanji environment using the registry

## Instantiate the Jumanji environment 
env = jumanji.make('BinPack-v2')

# Reset your (jit-able) environment
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
state, timestep = jax.jit(env.reset)(subkey)

# Interact with the (jit-able) environment
action_spec_val = env.action_spec() # Get it once
action = action_spec_val.generate_value()
state, timestep = jax.jit(env.step)(state, action)

# Get number of actions and item choices
num_actions = action_spec_val.num_values.prod().item() # Better way for MultiDiscrete

NUM_ITEM_CHOICES = action_spec_val.num_values[1].item() # e.g., 20

transformer_num_heads = 1
num_transformer_layers = 1
qkv_features = 2
policy_hidden_layer_sizes = (qkv_features)
action_dim = num_actions

attention_kwargs = dict(
        num_heads=transformer_num_heads,
        qkv_features=qkv_features,
        kernel_init=nn.initializers.ones,
        bias_init=nn.initializers.zeros
)

policy_network = BinPackActor(
    torso=BinPackTorso(
            num_transformer_layers = num_transformer_layers,
            attention_kwargs = attention_kwargs),
    input_layer=Obs_to_Arrays(),
    action_head=BPActorHead()
    )

## Utils to interact with the environment
#Define a way to process the observation and define a way to play a step in the environment, given the parameters of a policy_network.

def observation_processing(observation):
    return observation

def play_step_fn(
    env_state,
    timestep,
    policy_params,
    key,
):
    network_input = observation_processing(timestep.observation)
    proba_action_flat = policy_network.apply(policy_params, network_input)
    flat_action_idx = jnp.argmax(proba_action_flat, axis=-1)
    
    # Use the globally defined NUM_ITEM_CHOICES
    chosen_ems_idx = flat_action_idx // NUM_ITEM_CHOICES
    chosen_item_idx = flat_action_idx % NUM_ITEM_CHOICES
    
    env_action = jnp.array([chosen_ems_idx, chosen_item_idx], dtype=jnp.int32)
    
    state_desc = None
    next_state, next_timestep = env.step(env_state, env_action)

    next_state_desc = None

    transition = QDTransition(
        obs=timestep.observation,
        next_obs=next_timestep.observation,
        rewards=next_timestep.reward,
        dones=jnp.where(next_timestep.last(), jnp.array(1.0), jnp.array(0.0)),
        actions=flat_action_idx,
        truncations=jnp.array(0.0),
        state_desc=state_desc,
        next_state_desc=next_state_desc,
    )

    return next_state, next_timestep, policy_params, key, transition
## Init a population of policies
#Also init init states and timesteps


# Init population of controllers
key, subkey = jax.random.split(key)
# Create one PRNG key for each policy in the population
population_keys = jax.random.split(subkey, num=population_size) # Changed from batch_size to population_size

# compute observation size from observation spec (not needed for descriptor anymore)
obs_spec = env.observation_spec()

# Generate a fake batch for a SINGLE instance to get shapes for init
# Your policy_network.init expects a batch dimension for the observation.
# So, fake_batch should have a leading dimension of 1.
single_fake_obs = obs_spec.generate_value()
fake_batch_for_init = jax.tree_util.tree_map(lambda x: x[None, ...], single_fake_obs)

# Initialize a population of policy networks using jax.vmap
# The vmap is over the keys. The fake_batch_for_init is broadcasted.
init_variables = jax.vmap(policy_network.init, in_axes=(0, None))(
    population_keys,
    fake_batch_for_init
)
# Now, init_variables will be a PyTree where each leaf parameter
# has a leading dimension of `population_size`.

# Create the initial environment states
key, subkey = jax.random.split(key)
# Keys for resetting the environment, one for each member of the population
env_reset_keys = jax.random.split(subkey, num=population_size) # Match population_size
reset_fn = jax.jit(jax.vmap(env.reset))

init_states, init_timesteps = reset_fn(env_reset_keys)

## Define a method to extract descriptor when relevant

descriptor_extraction_fn = functools.partial(
    binpack_descriptor_extraction,
    num_item_choices_from_spec=NUM_ITEM_CHOICES # Pass the Python int
)

from qdax_jumanji_utils import jumanji_scoring_function_eval_multiple_envs # Import the new function
N_EVAL_ENVS = 1
scoring_fn = functools.partial(
    jumanji_scoring_function_eval_multiple_envs,
    env=env,                             # Pass the env instance (static)
    n_eval_envs=N_EVAL_ENVS,             # Pass num envs per policy (static)
    episode_length=episode_length,       # Pass episode length (static)
    play_step_fn=play_step_fn,           # Pass play_step_fn (static)
    descriptor_extractor=descriptor_extraction_fn, # Pass descriptor_extractor (static)
)

# The scoring_fn definition remains the same, as it takes descriptor_extractor as an argument
# scoring_fn = functools.partial(
#     jumanji_scoring_function,
#     init_states=init_states,
#     init_timesteps=init_timesteps,
#     episode_length=episode_length,
#     play_step_fn=play_step_fn,
#     descriptor_extractor=descriptor_extraction_fn, # This now uses your new function
# )

## Define Scoring Function
def scoring_function(
    genotypes: jnp.ndarray, key: RNGKey
) -> Tuple[Fitness, ExtraScores, RNGKey]:
    fitnesses, _, extra_scores = scoring_fn(genotypes, key)
    return fitnesses.reshape(-1, 1), extra_scores

##  Define emitter
variation_fn = functools.partial(
    isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
)
mixing_emitter = MixingEmitter(
    mutation_fn=None,
    variation_fn=variation_fn,
    variation_percentage=1.0,
    batch_size=batch_size
)

## Define the algorithm used and apply the initial step
#One can either use a simple genetic algorithm or use MAP-Elites.

use_map_elites = True

if not use_map_elites:
    algo_instance = GeneticAlgorithm(
        scoring_function=scoring_function,
        emitter=mixing_emitter,
        metrics_function=default_ga_metrics,
    )

    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algo_instance.init(
        init_variables, population_size, subkey
    )

else:
    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=0,
    )

    ## Instantiate MAP-Elites
    algo_instance = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )
    
    # ## Instantiate Distributed MAP-Elites
    # algo_instance = DistributedMAPElites(
    # scoring_function=scoring_function,
    # emitter=mixing_emitter,
    # metrics_function=metrics_function,
    # )
    
    # # Compute the centroids
    # centroids = compute_euclidean_centroids(
    #     grid_shape=(10, 10),
    #     minval=0.0,
    #     maxval=1.0,
    # )
    
    #############
    ## This uses the Vornoi Tessalaiton and not a simple grid
    cvt_key = jax.random.key(seed)

    # Compute the centroids
    centroids = compute_cvt_centroids(
    num_descriptors = 2,
    num_init_cvt_samples= 100,
    num_centroids =64,
    minval= 0.0,
    maxval= 1.0,
    key = cvt_key,
    )
    
    #############

    # Compute initial repertoire and emitter state
    key, subkey = jax.random.split(key)
    repertoire, emitter_state, init_metrics = algo_instance.init(init_variables, centroids, subkey)
    
## Run the optimization loop
# Run the algorithm

(repertoire, emitter_state, key,), metrics = jax.lax.scan(
    algo_instance.scan_update,
    (repertoire, emitter_state, key),
    (),
    length=num_iterations,
)

## Plotting

from qdax.utils.plotting import plot_map_elites_results
from matplotlib.pyplot import savefig

# create the x-axis array
env_steps = jnp.arange(num_iterations) * episode_length * batch_size

# create the plots and the grid
fig, axes = plot_map_elites_results(
    env_steps=env_steps, 
    metrics=metrics, 
    repertoire=repertoire, 
    min_descriptor=0.0, 
    max_descriptor=1.0
)
savefig("qdax_binpack/repertoire_plot.png")

