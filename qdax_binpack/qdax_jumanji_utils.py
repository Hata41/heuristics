# qdax_jumanji_utils.py (or whatever you name the file)

from functools import partial
from typing import Any, Callable, Tuple

import flax.linen as nn 
import jax
import jax.numpy as jnp
import jumanji
from chex import ArrayTree 
from typing_extensions import TypeAlias

from qdax.core.neuroevolution.buffers.buffer import QDTransition, Transition
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Params,
    RNGKey,
)

JumanjiState: TypeAlias = ArrayTree
JumanjiTimeStep: TypeAlias = jumanji.types.TimeStep


def generate_jumanji_unroll(
    init_state: JumanjiState,
    init_timestep: JumanjiTimeStep,
    policy_params: Params,
    key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [JumanjiState, JumanjiTimeStep, Params, RNGKey],
        Tuple[
            JumanjiState,
            JumanjiTimeStep,
            Params,
            RNGKey,
            Transition,
        ],
    ],
) -> Tuple[JumanjiState, JumanjiTimeStep, Transition]:
    """
    Generates an episode (a sequence of transitions) for a single policy
    in a single environment instance.
    """

    def _scan_play_step_fn(
        carry: Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey], _unused_arg: Any
    ) -> Tuple[Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey], Transition]:
        env_state_scan, timestep_scan, current_policy_params_scan, current_key_scan = carry
        (
            next_env_state_scan,
            next_timestep_scan,
            _returned_policy_params,
            returned_key_scan,
            transition_output,
        ) = play_step_fn(
            env_state_scan, timestep_scan, current_policy_params_scan, current_key_scan
        )
        next_carry = (
            next_env_state_scan,
            next_timestep_scan,
            current_policy_params_scan,
            returned_key_scan,
        )
        return next_carry, transition_output

    initial_scan_carry = (init_state, init_timestep, policy_params, key)
    final_carry, transitions = jax.lax.scan(
        _scan_play_step_fn,
        initial_scan_carry,
        xs=None,
        length=episode_length,
    )
    final_env_state, final_timestep, _, _ = final_carry
    return final_env_state, final_timestep, transitions


@partial(
    jax.jit,
    static_argnames=(
        "env",
        "n_eval_envs",
        "episode_length",
        "play_step_fn",
        "descriptor_extractor",
    ),
)
def jumanji_scoring_function_eval_multiple_envs(
    policies_params: Genotype,
    eval_batch_key: RNGKey,
    env: jumanji.env.Environment,
    n_eval_envs: int,
    episode_length: int,
    play_step_fn: Callable[
        [JumanjiState, JumanjiTimeStep, Params, RNGKey],
        Tuple[JumanjiState, JumanjiTimeStep, Params, RNGKey, QDTransition],
    ],
    descriptor_extractor: Callable[[QDTransition, jnp.ndarray], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores]:
    """
    Evaluates a batch of policies in parallel using Jumanji environments.
    Each policy is evaluated on `n_eval_envs` newly generated environment
    instances, and the results (fitness, descriptors) are averaged per policy.
    The input `eval_batch_key` is consumed for all stochastic operations within
    this function for the current batch of evaluations.

    Args:
        policies_params: A PyTree of policy parameters. Leaves have a leading
            dimension corresponding to the number of policies.
        eval_batch_key: JAX random key for the entire evaluation batch.
        env: The Jumanji environment instance (used for env.reset).
        n_eval_envs: Number of different environment instances per policy.
        episode_length: Maximum number of steps per episode.
        play_step_fn: Function to play one step in the environment.
        descriptor_extractor: Function to extract behavior descriptors.

    Returns:
        A tuple (fitnesses, descriptors, extra_scores):
            - fitnesses: Averaged fitness per policy (shape: [num_policies]).
            - descriptors: Averaged descriptors per policy 
                           (shape: [num_policies, num_descriptors]).
            - extra_scores: Dictionary, typically including all transitions.
    """
    num_policies = jax.tree.leaves(policies_params)[0].shape[0]
    total_rollouts = num_policies * n_eval_envs

    # Split the eval_batch_key for resets and rollouts within this function
    key_for_resets, key_for_rollouts_pool = jax.random.split(eval_batch_key)
    
    reset_keys_flat = jax.random.split(key_for_resets, total_rollouts)
    rollout_keys_flat = jax.random.split(key_for_rollouts_pool, total_rollouts)

    # Tile/Repeat policy parameters: (P, ...) -> (P*N, ...)
    tiled_policies_params = jax.tree.map(
        lambda x: jnp.repeat(x, n_eval_envs, axis=0), policies_params
    )

    # Reset P*N environments
    vmapped_reset_fn = jax.vmap(env.reset)
    batch_init_states, batch_init_timesteps = vmapped_reset_fn(reset_keys_flat)

    # Partially apply static arguments to the unroll function for vmapping
    unroll_fn_for_vmap = partial(
        generate_jumanji_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )
    
    # Perform all P*N rollouts in parallel
    _final_states, _final_timesteps, all_transitions = jax.vmap(unroll_fn_for_vmap)(
        batch_init_states,
        batch_init_timesteps,
        tiled_policies_params,
        rollout_keys_flat,
    )
    # all_transitions leaves have shape (P*N, episode_length, ...)

    # Create mask for valid steps and calculate fitnesses/descriptors per rollout
    # Ensure dones are float for calculations
    dones_float = all_transitions.dones.astype(jnp.float32)
    is_done_all = jnp.clip(jnp.cumsum(dones_float, axis=1), 0, 1) # (P*N, episode_length)
    
    # Mask for steps that are part of the episode (not after done)
    mask_all = jnp.roll(is_done_all, 1, axis=1) 
    mask_all = mask_all.at[:, 0].set(0.0) # First step is never masked by a previous 'done'

    # Ensure rewards are float
    rewards_float = all_transitions.rewards.astype(jnp.float32) # (P*N, episode_length)
    # Sum rewards over episode length for each of P*N rollouts
    fitnesses_all_rollouts = jnp.sum(rewards_float * (1.0 - mask_all), axis=1) # (P*N,)
    
    # Extract descriptors for each of P*N rollouts
    descriptors_all_rollouts = descriptor_extractor(all_transitions, mask_all) # (P*N, num_descriptors)

    # Reshape and average results per policy
    fitnesses_per_policy_eval = fitnesses_all_rollouts.reshape(
        (num_policies, n_eval_envs)
    ) # (P, N)
    
    num_descriptors = descriptors_all_rollouts.shape[-1]
    descriptors_per_policy_eval = descriptors_all_rollouts.reshape(
        (num_policies, n_eval_envs, num_descriptors)
    ) # (P, N, num_descriptors)

    # Average over the n_eval_envs dimension
    final_fitnesses = jnp.mean(fitnesses_per_policy_eval, axis=1)  # (P,)
    final_descriptors = jnp.mean(
        descriptors_per_policy_eval, axis=1
    )  # (P, num_descriptors)

    # Store all transitions if needed, or other aggregated metrics
    extra_scores = {"transitions_all_rollouts": all_transitions}

    return final_fitnesses, final_descriptors, extra_scores