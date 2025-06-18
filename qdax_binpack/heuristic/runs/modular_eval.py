from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

@dataclass(frozen=True)
class Config:
    """Configuration for the hierarchical deterministic evaluation with ExtendedBinPack."""
    # --- Hardware & Parallelism Configuration ---
    N_DEVICES: int = 10
    # --- Experiment Grid Configuration ---
    N_POLICIES: int = 10
    N_ENVS: int = 1000
    # --- Environment & Episode Configuration ---
    ENV_ID: str = "Extended_BinPack-v0"
    # --- Reproducibility ---
    SEED: int = 42
    # --- Rounds-based Evaluation ---
    EVALUATION_ROUNDS: Optional[int] = None
    MAX_ENVS_PER_ROUND: Optional[int] = 100
    # --- Results Saving ---
    RESULTS_FILENAME: Optional[str] = "evaluation_results.csv"

CONFIG = Config()

import os
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={CONFIG.N_DEVICES}"

import time
import chex
import jax
import jax.numpy as jnp
import jumanji
from colorama import Fore, Style
from jumanji.env import State as JumanjiState
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
from flax import linen as nn
from flax.struct import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm

@dataclass
class Observation:
    items: chex.ArrayTree; items_mask: chex.Array; items_placed: chex.Array
    action_mask: chex.Array; step_count: chex.Array

@dataclass
class AdapterState:
    env_state: JumanjiState; step_count: chex.Array

class ParametricHeuristicPolicy(nn.Module):
    action_spec_num_values: chex.Array
    @nn.compact
    def __call__(self, observation: Observation) -> chex.Array:
        volume_weight = self.param('volume_weight', nn.initializers.uniform(), ())
        complexity_weight = self.param('complexity_weight', nn.initializers.uniform(), ())
        items = observation.items
        item_volumes = items.x_len * items.y_len * items.z_len
        item_complexity = 1.0 / (item_volumes + 1e-6)
        action_mask = observation.action_mask
        volume_scores = jnp.broadcast_to(item_volumes[jnp.newaxis, :], action_mask.shape)
        complexity_scores = jnp.broadcast_to(item_complexity[jnp.newaxis, :], action_mask.shape)
        scores = (volume_weight * volume_scores + complexity_weight * complexity_scores)
        final_scores = jnp.where(action_mask, scores, -jnp.inf)
        flat_action_idx = jnp.argmax(final_scores.flatten(), axis=-1)
        _, num_items = self.action_spec_num_values[1], self.action_spec_num_values[2]
        ems_idx, combined_item_idx = jnp.unravel_index(flat_action_idx, action_mask.shape)
        orientation_idx = combined_item_idx // num_items
        item_idx = combined_item_idx % num_items
        return jnp.array([orientation_idx, ems_idx, item_idx], dtype=jnp.int32)

def run_evaluation(env: jumanji.env) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array]]:
    n_devices = jax.local_device_count()
    total_runs = CONFIG.N_POLICIES * CONFIG.N_ENVS
    action_spec_num_values = env.action_spec().num_values

    print(f"{Style.BRIGHT}{Fore.CYAN}--- Starting Hierarchical Deterministic Benchmark ---{Style.RESET_ALL}")
    print(f"  - Policy: Parametric Heuristic (Flax Module)")
    print(f"  - Environment: {CONFIG.ENV_ID}")
    print(f"  - Policies to Evaluate: {CONFIG.N_POLICIES}")
    print(f"  - Environment Instances per Policy: {CONFIG.N_ENVS}")
    print(f"  - Total Runs (Episodes): {total_runs:,}")
    print(f"  - Devices: {n_devices}")

    if CONFIG.EVALUATION_ROUNDS and CONFIG.MAX_ENVS_PER_ROUND:
        raise ValueError("Please specify either EVALUATION_ROUNDS or MAX_ENVS_PER_ROUND, not both.")
    if CONFIG.EVALUATION_ROUNDS:
        num_rounds = CONFIG.EVALUATION_ROUNDS
    elif CONFIG.MAX_ENVS_PER_ROUND:
        num_rounds = math.ceil(total_runs / CONFIG.MAX_ENVS_PER_ROUND)
    else:
        num_rounds = 1

    if total_runs % num_rounds != 0:
        raise ValueError(f"Total runs ({total_runs}) must be divisible by the number of rounds ({num_rounds}).")
    runs_per_round = total_runs // num_rounds
    if runs_per_round % n_devices != 0:
        raise ValueError(f"Runs per round ({runs_per_round}) must be divisible by the number of devices ({n_devices}).")
    runs_per_device_per_round = runs_per_round // n_devices

    print(f"  - Evaluation Rounds: {num_rounds}")
    print(f"  - Runs per Round (Total): {runs_per_round:,}")
    print(f"  - Runs per Device per Round: {runs_per_device_per_round:,}")
    print("-" * 50)

    main_key = jax.random.PRNGKey(CONFIG.SEED)
    policy = ParametricHeuristicPolicy(action_spec_num_values=action_spec_num_values)
    init_key, env_keys_key = jax.random.split(main_key)
    _, dummy_timestep = env.reset(init_key)
    policy_keys = jax.random.split(init_key, CONFIG.N_POLICIES)
    vmapped_init = jax.vmap(policy.init, in_axes=(0, None))
    policy_params = vmapped_init(policy_keys, dummy_timestep.observation)['params']
    
    env_instance_keys = jax.random.split(env_keys_key, CONFIG.N_ENVS)
    params_for_all_runs = jax.tree.map(lambda x: jnp.repeat(x, CONFIG.N_ENVS, axis=0), policy_params)
    keys_for_all_runs = jnp.tile(env_instance_keys, (CONFIG.N_POLICIES, 1))

    ### MODIFICATION START: Replaced `scan` logic with `while_loop`. ###
    def _evaluate_on_device(params_on_device: Dict[str, chex.Array], keys_on_device: chex.PRNGKey) -> Dict[str, chex.Array]:
        def _run_episode_for_env_instance(params: Dict, env_key: chex.PRNGKey) -> Dict:
            # Initialize the environment for this specific run.
            env_state, timestep = env.reset(env_key)
            
            # Define the initial state for the while_loop.
            # loop_state: (env_state, timestep, cumulative_reward)
            initial_loop_state = (env_state, timestep, 0.0)

            # Define the condition function for the while_loop.
            # Continue looping as long as the episode is not done.
            def cond_fun(loop_state: Tuple[AdapterState, TimeStep, float]) -> bool:
                _, timestep, _ = loop_state
                return ~timestep.last()

            # Define the body function for the while_loop.
            # This function executes one step in the environment.
            def body_fun(loop_state: Tuple[AdapterState, TimeStep, float]) -> Tuple[AdapterState, TimeStep, float]:
                current_state, current_timestep, cumulative_reward = loop_state
                action = policy.apply({'params': params}, current_timestep.observation)
                next_state, next_timestep = env.step(current_state, action)
                cumulative_reward += next_timestep.reward
                return next_state, next_timestep, cumulative_reward

            # Run the episode until completion.
            _, _, final_reward = jax.lax.while_loop(cond_fun, body_fun, initial_loop_state)
            
            return {"episode_return": final_reward}
        
        # vmap the episode runner over the batch of params and keys for this device.
        vmapped_runner = jax.vmap(_run_episode_for_env_instance, in_axes=(0, 0))
        return vmapped_runner(params_on_device, keys_on_device)
    ### MODIFICATION END ###

    @jax.pmap
    def pmapped_evaluation(params: Dict, evaluation_keys: chex.PRNGKey) -> chex.Array:
        return _evaluate_on_device(params, evaluation_keys)

    start_time = time.time()
    all_round_results_flat = []
    for i in tqdm(range(num_rounds), desc="Evaluation Rounds"):
        start_idx = i * runs_per_round
        end_idx = start_idx + runs_per_round
        
        params_for_this_round = jax.tree.map(lambda x: x[start_idx:end_idx], params_for_all_runs)
        keys_for_this_round = keys_for_all_runs[start_idx:end_idx]

        params_for_pmap = jax.tree.map(lambda x: x.reshape(n_devices, runs_per_device_per_round, *x.shape[1:]), params_for_this_round)
        keys_for_pmap = keys_for_this_round.reshape(n_devices, runs_per_device_per_round, -1)

        results_devices_this_round = pmapped_evaluation(params_for_pmap, keys_for_pmap)
        
        results_flat_this_round = jax.tree.map(lambda x: x.flatten(), results_devices_this_round)
        all_round_results_flat.append(results_flat_this_round)
    
    final_flat_results = jax.tree.map(lambda *x: jnp.concatenate(x), *all_round_results_flat)
    final_results = jax.tree.map(lambda x: x.reshape(CONFIG.N_POLICIES, CONFIG.N_ENVS), final_flat_results)
    
    jax.block_until_ready(final_results)
    duration = time.time() - start_time
    
    print(f"\n{Style.BRIGHT}{Fore.CYAN}--- Evaluation Complete ---{Style.RESET_ALL}")
    print(f"  - Total time: {duration:.2f} seconds")
    print(f"  - Throughput: {total_runs / duration:,.2f} episodes/sec")

    return final_results, policy_params

def save_results_to_dataframe(
    episode_returns: chex.Array, policy_params: Dict, config: Config
) -> None:
    policy_ids = np.repeat(np.arange(config.N_POLICIES), config.N_ENVS)
    env_ids = np.tile(np.arange(config.N_ENVS), config.N_POLICIES)
    data = {
        'policy_id': policy_ids,
        'env_id': env_ids,
        'episode_return': np.array(episode_returns).flatten(),
    }
    for param_name, param_values in policy_params.items():
        data[f'param_{param_name}'] = np.repeat(param_values, config.N_ENVS)
    df = pd.DataFrame(data)
    df.to_csv(config.RESULTS_FILENAME, index=False)
    print(f"\n{Style.BRIGHT}{Fore.BLUE}--- Results Saved ---{Style.RESET_ALL}")
    print(f"  - Data saved to '{config.RESULTS_FILENAME}'")

def main() -> None:
    env = jumanji.make(CONFIG.ENV_ID, is_rotation_allowed=True)
    
    class JumanjiAdapter(Wrapper):
        def reset(self, key: chex.PRNGKey) -> Tuple[AdapterState, TimeStep]:
            env_state, timestep = self._env.reset(key)
            obs = Observation(items=timestep.observation.items, items_mask=timestep.observation.items_mask,
                              items_placed=timestep.observation.items_placed, action_mask=timestep.observation.action_mask,
                              step_count=jnp.array(0, dtype=jnp.int32))
            return AdapterState(env_state, jnp.array(0, dtype=jnp.int32)), timestep.replace(observation=obs)

        def step(self, state: AdapterState, action: chex.Array) -> Tuple[AdapterState, TimeStep]:
            env_state, timestep = self._env.step(state.env_state, action)
            new_step_count = state.step_count + 1
            obs = Observation(items=timestep.observation.items, items_mask=timestep.observation.items_mask,
                              items_placed=timestep.observation.items_placed, action_mask=timestep.observation.action_mask,
                              step_count=new_step_count)
            return AdapterState(env_state, new_step_count), timestep.replace(observation=obs)

    env = JumanjiAdapter(env)
    results, policy_params = run_evaluation(env)
    episode_returns = results["episode_return"]
    mean_returns = episode_returns.mean(axis=1)
    std_returns = episode_returns.std(axis=1)

    print(f"\n{Style.BRIGHT}{Fore.GREEN}--- Results Summary ---{Style.RESET_ALL}")
    print(f"{'Policy #':<10} | {'Mean Return':<15} | {'Std Dev':<15} | {'Params (Vol, Comp)':<25}")
    print("-" * 70)
    for i in range(CONFIG.N_POLICIES):
        vol_w = policy_params['volume_weight'][i]
        comp_w = policy_params['complexity_weight'][i]
        params_str = f"({vol_w:.2f}, {comp_w:.2f})"
        print(f"{i:<10} | {mean_returns[i]:<15.2f} | {std_returns[i]:<15.2f} | {params_str:<25}")
    
    if CONFIG.RESULTS_FILENAME:
        save_results_to_dataframe(episode_returns, policy_params, CONFIG)

if __name__ == "__main__":
    main()