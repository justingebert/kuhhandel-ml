import argparse
import functools
import multiprocessing
from pathlib import Path
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.maskable import distributions as maskable_dist

from rl.env import KuhhandelEnv
from rl.model_agent import ModelAgent
from tests.demo_game import RandomAgent

from rl.train_selfplay import robust_apply_masking

original_apply_masking = maskable_dist.MaskableCategorical.apply_masking #Simplex error fix
maskable_dist.MaskableCategorical.apply_masking = robust_apply_masking

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "kuhhandel_ppo_latest"



def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

def opponent_generator_func(opponent_type, opponent_model_path, env_ref, n):
    """
    Module-level opponent generator function (picklable for multiprocessing).
    Uses fixed parameters for opponent_type and opponent_model_path.
    """
    return create_specific_opponents(n, env_ref, opponent_type, opponent_model_path)

def create_specific_opponents(n_opponents, env_ref, opponent_type, opponent_model_path):
    opponents = []
    
    for i in range(n_opponents):
        if opponent_type == "random":
            opponents.append(RandomAgent(f"Random_{i}"))
        elif opponent_type == "model":
            # Each subprocess loads the model once (shared within subprocess)
            opponents.append(ModelAgent(f"Opponent_{i}", opponent_model_path, env=env_ref))
    
    return opponents

def make_eval_env(rank, opponent_type, opponent_model_path):
    def _init():
        env = KuhhandelEnv(num_players=3)
        
        # Create opponent generator using functools.partial (picklable)
        env.opponent_generator = functools.partial(
            opponent_generator_func,
            opponent_type,
            opponent_model_path
        )
        env = ActionMasker(env, mask_valid_action)
        return env
    
    return _init



def evaluate_fair(model_a_path, model_b_path, n_total_games):
    """
    Evaluates Model A vs Model B by running both configurations in parallel.
    
    Creates mixed environments:
    - Half: A as P0 vs 2xB
    - Half: B as P0 vs 2xA
    
    Both run simultaneously, eliminating sequential overhead and cache misses.
    """
    
    n_half = n_total_games // 2
    total_envs = min(n_total_games, multiprocessing.cpu_count(), 16)
    
    # Split environments 40/60: Config2 needs more resources (2x model opponents)
    if model_a_path:
        n_envs_config1 = int(total_envs * 0.4)  # A vs Random (faster)
    else:
        n_envs_config1 = int(total_envs / 2)  # Random vs A (slower, 2x models in opponents)
    
    print("\n" + "="*40)
    print(f"FAIR EVALUATION ({n_total_games} total games)")
    print("="*40)
    print(f"Model A: {model_a_path if model_a_path else 'Random'}")
    print(f"Model B: {model_b_path if model_b_path else 'Random'}")
    print(f"Using {total_envs} parallel environments (mixed configuration)")
    print("="*40)
    
    # --- Create mixed environments ---
    env_fns = []
    
    # Config 1: A as P0 vs 2xB
    opp_type_1 = "model" if model_b_path else "random"
    for i in range(n_envs_config1):
        env_fns.append(make_eval_env(i, opp_type_1, model_b_path))
    
    # Config 2: B as P0 vs 2xA  
    opp_type_2 = "model" if model_a_path else "random"
    for i in range(n_envs_config1, total_envs):
        env_fns.append(make_eval_env(i, opp_type_2, model_a_path))
    
    vec_env = SubprocVecEnv(env_fns)
    
    # Track stats for each configuration separately
    config1_wins = 0  # A as P0
    config1_games = 0
    config2_wins = 0  # B as P0
    config2_games = 0
    
    try:
        # Load models for both configurations
        if model_a_path:
            model_a = MaskablePPO.load(model_a_path, env=vec_env)
            # Clear rollout buffer to save memory
            if hasattr(model_a, "rollout_buffer"):
                model_a.rollout_buffer = None
        else:
            model_a = None
            
        if model_b_path:
            model_b = MaskablePPO.load(model_b_path, env=vec_env)
            # Clear rollout buffer to save memory
            if hasattr(model_b, "rollout_buffer"):
                model_b.rollout_buffer = None
        else:
            model_b = None
        
        obs = vec_env.reset()
        
        # Precompute environment indices for each configuration
        config1_indices = list(range(n_envs_config1))
        config2_indices = list(range(n_envs_config1, total_envs))
        
        # Run until we have enough games for each configuration
        while config1_games < n_half or config2_games < n_half:
            action_masks = [vec_env.env_method("get_action_mask", indices=[i])[0] for i in range(total_envs)]
            action_masks = np.stack(action_masks)
            
            # Initialize actions array
            actions = np.zeros(total_envs, dtype=int)
            
            # Batch predict for Config1 (Model A)
            if model_a and config1_indices:
                config1_obs = {key: obs[key][config1_indices] for key in obs.keys()}
                config1_masks = action_masks[config1_indices]
                config1_actions, _ = model_a.predict(config1_obs, action_masks=config1_masks, deterministic=True)
                actions[config1_indices] = config1_actions
            elif not model_a and config1_indices:
                # Random actions for Config1
                for i in config1_indices:
                    valid_actions = np.flatnonzero(action_masks[i])
                    actions[i] = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            
            # Batch predict for Config2 (Model B)
            if model_b and config2_indices:
                config2_obs = {key: obs[key][config2_indices] for key in obs.keys()}
                config2_masks = action_masks[config2_indices]
                config2_actions, _ = model_b.predict(config2_obs, action_masks=config2_masks, deterministic=True)
                actions[config2_indices] = config2_actions
            elif not model_b and config2_indices:
                # Random actions for Config2
                for i in config2_indices:
                    valid_actions = np.flatnonzero(action_masks[i])
                    actions[i] = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0
            
            obs, rewards, dones, infos = vec_env.step(actions)
            
            for i, done in enumerate(dones):
                if done:
                    # Determine which config this env belongs to
                    is_config1 = i < n_envs_config1
                    
                    if is_config1 and config1_games < n_half:
                        config1_games += 1
                        if "winners" in infos[i] and 0 in infos[i]["winners"]:
                            config1_wins += 1
                    elif not is_config1 and config2_games < n_half:
                        config2_games += 1
                        if "winners" in infos[i] and 0 in infos[i]["winners"]:
                            config2_wins += 1
                    
                    total_games = config1_games + config2_games
                    if total_games % 10 == 0 or (config1_games == n_half and config2_games == n_half):
                        print(f"Progress: {total_games}/{n_total_games} | Config1 (A as P0): {config1_games}/{n_half} | Config2 (B as P0): {config2_games}/{n_half}", end="\r")
        
        print()  # New line after progress
        
    finally:
        vec_env.close()
    
    # --- Aggregation ---
    wr_a_solo = config1_wins / config1_games if config1_games > 0 else 0
    wr_b_solo = config2_wins / config2_games if config2_games > 0 else 0
    
    print("\n" + "="*40)
    print(f"FINAL RESULTS")
    print("="*40)
    print(f"1. When A is Solo (vs 2xB): A wins {wr_a_solo:.1%}")
    print(f"2. When B is Solo (vs 2xA): A wins {1-wr_b_solo:.1%}")
    print("-" * 40)
    
    combined_score = (wr_a_solo + (1.0 - wr_b_solo)) / 2
    print(f"Combined Score (A Performance): {combined_score:.1%}")
    print("="*40)


if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows support
    gym.logger.min_level = gym.logger.ERROR
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to main model zip")
    # Opponent defaults to random if not specified
    parser.add_argument("--opp", type=str, default="random", help="'random' or path to opponent model zip")
    parser.add_argument("--n", type=int, default=100, help="Total number of games")
    
    args = parser.parse_args()
    
    path_a = args.main 
    path_b = args.opp if args.opp != "random" else None
    
    evaluate_fair(path_a, path_b, args.n)