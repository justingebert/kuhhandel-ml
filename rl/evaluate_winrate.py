import os
import argparse
import multiprocessing
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.env import KuhhandelEnv
from rl.model_agent import ModelAgent
from tests.demo_game import RandomAgent

def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

def create_specific_opponents(n_opponents, env_ref, opponent_type, opponent_model_path=None):
    """
    Generator that creates specific opponents for evaluation.
    """
    opponents = []
    for i in range(n_opponents):
        if opponent_type == "random":
            opponents.append(RandomAgent(f"Random_{i}"))
        elif opponent_type == "model":
            if not opponent_model_path:
                # Fallback to random if no path
                opponents.append(RandomAgent(f"Random_{i}"))
            else:
                try:
                    # Note: In subprocess, we load the model fresh per env mostly.
                    opponents.append(ModelAgent(f"Opponent_{i}", opponent_model_path, env=env_ref))
                except Exception as e:
                    print(f"Error loading opponent model: {e}")
                    opponents.append(RandomAgent(f"Fallback_{i}"))
    return opponents

def make_eval_env(rank, opponent_type, opponent_model_path):
    def _init():
        env = KuhhandelEnv(num_players=3)
        
        # Setup Opponent Generator
        def eval_generator(n, env_ref):
            return create_specific_opponents(n, env_ref, opponent_type, opponent_model_path)
            
        env.opponent_generator = eval_generator
        env = ActionMasker(env, mask_valid_action)
        return env
    return _init

def evaluate(main_model_path, opponent_type="random", opponent_model_path=None, n_games=100):
    # Determine parallel envs
    # Don't use more envs than games requested
    n_cpu = multiprocessing.cpu_count()
    # Leave some headroom or cap at 16? User has 16 cores mentioned.
    n_envs = min(n_games, n_cpu, 16)
    
    print(f"\nEvaluating Main Model: {main_model_path}")
    print(f"Against Opponents: {opponent_type} ({opponent_model_path if opponent_model_path else ''})")
    print(f"Total Games: {n_games} | Parallel Envs: {n_envs}")
    
    # Create Vec Env
    env_fns = [make_eval_env(i, opponent_type, opponent_model_path) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    try:
        # Load Main Model
        # attach vec_env to model (sometimes needed for normalizing, though we don't normalize here)
        main_model = MaskablePPO.load(main_model_path, env=vec_env)
        
        wins = 0
        games_completed = 0
        # We need to track scores per env? 
        # The env doesn't return score in 'info' by default yet.
        # But we can update KuhhandelEnv to return score in info?
        # OR we rely on reward=1 for win.
        # Let's rely on reward for winrate.
        
        obs = vec_env.reset()
        
        while games_completed < n_games:
            # Predict
            # sb3_contrib.common.maskable.utils.get_action_masks is handled by MaskablePPO.predict if env is attached?
            # Actually MaskablePPO.predict expects `action_masks` argument if utilizing masking explicitly
            # OR if env is wrapped with ActionMasker and passed to load?
            # The cleanest way with VecEnv is to let the model access the envs action masks.
            # But SubprocVecEnv hides them.
            # MaskablePPO handles this by checking if env has `get_action_masks` method? 
            # SubprocVecEnv does NOT have it by default.
            
            # We must manually fetch masks from all envs.
            action_masks = [vec_env.env_method("get_action_mask", indices=[i])[0] for i in range(n_envs)]
            action_masks = np.stack(action_masks)
            
            action, _ = main_model.predict(obs, action_masks=action_masks, deterministic=True)
            
            obs, rewards, dones, infos = vec_env.step(action)
            
            for i, done in enumerate(dones):
                if done:
                    games_completed += 1
                    # Check reward for win
                    if rewards[i] > 0:
                        wins += 1
                        
                    if games_completed % 10 == 0 or games_completed == n_games:
                         print(f"Progress: {games_completed}/{n_games} | Wins: {wins} | WR: {wins/games_completed:.2%}", end="\r")
                    
                    if games_completed >= n_games:
                        break
                        
    finally:
        vec_env.close()

    winrate = wins / games_completed if games_completed > 0 else 0
    
    print("\n" + "="*30)
    print(f"RESULTS ({games_completed} games)")
    print(f"Winrate: {winrate:.2%}")
    print("="*30)

if __name__ == "__main__":
    # Windows support
    multiprocessing.freeze_support()
    gym.logger.min_level = gym.logger.ERROR
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, required=True, help="Path to main model zip")
    parser.add_argument("--opp", type=str, default="random", help="'random' or path to opponent model zip")
    parser.add_argument("--n", type=int, default=100, help="Number of games")
    
    args = parser.parse_args()
    
    opp_type = "model" if args.opp != "random" else "random"
    opp_path = args.opp if args.opp != "random" else None
    
    evaluate(args.main, opp_type, opp_path, args.n)

