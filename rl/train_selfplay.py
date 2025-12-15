import os
import glob
import random
import numpy as np
import gymnasium as gym
from pathlib import Path
import multiprocessing

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from rl.env import KuhhandelEnv
from rl.model_agent import ModelAgent
from tests.demo_game import RandomAgent


def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "models"

LOG_DIR = "logs"
SELFPLAY_DIR = f"{MODEL_DIR}/selfplay_pool"
LATEST_MODEL_PATH = f"{MODEL_DIR}/kuhhandel_ppo_latest"
FINAL_MODEL_PATH = f"{MODEL_DIR}/kuhhandel_ppo_final"

N_GENERATIONS = 20
STEPS_PER_GEN = 20000  
N_ENVS = min(multiprocessing.cpu_count(), 16) #use available cores up to a maximum of 16

# Opponent Distribution
PROB_RANDOM = 0

MODEL_CACHE = {}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SELFPLAY_DIR, exist_ok=True)


def get_cached_model(model_path):
    """
    Get model from cache or load it.
    """
    name = os.path.basename(model_path)
    
    if name not in MODEL_CACHE:
        try:
            model = MaskablePPO.load(model_path, env=None, device="auto")
            MODEL_CACHE[name] = model
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            return None
        
    return MODEL_CACHE[name]


def create_opponents(env_ref: KuhhandelEnv, n_opponents: int,) -> list:
    opponents = []
    pool_files = glob.glob(f"{SELFPLAY_DIR}/*.zip")
    
    for i in range(n_opponents):
        #if not models exist append random agent, or sometimes 
        if not pool_files or random.random() < PROB_RANDOM: #experiment with this
            opponents.append(RandomAgent(f"Random_{i}"))
        else:
            model_file = random.choice(pool_files)
            name = os.path.basename(model_file).replace(".zip", "")
            
            model_instance = get_cached_model(model_file)
            
            if model_instance:
                opponents.append(ModelAgent(f"Opp_{name}_{i}", model_file, env=env_ref, model_instance=model_instance))
            else:
                raise RuntimeError(f"Some wrong with loading model: {model_file}")
    
    return opponents

def make_env(rank: int, opponent_generator_func):
    def _init():
        env = KuhhandelEnv(num_players=3)
        
        env.opponent_generator = opponent_generator_func
        
        log_file = os.path.join(LOG_DIR, str(rank))
        env = Monitor(env, log_file)
        
        env = ActionMasker(env, mask_valid_action)
        return env
    
    return _init


def main():
    print(f"Starting Self-Play Training with {N_ENVS} parallel environments...")
    
    # create n envs for parallel training
    env_fns = [make_env(i, create_opponents) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns) 

    
    # load existing pr create new one
    if os.path.exists(LATEST_MODEL_PATH + ".zip"):
        model = MaskablePPO.load(LATEST_MODEL_PATH, env=vec_env, device="auto")
    else:
        model = MaskablePPO("MultiInputPolicy", vec_env, verbose=1, device="auto")
    
    
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n--- Generation {gen} ---")
        
        # STEPS_PER_GEN is total steps. With N_ENVS, each env does STEPS / N.
        model.learn(total_timesteps=STEPS_PER_GEN, reset_num_timesteps=False)
        
        pool_path = f"{SELFPLAY_DIR}/gen_{gen}"
        model.save(pool_path)
        print(f"Saved generation {gen} to {pool_path}")
        
        model.save(LATEST_MODEL_PATH)
        
    model.save(FINAL_MODEL_PATH)
    print(f"Training loop finished. Model saved to {FINAL_MODEL_PATH}")
    vec_env.close()

if __name__ == "__main__":
    gym.logger.min_level = gym.logger.ERROR
    main()
