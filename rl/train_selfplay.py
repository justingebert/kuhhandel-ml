import os
import glob
import random
import numpy as np
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from rl.env import KuhhandelEnv
from rl.model_agent import ModelAgent
from tests.demo_game import RandomAgent


def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

# --- Configuration ---
LOG_DIR = "logs"
MODELS_DIR = "models"
SELFPLAY_DIR = "models/selfplay_pool_v2"
LATEST_MODEL_PATH = f"{MODELS_DIR}/kuhhandel_ppo_v2_latest"
FINAL_MODEL_PATH = f"{MODELS_DIR}/kuhhandel_ppo_v2_final"

N_GENERATIONS = 3
STEPS_PER_GEN = 10000  # Increased steps because we are faster now!
N_ENVS = 16  # Number of parallel environments (target 8-16 depending on CPU)

# Opponent Distribution
PROB_RANDOM = 0

# --- GLOBAL MODEL CACHE ---
# Cache format: { "model_name": ModelInstance }
# This is process-local. In SubprocVecEnv, each process has its own cache.
MODEL_CACHE = {}

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SELFPLAY_DIR, exist_ok=True)

def get_opponent_pool_files():
    return glob.glob(f"{SELFPLAY_DIR}/*.zip")

def get_cached_model(model_path, env_ref):
    """
    Get model from cache or load it.
    Crucial for performance: prevents disk I/O on every reset.
    """
    name = os.path.basename(model_path)
    if name not in MODEL_CACHE:
        # print(f"DEBUG: Loading model {name} from disk...")
        try:
            # We strictly map based on path, but we need env for validation?
            # MaskablePPO.load usually doesn't need env unless continuing training.
            # But for prediction it might be safer to pass it if spaces differ?
            # Actually, we just need it for prediction.
            model = MaskablePPO.load(model_path, env=None, device="auto")
            MODEL_CACHE[name] = model
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            return None
    return MODEL_CACHE[name]


def create_opponents(n_opponents: int, env_ref: KuhhandelEnv) -> list:
    """
    Generator function. Now accepts env_ref (the local KuhhandelEnv instance).
    """
    opponents = []
    pool_files = get_opponent_pool_files()
    
    for i in range(n_opponents):
        if not pool_files or random.random() < PROB_RANDOM:
            opponents.append(RandomAgent(f"Random_{i}"))
        else:
            model_file = random.choice(pool_files)
            name = os.path.basename(model_file).replace(".zip", "")
            
            # Use Cache!
            model_instance = get_cached_model(model_file, env_ref)
            
            if model_instance:
                # Pass both model_instance AND env_ref (for get_observation)
                opponents.append(ModelAgent(f"Opp_{name}_{i}", model_file, env=env_ref, model_instance=model_instance))
            else:
                opponents.append(RandomAgent(f"Fallback_{i}"))
    
    return opponents

def make_env(rank: int, geneartor_func):
    """
    Utility function for multiprocess env.
    """
    def _init():
        env = KuhhandelEnv(num_players=3)
        
        # Attach generator
        env.opponent_generator = geneartor_func
        
        # Wrap
        env = Monitor(env) # Monitor log might be messy with Subproc? Usually ok if filenames differ?
        # SB3 Monitor handles filename logging if filename provided.
        # But we just use default, which logs to `monitor.csv`? 
        # Actually SubprocVecEnv doesn't automatically merge logs.
        # We should use a unique log file per env if we want detailed logs.
        # For performance, we can skip Monitor or log to specific file.
        # Let's simple Monitor(env) -> it might just log to nothing if no file specified?
        # Re-adding explicit log dir with rank:
        log_file = os.path.join(LOG_DIR, str(rank))
        env = Monitor(env, log_file)
        
        env = ActionMasker(env, mask_valid_action)
        return env
    return _init


def main():
    print(f"Starting Self-Play Training with {N_ENVS} parallel environments...")
    
    # 1. Create Vectorized Env
    # opponent_generator is just a function reference. 
    # The 'env' arg is passed inside KuhhandelEnv.reset(), so we don't need closure here!
    # We just pass the function `create_opponents`.
    
    env_fns = [make_env(i, create_opponents) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns) 
    # Note: ActionMasker must be applied inside make_env. 
    # But MaskablePPO expects the VecEnv to have the wrapper? 
    # Actually MaskablePPO handles VecEnvs where the internal envs are masked. 
    # BUT, we might need a VecEnv wrapper if we want to expose masks cleanly?
    # NO, sb3_contrib.common.maskable.utils.get_action_masks handles standard SB3 VecEnvs 
    # by calling `get_action_mask` on each env. 
    # SubprocVecEnv uses `env_method`.
    
    # 2. Initialize Model
    if os.path.exists(LATEST_MODEL_PATH + ".zip"):
        print(f"Loading existing model from {LATEST_MODEL_PATH}")
        # Need to attach the new vec_env
        model = MaskablePPO.load(LATEST_MODEL_PATH, env=vec_env, device="auto")
    else:
        print("Creating new model")
        model = MaskablePPO("MultiInputPolicy", vec_env, verbose=1, device="auto")
    
    # 3. Training Loop
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n--- Generation {gen} ---")
        
        # Learn
        # STEPS_PER_GEN is total steps. With N_ENVS, each env does STEPS / N.
        model.learn(total_timesteps=STEPS_PER_GEN, reset_num_timesteps=False)
        
        # Save
        pool_path = f"{SELFPLAY_DIR}/gen_{gen}"
        model.save(pool_path)
        print(f"Saved generation {gen} to {pool_path}")
        
        model.save(LATEST_MODEL_PATH)
        
    print("Training loop finished.")
    model.save(FINAL_MODEL_PATH)
    vec_env.close()

if __name__ == "__main__":
    # Fix for Windows multiprocessing
    gym.logger.min_level = gym.logger.ERROR
    main()
