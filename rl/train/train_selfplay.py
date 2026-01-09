import os
import glob
import random
import numpy as np
import gymnasium as gym
from pathlib import Path
import multiprocessing
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable import distributions as maskable_dist
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import wandb
from wandb.integration.sb3 import WandbCallback

from rl.env import KuhhandelEnv
from rl.rewardconfigs.reward_configs import RewardMinimalAggressiveConfig, RewardConfig, WinOnlyConfig

from rl.agents.model_agent import ModelAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.rdm_schwaben_agent import RandomSchwabenAgent

# ==========================================
# CONFIGURATION
# ==========================================

# ITP Server Configuration (Uncomment to use)
# N_GENERATIONS = 150
# STEPS_PER_GEN = 60000
# MAX_ENVS = 32
# PROB_RANDOM = 0.05
# REWARD_CONFIG_CLASS = "Default" # Options: "Default", "RewardMinimalAggressiveConfig", "WinOnlyConfig"
# POOL_SAVE_MODULO = 30 # Save to folders gen_{gen % 30}
# CREATE_PROGRESS_FILES = True

# Local / Standard Configuration
N_GENERATIONS = 5
STEPS_PER_GEN = 1000
MAX_ENVS = 16
PROB_RANDOM = 0.1
REWARD_CONFIG_CLASS = "RewardMinimalAggressiveConfig" # Options: "Default", "RewardMinimalAggressiveConfig", "WinOnlyConfig"
POOL_SAVE_MODULO = 0 # 0 means save to unique folder for every gen
CREATE_PROGRESS_FILES = False

# Common Config
PROB_SCHWABE = 0.8

DEVICE = "cpu"
# !WICHTIG!!!!!
RUN_NAME = "None" # Set this to a string to name the run, or leave None for auto-generated name

# ==========================================
# END CONFIGURATION
# ==========================================


# Fix for Simplex error due to floating point precision issues
def robust_apply_masking(self, masks: torch.Tensor):
    # Access logits directly
    if hasattr(self, 'logits'):
        logits = self.logits
    elif hasattr(self, 'probs'):
         logits = torch.log(self.probs + 1e-8)
    else:
        # Fallback 
        logits = torch.tensor([])

    HUGE_NEG = -1e8
    if masks is not None and logits.numel() > 0:
        # Handle numpy masks
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks).to(logits.device)
        
        # Apply masking
        masked_logits = torch.where(masks.bool(), logits, torch.tensor(HUGE_NEG).to(logits.device))
        
        # Re-initialize the Categorical distribution with validate_args=False
        # This bypasses the strict Simplex check that fails on 1e-7 errors
        torch.distributions.Categorical.__init__(self, logits=masked_logits, validate_args=False)
    
    return self

maskable_dist.MaskableCategorical.apply_masking = robust_apply_masking


def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

# Dictionary to cache loaded models
MODEL_CACHE = {}

def get_cached_model(model_path):
    """
    Get model from cache or load it.
    """
    name = os.path.basename(model_path)
    
    if name not in MODEL_CACHE:
        try:
            model = MaskablePPO.load(model_path, env=None, device=DEVICE)
            
            if hasattr(model, "rollout_buffer"): #Buffer safes past moves required for training not for simulation
                model.rollout_buffer = None
                
            MODEL_CACHE[name] = model
        except Exception as e:
            print(f"Error loading model {name}: {e}")
            return None
        
    return MODEL_CACHE[name]


def create_opponents(env_ref: KuhhandelEnv, n_opponents: int) -> list:
    opponents = []
    
    # We re-calculate SELFPLAY_DIR here to be safe across processes
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir / "models"
    selfplay_dir = model_dir / "selfplay_pool"
    
    pool_files = glob.glob(f"{selfplay_dir}/*.zip")
    
    for i in range(n_opponents):
        r = random.random()
        if not pool_files or r < PROB_RANDOM: 
            if r < PROB_SCHWABE:
                opponents.append(RandomSchwabenAgent(f"Random_{i}"))
            else:
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

def make_env(rank: int, reward_config):
    def _init():
        torch.set_num_threads(1) #damit die subprocesse nicht um kerne streiten
        
        env = KuhhandelEnv(num_players=3, reward_config=reward_config)
        
        env.opponent_generator = create_opponents
        
        env = ActionMasker(env, mask_valid_action)
        return env
    
    return _init


def main():
    print(f"Starting Self-Play Training with {MAX_ENVS} parallel environments...")
    
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir / "models"
    log_dir = script_dir / "logs" # Not strictly used but kept
    tb_log_dir = script_dir / "tensorboard_logs"
    selfplay_dir = model_dir / "selfplay_pool"
    
    latest_model_path = str(model_dir / "kuhhandel_ppo_latest")
    final_model_path = str(model_dir / "kuhhandel_ppo_final")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(selfplay_dir, exist_ok=True)
    
    # Resolve Reward Config
    if REWARD_CONFIG_CLASS == "RewardMinimalAggressiveConfig":
        reward_config = RewardMinimalAggressiveConfig()
    elif REWARD_CONFIG_CLASS == "WinOnlyConfig":
        reward_config = WinOnlyConfig()
    else:
        reward_config = RewardConfig()

    print(f"Configuration:")
    print(f"  Gens: {N_GENERATIONS}, Steps: {STEPS_PER_GEN}, Envs: {MAX_ENVS}")
    print(f"  Random Prob: {PROB_RANDOM}, Reward: {REWARD_CONFIG_CLASS}")

    # Create Envs
    n_envs = min(multiprocessing.cpu_count(), MAX_ENVS)
    
    env_fns = [make_env(i, reward_config) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns) 

    policy_kwargs = dict(net_arch=[256, 256])
    
    # load existing or create new one
    if os.path.exists(latest_model_path + ".zip"):
        print(f"Loading existing model from {latest_model_path}")
        model = MaskablePPO.load(latest_model_path, env=vec_env, device=DEVICE, tensorboard_log=str(tb_log_dir))
    else:
        print("Creating new model")
        model = MaskablePPO(
            "MultiInputPolicy", 
            vec_env,
            verbose=1, 
            device=DEVICE,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tb_log_dir)
        )

    # Initialize W&B
    run = wandb.init(
        project="kuhhandel",
        name=RUN_NAME,
        config={
            "n_generations": N_GENERATIONS,
            "steps_per_gen": STEPS_PER_GEN,
            "max_envs": MAX_ENVS,
            "prob_random": PROB_RANDOM,
            "reward_config": REWARD_CONFIG_CLASS,
            "prob_schwabe": PROB_SCHWABE,
            "device": DEVICE
        },
        sync_tensorboard=True,
    )
    
    start_time = time.time()
    
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n--- Generation {gen} ---")
        
        model.learn(
            total_timesteps=STEPS_PER_GEN, 
            reset_num_timesteps=False,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"{selfplay_dir}/wandb_models",
                verbose=2,
            )
        )
        
        # Save to pool
        if POOL_SAVE_MODULO > 0:
            pool_path = f"{selfplay_dir}/gen_{gen % POOL_SAVE_MODULO}"
        else:
            pool_path = f"{selfplay_dir}/gen_{gen}"
            
        model.save(pool_path)
        print(f"Saved generation {gen} to {pool_path}")
        
        model.save(latest_model_path)
        
        # Progress indication files (ITP specific)
        if CREATE_PROGRESS_FILES:
            txt_file = script_dir / f"gen{gen}.txt"
            with open(txt_file, "w") as f:
                f.write(f"Generation {gen} completed at {time.ctime()}")
            
            if gen > 1:
                prev_txt = script_dir / f"gen{gen-1}.txt"
                if os.path.exists(prev_txt):
                    os.remove(prev_txt)

        # Print interim timing
        elapsed_so_far = time.time() - start_time
        print(f"Time elapsed: {elapsed_so_far/60:.2f} mins")
        
    total_time = time.time() - start_time
    print(f"Training loop finished in {total_time/60:.2f} minutes. Model saved to {final_model_path}")
    run.finish()
    vec_env.close()

if __name__ == "__main__":
    gym.logger.min_level = gym.logger.ERROR
    multiprocessing.freeze_support() # Windows support
    main()