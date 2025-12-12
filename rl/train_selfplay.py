import os
import glob
import random
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from rl.env import KuhhandelEnv
from rl.model_agent import ModelAgent
from tests.demo_game import RandomAgent


def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

# --- Configuration ---
LOG_DIR = "logs"
MODELS_DIR = "models"
SELFPLAY_DIR = "models/selfplay_pool"
LATEST_MODEL_PATH = f"{MODELS_DIR}/kuhhandel_ppo_latest"
FINAL_MODEL_PATH = f"{MODELS_DIR}/kuhhandel_ppo_final"

N_GENERATIONS = 10
STEPS_PER_GEN = 20000  # steps per generation

# Opponent Distribution
PROB_RANDOM = 0.2  # 20% chance for pure random opponent
# Remaining 80% picked from pool of past models

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SELFPLAY_DIR, exist_ok=True)

def get_opponent_pool_files():
    """Return list of valid model files from the pool."""
    return glob.glob(f"{SELFPLAY_DIR}/*.zip")

def create_opponents(n_opponents: int) -> list:
    """
    Generator function to create opponents.
    Picks randomly from:
    1. RandomAgent (exploration)
    2. Available past models (exploitation/robustness)
    """
    opponents = []
    pool_files = get_opponent_pool_files()
    
    for i in range(n_opponents):
        # Decide type
        if not pool_files or random.random() < PROB_RANDOM:
            opponents.append(RandomAgent(f"Random_{i}"))
        else:
            # Pick a random past model
            model_file = random.choice(pool_files)
            # Use basename for name
            name = os.path.basename(model_file).replace(".zip", "")
            try:
                opponents.append(ModelAgent(f"Opponent_{name}_{i}", model_file, env_ref=None)) 
                # Note: ModelAgent needs env ref. 
                # But inside 'reset', the env exists. 
                # Wait, ModelAgent needs an env instance to call get_observation.
                # The env calls this generator. 
                # WE HAVE A CIRCULAR DEPENDENCY if we pass the env to the generator used by the env.
                # Solution: The generator should just return agents. The agents need the env attached later?
                # Or we pass 'env' to this function.
                # Env.reset calls self.opponent_generator(n).
                # We can use a partial or lambda in main to bind the env?
                # No, env isn't fully initialized inside __init__ when we bind it?
                # Actually, inside reset(), 'self' is the env.
                # We can make opponent_generator accept (n_opponents, env_instance).
            except Exception as e:
                print(f"Failed to load opponent {model_file}: {e}")
                opponents.append(RandomAgent(f"RandomFallback_{i}"))
    
    return opponents

# We need a wrapper for the generator to handle the env injection
# Updated Env signature in my mind: opponent_generator(num, env) ?
# I didn't update Env to pass 'self' to the generator. I just called `self.opponent_generator(self.num_players - 1)`.
# I should have checked that!
# Let's check rl/env.py content I just wrote.
# Line 78: others = self.opponent_generator(self.num_players - 1)
# It does NOT pass self.
# 
# FIX: I can attach the env to the agents AFTER they are returned?
# Env code:
#   others = ...
#   self.agents.extend(others)
# 
# The agents are just objects.
# ModelAgent needs 'env' to run `get_action`.
# `get_action` is called during `step`.
# So I can attach `env` to the agents in `reset`!
# But Env doesn't know they are ModelAgents potentially needing env.
# 
# Workaround: Pass env via a global or closure?
# Better Workaround: The generator is a bound method of a class that holds the env? No.
# Best: Modify Env to pass 'self' to generator.
# But I just wrote Env.
# 
# Alternative: 
# `create_opponents` returns a list of Agents. 
# ModelAgent(..., env=None).
# Then in the loop, we set agent.env = current_env?
# But Env logic doesn't do that.
# 
# I will use a simple hack: 
# The `ModelAgent` will accept `env=None` initially.
# But `get_action` fails if `env` is None.
# 
# I can define the opponent generator as a closure that captures the env?
# `env = KuhhandelEnv(...)`
# `env.opponent_generator = lambda n: create_opponents(n, env)`
# But Env init takes generator.
# 
# `env = KuhhandelEnv(opponent_generator=None)`
# `env.opponent_generator = lambda n: create_opponents(n, env)`
# This works! I can set it after init.
#
# So in this script:
# 1. Init Env with None.
# 2. Define generator that uses 'env' variable.
# 3. Set env.opponent_generator = generator.
# 
# wait, create_opponents needs to instantiate ModelAgent(..., env=env).
# Yes.
    
    return opponents

def main():
    # 1. Create Env
    # We delay setting generator until we have the env variable
    env = KuhhandelEnv(num_players=3)
    
    # 2. Define Generator with captured env
    def generator(n):
        return create_opponents_with_env(n, env)
        
    env.opponent_generator = generator
    
    # 3. Wrap Env
    env = Monitor(env, LOG_DIR)
    env = ActionMasker(env, mask_valid_action)
    
    # 4. Initialize Model
    # Try to load latest if exists, else new
    if os.path.exists(LATEST_MODEL_PATH + ".zip"):
        print(f"Loading existing model from {LATEST_MODEL_PATH}")
        model = MaskablePPO.load(LATEST_MODEL_PATH, env=env)
    else:
        print("Creating new model")
        model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    
    # 5. Training Loop
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n--- Generation {gen} ---")
        
        # Learn
        model.learn(total_timesteps=STEPS_PER_GEN, reset_num_timesteps=False)
        
        # Save current version to pool
        pool_path = f"{SELFPLAY_DIR}/gen_{gen}"
        model.save(pool_path)
        print(f"Saved generation {gen} to {pool_path}")
        
        # Update "Latest" pointer
        model.save(LATEST_MODEL_PATH)
        
        # Note: The Env's reset() is called inside learn().
        # automatically, sb3 calls reset().
        # reset() calls opponent_generator.
        # opponent_generator uses get_opponent_pool_files().
        # create_opponents_with_env will pick up the NEW file we just saved in the NEXT episode/reset.
        # So the opponents evolve automatically as the pool grows.
        
    print("Training loop finished.")
    model.save(FINAL_MODEL_PATH)

def create_opponents_with_env(n_opponents, env_ref):
    opponents = []
    pool_files = get_opponent_pool_files()
    
    for i in range(n_opponents):
        if not pool_files or random.random() < PROB_RANDOM:
            opponents.append(RandomAgent(f"Random_{i}"))
        else:
            model_file = random.choice(pool_files)
            name = os.path.basename(model_file).replace(".zip", "")
            try:
                # We pass the env_ref here!
                opponents.append(ModelAgent(f"Opp_{name}_{i}", model_file, env=env_ref))
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
                opponents.append(RandomAgent(f"RandomFallback_{i}"))
    return opponents

if __name__ == "__main__":
    main()
