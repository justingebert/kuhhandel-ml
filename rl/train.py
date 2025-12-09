import os
import numpy as np
from sb3_contrib import MaskablePPO
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from rl.env import KuhhandelEnv


def mask_valid_action(env: gym.Env) -> np.ndarray:
    """
    Return a mask of valid actions for the current game state.
    1 = action is valid, 0 = action is invalid.
    """
    return env.unwrapped.get_action_mask()


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

env = KuhhandelEnv()
env = Monitor(env, log_dir)
env = ActionMasker(env, mask_valid_action)

# Save a checkpoint every 10000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=models_dir,
    name_prefix="kuhhandel_ppo",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

model = MaskablePPO("MultiInputPolicy", env, verbose=1)

print("Starting training...")
model.learn(total_timesteps=50000, callback=checkpoint_callback)

print("Saving final model...")
model.save(f"{models_dir}/kuhhandel_ppo_final")
print("Done!")