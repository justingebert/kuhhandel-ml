import numpy as np
from sb3_contrib import MaskablePPO
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker

from rl.env import KuhhandelEnv


def mask_valid_action(env: gym.Env) -> np.ndarray:
    """
    Return a mask of valid actions for the current game state.
    1 = action is valid, 0 = action is invalid.
    """
    return env.unwrapped.get_action_mask()


env = KuhhandelEnv()
env = ActionMasker(env, mask_valid_action)

model = MaskablePPO("MultiInputPolicy", env)
model.learn(total_timesteps=1000)