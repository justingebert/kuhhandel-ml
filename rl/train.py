import numpy as np
from sb3_contrib import MaskablePPO
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker

from rl.env import KuhhandelEnv

#TODO
def mask_valid_action(env: gym.Env) -> np.ndarray:
    pass

env = KuhhandelEnv()
env = ActionMasker(env, mask_valid_action)

model = MaskablePPO("MlpPolicy", env)
# model.learn()