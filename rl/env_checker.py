from rl.env import KuhhandelEnv
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

env = KuhhandelEnv(num_players=3)
# It will check your custom environment and output additional warnings if needed
check_env(env)