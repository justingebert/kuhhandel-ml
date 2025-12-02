import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete


# Discrete action space (button presses)
env = gym.make("CartPole-v1")
env.action_space = Dict({
                "Gegner": Discrete(3),
                "Tier": Discrete(10),
                "Gebot": MultiBinary(33), # (7 + 4) * 3,
                "Bieten": MultiBinary(33),
                "Passen": Discrete(3)
        })

env.observation_space = Dict({
        "Game_phase": Discrete(100),  # Number of round, gamestate wird die Entscheidung nie beeinflussen, da durch maskieren des action space irrelevant
        "Player_turn": Discrete(3),
        "Money_own": MultiBinary(33),
        "Money_opp_0": Discrete(33),
        "Money_opp_1": Discrete(33),
        "Animals_0": MultiDiscrete(10* [4]),
        "Animals_1": MultiDiscrete(10* [4]),
        "Animals_2": MultiDiscrete(10* [4]),
        "Current_auction_animal": Discrete(10),
        "Current_auction_highest_bid": Discrete(int(940*3/10)),
        "Current_auction_highest_bidder": Discrete(3),
})

env.action_space = Dict({
    "move": Discrete(2),  # Left or right
    "jump": Discrete(2),  # Jump or not                 # probability=np.array([0.1,0.9], dtype=np.float64)
})

print(env.action_space.sample(mask={"move": np.array([0,1], dtype=np.int8), "jump": np.array([0,1], dtype=np.int8)}))  # Sample action based on given probabilities
print(env.action_space.sample(probability={"move": np.array([0.5,0.5], dtype=np.float64), "jump": np.array([0.5,0.5], dtype=np.float64)}))


    
# print(f"Action space: {env.action_space}")  # Discrete(2) - left or right
# print(f"Sample action: {env.action_space.sample()}")  # 0 or 1

# Box observation space (continuous values)
#print(f"Observation space: {env.observation_space}")  # Box with 4 values
# Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
#print(f"Sample observation: {env.observation_space.sample()}") 