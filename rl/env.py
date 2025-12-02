from typing import Optional, List
import gymnasium as gym
from gymnasium import spaces

from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete

from gameengine.actions import GameAction
from gameengine.agent import Agent
from gameengine.controller import GameController
from gameengine.game import Game
from rl.rl_agent import RLAgent
from tests.demo_game import RandomAgent

class KuhhandelEnv(gym.Env):
    
    def __init__(self, num_players: int = 3):
        super().__init__()

        self.num_players = num_players

        self.action_space = Dict(  
        {
        "Handeln/Versteigern": Discrete(2),
        "Handeln": Dict({
                "Gegner": Discrete(3),
                "Tier": Discrete(10),
                "Gebot": Discrete(33)}), # (7 + 4) * 3
        
        "Versteigern": Dict({
                "Bieten/Passen/Vorkaufen": Discrete(3),
                "Bieten": Discrete(33),})
        })





        self.observation_space = None #obs vector
        
        self.game: Optional[Game] = None
        self.controller: Optional[GameController] = None
        self.agents: List[Agent] = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.game = Game(num_players=self.num_players, seed=seed).setup()

        self.agents = []
        # RL agent at position 0
        self.agents.append(RLAgent("RL_Agent", self))
        # Random agents for opponents
        for i in range(1, self.num_players):
            self.agents.append(RandomAgent(f"Random_{i}"))
            
        self.controller = GameController(self.game, self.agents)


    def _decode_action(self, action_int: int, game: Game) -> GameAction:
        """Decode an integer action to a GameAction."""
        pass




