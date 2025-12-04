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
                # decide wether to start an auction or trade with another player
                "auction_or_trade": Discrete(2),
                "auction": Dict(
                    {
                        # Bids from 0 to 2820 in card logic - 33 is the max money for 3 players, the agent can have all 33
                        "auction_bid": MultiBinary(33), # or discrete(282)
                        # Pass (Not Smash), pass as bidder
                        "auction_pass": Discrete(1),
                        "auction_buy_or_sell_as_auctioneer": Discrete(2),
                    }
                ),
                "trade": Dict(
                    {
                        "trade_enemy": Discrete(3),
                        "trade_animal": Discrete(10),
                        "trade_offer": MultiBinary(33), # or discrete(282)
                        "trade_counter": MultiBinary(33), # or discrete(282)
                    }
                ),
            }
        )



        # broken down observation space for first runs
        self.observation_space = Dict(
            {
                "animals_player_0": Discrete(40),
                "animals_player_1": Discrete(40),
                "animals_player_2": Discrete(40),
                "own_money": MultiBinary(33), # or discrete(282)
            }
        )

        self.game: Optional[Game] = None
        self.controller: Optional[GameController] = None
        self.agents: List[Agent] = []
        self.rl_agent_id = 0 # RL agent is always player 0
        
        self.episode_step = 0
        self.max_steps = 500

    

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
        
        self.episode_steps = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
        
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: TODO
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.episode_steps += 1
        
        self.agents[self.rl_agent_id].set_action(action)
    
    
    def _get_observation(self):
        """Get the current observation for the RL agent."""
        pass
    

    def _decode_action(self, action_int: int, game: Game) -> GameAction:
        """Decode an integer action to a GameAction."""
        pass
