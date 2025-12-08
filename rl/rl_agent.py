from typing import Optional, List

import gymnasium

from gameengine import Game
from gameengine.actions import GameAction, ActionType
from gameengine.agent import Agent
from rl.env import KuhhandelEnv


class RLAgent(Agent):
    """Agent wrapper for RL model - converts int actions to GameActions."""

    def __init__(self, name: str, env: gymnasium.Env):
        super().__init__(name)
        self.env = env
        self.last_action_int: Optional[int] = None

    def get_action(self, game: Game, valid_actions: List[ActionType]) -> GameAction:
        """This will be called by the controller - we return the decoded action."""
        return self.env._decode_action(self.last_action_int, game)
