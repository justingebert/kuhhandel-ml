from typing import Optional, List

import gymnasium

from gameengine import Game
from gameengine.actions import GameAction
from gameengine.agent import Agent
from rl.env import decode_action


class RLAgent(Agent):
    """Agent wrapper for training - converts int actions to GameActions.
    """

    def __init__(self, name: str, env: 'gymnasium.Env'):
        super().__init__(name)
        self.env = env
        self.last_action_int: Optional[int] = None

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        """Convert stored integer action to GameAction based on current game phase."""
        action_idx = self.last_action_int
        decoded_action = decode_action(action_idx, game)
        return decoded_action
