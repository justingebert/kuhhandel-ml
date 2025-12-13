from typing import Optional, List

from gameengine import Game
from gameengine.actions import GameAction
from gameengine.agent import Agent
from rl.env import KuhhandelEnv


class RLAgent(Agent):
    """Agent wrapper for training - converts int actions to GameActions.
    """

    def __init__(self, name: str, env: KuhhandelEnv):
        super().__init__(name)
        self.env = env
        self.last_action_int: Optional[int] = None

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        """Convert stored integer action to GameAction based on current game phase."""
        action_idx = self.last_action_int
        return self.env.decode_action(action_idx, game)
