import random
from typing import List

from gameengine import Game
from gameengine.actions import GameAction
from gameengine.agent import Agent


class RandomAgent(Agent):
    """A simple agent that makes random valid moves."""

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        """Select a random valid action."""
        return random.choice(valid_actions)
