from abc import ABC, abstractmethod
from typing import Optional, List

from gameengine import Game
from gameengine.actions import GameAction, ActionType


class Agent(ABC):
    """Abstract base class for a game agent."""

    def __init__(self, name: str):
        self.name = name
        self.player_id: Optional[int] = None

    def set_player_id(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def get_action(self, game: Game, valid_actions: List[ActionType]) -> GameAction:
        """
        Decide on an action to take.
        Returns a typed action dictionary.
        """
        pass

    def notify(self, event: str, data: dict):
        """Receive notification about game events (optional)."""
        pass
