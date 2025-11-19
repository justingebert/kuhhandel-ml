from enum import Enum
from typing import List


class AnimalType(Enum):
    """Enum for different animal types in the game.

    The points value represents the total points for having all 4 cards of that animal type.
    Only complete sets of 4 cards score points.
    """
    CHICKEN = ("Chicken", 10)
    GOOSE = ("Goose", 40)
    CAT = ("Cat", 90)
    DOG = ("Dog", 160)
    SHEEP = ("Sheep", 250)
    GOAT = ("Goat", 350)
    DONKEY = ("Donkey", 500)
    PIG = ("Pig", 650)
    COW = ("Cow", 800)
    HORSE = ("Horse", 1000)

    def __init__(self, display_name: str, points: int):
        self.display_name = display_name
        self.points = points

    def get_value(self) -> int:
        """Get the total points for this animal type."""
        return self.points

    @staticmethod
    def get_all_types() -> List['AnimalType']:
        """Return all animal types."""
        return list(AnimalType)


class AnimalCard:
    """Represents a single animal card."""

    def __init__(self, animal_type: AnimalType, card_id: int):
        self.animal_type = animal_type
        self.card_id = card_id

    def __repr__(self) -> str:
        return f"AnimalCard({self.animal_type.display_name})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnimalCard):
            return False
        return self.card_id == other.card_id

    def __hash__(self) -> int:
        return hash(self.card_id)

