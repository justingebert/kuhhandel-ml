from enum import Enum
from typing import List


class AnimalType(Enum):
    """Enum for different animal types in the game."""
    CAT = ("Cat", 10, 25, 50, 90)
    CHICKEN = ("Chicken", 10, 20, 40, 70)
    COW = ("Cow", 100, 200, 300, 400)
    DOG = ("Dog", 40, 80, 120, 160)
    DONKEY = ("Donkey", 50, 100, 150, 200)
    GOAT = ("Goat", 60, 120, 180, 240)
    GOOSE = ("Goose", 20, 40, 80, 140)
    HORSE = ("Horse", 200, 300, 400, 500)
    LAMB = ("Lamb", 30, 60, 90, 120)
    PIG = ("Pig", 130, 260, 390, 650)

    def __init__(self, display_name: str, value_1: int, value_2: int, value_3: int, value_4: int):
        self.display_name = display_name
        self.values = [value_1, value_2, value_3, value_4]

    def get_value(self, count: int) -> int:
        """Get the total value for a certain count of this animal (1-4)."""
        if count < 1 or count > 4:
            return 0
        return self.values[count - 1]

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

