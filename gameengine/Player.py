from typing import List, Dict
from collections import Counter

from .Animal import AnimalCard, AnimalType
from .Money import MoneyCard, calculate_total_value


class Player:
    """Represents a player in the game."""

    def __init__(self, player_id: int, name: str = None):
        self.player_id = player_id
        self.name = name or f"Player {player_id}"
        self.money: List[MoneyCard] = []
        self.animals: List[AnimalCard] = []

    def add_money(self, cards: List[MoneyCard]):
        """Add money cards to player's hand."""
        self.money.extend(cards)

    def remove_money(self, cards: List[MoneyCard]):
        """Remove specific money cards from player's hand."""
        for card in cards:
            if card in self.money:
                self.money.remove(card)

    def add_animal(self, card: AnimalCard):
        """Add an animal card to player's collection."""
        self.animals.append(card)

    def remove_animals(self, animal_type: AnimalType, count: int = 1) -> List[AnimalCard]:
        """Remove and return a specific number of animals of a type."""
        removed = []
        for _ in range(count):
            for card in self.animals:
                if card.animal_type == animal_type:
                    self.animals.remove(card)
                    removed.append(card)
                    break
        return removed

    def get_total_money(self) -> int:
        """Get total money value."""
        return calculate_total_value(self.money)

    def get_animal_counts(self) -> Dict[AnimalType, int]:
        """Get count of each animal type owned."""
        counts = Counter(card.animal_type for card in self.animals)
        return dict(counts)

    def has_animal_type(self, animal_type: AnimalType) -> bool:
        """Check if player has at least one of the given animal type."""
        return any(card.animal_type == animal_type for card in self.animals)

    def get_animal_count(self, animal_type: AnimalType) -> int:
        """Get count of a specific animal type."""
        return sum(1 for card in self.animals if card.animal_type == animal_type)

    def has_complete_set(self, animal_type: AnimalType) -> bool:
        """Check if player has all 4 cards of an animal type."""
        return self.get_animal_count(animal_type) == 4

    def calculate_score(self) -> int:
        """Calculate final score according to game rules.

        Score = (sum of complete set values) × (number of complete sets)

        Only complete sets of 4 cards count towards scoring.
        Each animal type has a value representing all 4 cards together.

        Example: Horse(4), Pig(4), Cow(4) = (1000+650+800) × 3 = 7350 points
        """
        animal_counts = self.get_animal_counts()
        if not animal_counts:
            return 0

        # Only count complete sets (4 cards)
        complete_sets = {animal_type: count
                        for animal_type, count in animal_counts.items()
                        if count == 4}

        if not complete_sets:
            return 0

        # Sum of animal values for complete sets only
        total_value = sum(
            animal_type.get_value()
            for animal_type in complete_sets.keys()
        )

        # Number of complete sets
        num_complete_sets = len(complete_sets)

        return total_value * num_complete_sets

    def __repr__(self) -> str:
        return f"Player({self.name}, Money: {self.get_total_money()}, Animals: {len(self.animals)})"

