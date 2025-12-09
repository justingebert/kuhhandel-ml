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

    def get_money_histogram(self, money_values: List[int]) -> Dict[int, int]:
        """Get count of each money card value.

        Args:
            money_values: List of money values to count

        Returns:
            Dictionary mapping money value to count
        """
        counts = Counter(card.value for card in self.money)
        return {value: counts.get(value, 0) for value in money_values}

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

    def select_payment_cards(self, amount: int) -> List[MoneyCard]:
        """Select money cards to pay a specific amount (or more if exact not possible).

        Uses dynamic programming to find the optimal combination of cards.
        Prefers exact payment, falls back to minimum overpayment if necessary.

        Args:
            amount: The amount to pay

        Returns:
            List of money cards to use for payment

        Raises:
            ValueError: If payment is not possible with available cards
        """
        # Sort cards by value (descending - pay with largest bills first)
        sorted_cards = sorted(self.money, key=lambda c: c.value, reverse=True)

        # Build all reachable amounts with their card combinations
        reachable = {0: []}
        for card in sorted_cards:
            new = {}
            for s, combo in reachable.items():
                ns = s + card.value
                if ns not in reachable and ns not in new:
                    new[ns] = combo + [card]
            reachable.update(new)

        # Exact solution
        if amount in reachable:
            return reachable[amount]

        # Fallback: Overpay with minimum amount
        bigger = [x for x in reachable if x >= amount]
        if bigger:
            return reachable[min(bigger)]

        raise ValueError(f"Cannot pay {amount} with available money cards")

    def __repr__(self) -> str:
        return f"Player({self.name}, Money: {self.get_total_money()}, Animals: {len(self.animals)})"

