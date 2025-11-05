from typing import List, Optional


class MoneyCard:
    """Represents a single money card."""

    def __init__(self, value: int, card_id: int):
        self.value = value
        self.card_id = card_id

    def __repr__(self) -> str:
        return f"Money({self.value})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, MoneyCard):
            return False
        return self.card_id == other.card_id

    def __hash__(self) -> int:
        return hash(self.card_id)


class MoneyDeck:
    """Manages the money cards in the game."""

    # Initial distribution according to rules:
    # 10 x 0, 20 x 10, 10 x 50, 5 x 100, 5 x 200, 5 x 500
    INITIAL_DISTRIBUTION = {
        0: 10,
        10: 20,
        50: 10,
        100: 5,
        200: 5,
        500: 5
    }

    # Starting money per player: 2x0, 4x10, 1x50
    STARTING_MONEY = {
        0: 2,
        10: 4,
        50: 1
    }

    def __init__(self):
        self.cards: List[MoneyCard] = []
        self.next_id = 0
        self._create_deck()

    def _create_deck(self):
        """Create all money cards according to the distribution."""
        for value, count in self.INITIAL_DISTRIBUTION.items():
            for _ in range(count):
                self.cards.append(MoneyCard(value, self.next_id))
                self.next_id += 1

    def get_starting_money(self) -> List[MoneyCard]:
        """Get starting money for one player."""
        starting_cards = []
        for value, count in self.STARTING_MONEY.items():
            for _ in range(count):
                card = self._find_and_remove_card(value)
                if card:
                    starting_cards.append(card)
        return starting_cards

    def get_cards_by_value(self, value: int, count: int = 1) -> List[MoneyCard]:
        """Get specific money cards by value."""
        cards = []
        for _ in range(count):
            card = self._find_and_remove_card(value)
            if card:
                cards.append(card)
        return cards

    def _find_and_remove_card(self, value: int) -> Optional[MoneyCard]:
        """Find and remove a card with the specified value."""
        for card in self.cards:
            if card.value == value:
                self.cards.remove(card)
                return card
        return None

    def has_cards(self) -> bool:
        """Check if there are any cards left."""
        return len(self.cards) > 0


def calculate_total_value(money_cards: List[MoneyCard]) -> int:
    """Calculate the total value of a list of money cards."""
    return sum(card.value for card in money_cards)


def can_afford_bid(money_cards: List[MoneyCard], bid: int) -> bool:
    """Check if a player can afford a bid with their money cards."""
    return calculate_total_value(money_cards) >= bid

