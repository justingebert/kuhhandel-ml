import random
from typing import List, Dict, Optional, Any
from enum import Enum

from .Animal import AnimalCard, AnimalType
from .Money import MoneyCard, MoneyDeck, calculate_total_value
from .Player import Player


class GamePhase(Enum):
    """Different phases of the game."""
    SETUP = "setup"
    PLAYER_TURN = "player_turn"
    AUCTION = "auction"
    COW_TRADE = "cow_trade"
    GAME_OVER = "game_over"


class ActionType(Enum):
    """Types of actions players can take."""
    START_AUCTION = "start_auction"
    START_COW_TRADE = "start_cow_trade"
    BID = "bid"
    PASS = "pass"
    BUY_AS_AUCTIONEER = "buy_as_auctioneer"
    ACCEPT_OFFER = "accept_offer"
    COUNTER_OFFER = "counter_offer"


class Game:
    """Main game class for You're Bluffing."""

    def __init__(self, num_players: int = 3, seed: Optional[int] = None):
        if num_players < 3 or num_players > 5:
            raise ValueError("Game requires 3-5 players")

        self.num_players = num_players
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Game state
        self.phase = GamePhase.SETUP
        self.players: List[Player] = []
        self.current_player_idx = 0
        self.round_number = 0

        # Card decks
        self.animal_deck: List[AnimalCard] = []
        self.money_deck = MoneyDeck()

        # Current action state
        self.current_animal: Optional[AnimalCard] = None
        self.auction_bids: Dict[int, int] = {}  # player_id -> bid
        self.auction_high_bidder: Optional[int] = None
        self.auction_high_bid: int = 0

        # trade state
        self.trade_initiator: Optional[int] = None
        self.trade_target: Optional[int] = None
        self.trade_animal_type: Optional[AnimalType] = None
        self.trade_offer: List[MoneyCard] = []
        self.trade_counter_offer: List[MoneyCard] = []

        # Donkey counter for additional money distribution
        self.donkeys_revealed = 0

        # History for ML training
        self.action_history: List[Dict[str, Any]] = []

    def setup(self):
        """Initialize the game."""
        # Create players
        for i in range(self.num_players):
            player = Player(i)
            self.players.append(player)

        # Create and shuffle animal deck (4 of each type)
        card_id = 0
        for animal_type in AnimalType.get_all_types():
            for _ in range(4):
                self.animal_deck.append(AnimalCard(animal_type, card_id))
                card_id += 1
        random.shuffle(self.animal_deck)

        # Deal starting money to each player
        for player in self.players:
            starting_money = self.money_deck.get_starting_money()
            player.add_money(starting_money)

        self.phase = GamePhase.PLAYER_TURN
        self.current_player_idx = 0

    def get_current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.current_player_idx]

    def get_valid_actions(self, player_id: Optional[int] = None) -> List[ActionType]:
        """Get valid actions for the current game state."""
        if player_id is None:
            player_id = self.current_player_idx

        player = self.players[player_id]
        actions = []

        if self.phase == GamePhase.PLAYER_TURN:
            if self.animal_deck:
                actions.append(ActionType.START_AUCTION)

            # Check if cow trade is possible
            for animal_type in AnimalType.get_all_types():
                if player.has_animal_type(animal_type):
                    # Can't trade if current player has complete set
                    if player.has_complete_set(animal_type):
                        continue
                    # Check if any other player has the same type
                    for other_player in self.players:
                        if other_player.player_id != player_id:
                            if other_player.has_animal_type(animal_type):
                                # Can only trade if other player doesn't have complete set
                                if not other_player.has_complete_set(animal_type):
                                    actions.append(ActionType.START_COW_TRADE)
                                    break
                    if ActionType.START_COW_TRADE in actions:
                        break

        elif self.phase == GamePhase.AUCTION:
            if player_id == self.current_player_idx:
                # Auctioneer can always pass, and buy if there's a high bid
                actions.append(ActionType.PASS)
                if self.auction_high_bidder is not None:
                    actions.append(ActionType.BUY_AS_AUCTIONEER)
            else:
                # Other players can bid
                actions.append(ActionType.BID)
                actions.append(ActionType.PASS)

        elif self.phase == GamePhase.COW_TRADE:
            if player_id == self.trade_target and not self.trade_counter_offer:
                # Target can accept or counter
                actions.append(ActionType.ACCEPT_OFFER)
                actions.append(ActionType.COUNTER_OFFER)

        return actions

    def start_auction(self) -> Optional[AnimalCard]:
        """Start an auction by drawing the top animal card."""
        if not self.animal_deck:
            return None

        self.current_animal = self.animal_deck.pop(0)
        self.phase = GamePhase.AUCTION
        self.auction_bids.clear()
        self.auction_high_bidder = None
        self.auction_high_bid = 0

        # Check for donkey
        if self.current_animal.animal_type == AnimalType.DONKEY:
            self._handle_donkey()

        self._log_action("start_auction", {
            "animal": self.current_animal.animal_type.display_name,
            "player": self.current_player_idx
        })

        return self.current_animal

    def _handle_donkey(self):
        """Handle donkey card - distribute additional money."""
        self.donkeys_revealed += 1

        # Distribute money based on which donkey this is
        money_values = {1: 50, 2: 100, 3: 200, 4: 500}
        value = money_values.get(self.donkeys_revealed, 0)

        if value > 0:
            for player in self.players:
                money_cards = self.money_deck.get_cards_by_value(value, 1)
                player.add_money(money_cards)

            self._log_action("donkey_money", {
                "donkey_number": self.donkeys_revealed,
                "value": value
            })

    def place_bid(self, player_id: int, bid_amount: int) -> bool:
        """Place a bid during an auction."""
        if self.phase != GamePhase.AUCTION:
            return False

        player = self.players[player_id]

        # Check if player can afford the bid
        if player.get_total_money() < bid_amount:
            return False

        # Bid must be higher than current high bid
        if bid_amount <= self.auction_high_bid:
            return False

        self.auction_high_bid = bid_amount
        self.auction_high_bidder = player_id
        self.auction_bids[player_id] = bid_amount

        self._log_action("bid", {
            "player": player_id,
            "amount": bid_amount
        })

        return True

    def auctioneer_buys(self) -> bool:
        """Auctioneer decides to buy at the highest bid."""
        if self.phase != GamePhase.AUCTION:
            return False

        if self.auction_high_bidder is None:
            # No bids, auctioneer gets it for free
            self.get_current_player().add_animal(self.current_animal)
            self._end_auction()
            return True

        auctioneer = self.get_current_player()
        high_bidder = self.players[self.auction_high_bidder]

        # Auctioneer pays the high bidder
        payment = self._make_payment(auctioneer, high_bidder, self.auction_high_bid)
        if payment:
            auctioneer.add_animal(self.current_animal)
            self._log_action("auctioneer_buys", {
                "auctioneer": auctioneer.player_id,
                "paid": self.auction_high_bid,
                "to": high_bidder.player_id
            })
            self._end_auction()
            return True

        return False

    def auctioneer_passes(self) -> bool:
        """Auctioneer passes, high bidder gets the animal."""
        if self.phase != GamePhase.AUCTION:
            return False

        if self.auction_high_bidder is None:
            # No bids, auctioneer gets it for free
            self.get_current_player().add_animal(self.current_animal)
            self._end_auction()
            return True

        auctioneer = self.get_current_player()
        high_bidder = self.players[self.auction_high_bidder]

        # High bidder pays auctioneer and gets animal
        payment = self._make_payment(high_bidder, auctioneer, self.auction_high_bid)
        if payment:
            high_bidder.add_animal(self.current_animal)
            self._log_action("high_bidder_wins", {
                "bidder": high_bidder.player_id,
                "paid": self.auction_high_bid,
                "to": auctioneer.player_id
            })
            self._end_auction()
            return True

        return False

    def _make_payment(self, payer: Player, receiver: Player, amount: int) -> bool:
        """Make a payment from one player to another."""
        # Find cards that sum to at least the amount
        payment_cards = self._select_payment_cards(payer.money, amount)
        if not payment_cards:
            return False

        payer.remove_money(payment_cards)
        receiver.add_money(payment_cards)
        return True

    def _select_payment_cards(self, money_cards: List[MoneyCard], amount: int) -> List[MoneyCard]:
        """Select money cards to pay a specific amount (or more if exact not possible)."""
        # Sort cards by value
        sorted_cards = sorted(money_cards, key=lambda c: c.value)

        # Try to find exact match first
        for i in range(len(sorted_cards)):
            for j in range(i + 1, len(sorted_cards) + 1):
                subset = sorted_cards[i:j]
                total = calculate_total_value(subset)
                if total == amount:
                    return subset

        # If no exact match, find smallest overpayment
        for i in range(len(sorted_cards)):
            for j in range(i + 1, len(sorted_cards) + 1):
                subset = sorted_cards[i:j]
                total = calculate_total_value(subset)
                if total >= amount:
                    return subset

        return []

    def _end_auction(self):
        """End the auction and move to next turn."""
        self.current_animal = None
        self.auction_bids.clear()
        self.auction_high_bidder = None
        self.auction_high_bid = 0
        self._next_turn()

    def start_cow_trade(self, target_player_id: int, animal_type: AnimalType,
                       offer_cards: List[MoneyCard]) -> bool:
        """Start a cow trade with another player."""
        if self.phase != GamePhase.PLAYER_TURN:
            return False

        initiator = self.get_current_player()
        target = self.players[target_player_id]

        # Validate trade is possible
        if not initiator.has_animal_type(animal_type):
            return False
        if not target.has_animal_type(animal_type):
            return False
        if target.has_complete_set(animal_type):
            return False

        # Check initiator has the cards they're offering
        if not all(card in initiator.money for card in offer_cards):
            return False

        self.phase = GamePhase.COW_TRADE
        self.trade_initiator = initiator.player_id
        self.trade_target = target_player_id
        self.trade_animal_type = animal_type
        self.trade_offer = offer_cards
        self.trade_counter_offer = []

        self._log_action("start_cow_trade", {
            "initiator": initiator.player_id,
            "target": target_player_id,
            "animal": animal_type.display_name,
            "offer": calculate_total_value(offer_cards)
        })

        return True

    def accept_trade_offer(self) -> bool:
        """Target player accepts the trade offer."""
        if self.phase != GamePhase.COW_TRADE:
            return False

        target = self.players[self.trade_target]
        initiator = self.players[self.trade_initiator]

        # Transfer money and animal
        initiator.remove_money(self.trade_offer)
        target.add_money(self.trade_offer)

        animals_to_transfer = target.remove_animals(self.trade_animal_type, 1)
        for animal in animals_to_transfer:
            initiator.add_animal(animal)

        self._log_action("accept_trade", {
            "target": target.player_id,
            "initiator": initiator.player_id,
            "animal": self.trade_animal_type.display_name
        })

        self._end_cow_trade()
        return True

    def counter_trade_offer(self, counter_cards: List[MoneyCard]) -> bool:
        """Target player makes a counter offer."""
        if self.phase != GamePhase.COW_TRADE:
            return False

        target = self.players[self.trade_target]

        # Check target has the cards
        if not all(card in target.money for card in counter_cards):
            return False

        self.trade_counter_offer = counter_cards

        # Resolve the trade
        self._resolve_cow_trade()
        return True

    def _resolve_cow_trade(self):
        """Resolve a cow trade after counter offer."""
        initiator = self.players[self.trade_initiator]
        target = self.players[self.trade_target]

        offer_value = calculate_total_value(self.trade_offer)
        counter_value = calculate_total_value(self.trade_counter_offer)

        # Exchange money
        initiator.remove_money(self.trade_offer)
        target.add_money(self.trade_offer)
        target.remove_money(self.trade_counter_offer)
        initiator.add_money(self.trade_counter_offer)

        # Determine winner
        if offer_value > counter_value:
            # Initiator wins
            count = min(target.get_animal_count(self.trade_animal_type),
                       1 if initiator.get_animal_count(self.trade_animal_type) == 1
                       else 2)
            animals = target.remove_animals(self.trade_animal_type, count)
            for animal in animals:
                initiator.add_animal(animal)
            winner = "initiator"
        elif counter_value > offer_value:
            # Target wins
            count = min(initiator.get_animal_count(self.trade_animal_type),
                       1 if target.get_animal_count(self.trade_animal_type) == 1
                       else 2)
            animals = initiator.remove_animals(self.trade_animal_type, count)
            for animal in animals:
                target.add_animal(animal)
            winner = "target"
        else:
            # Tie - would need to bid again (simplified: initiator wins)
            count = min(target.get_animal_count(self.trade_animal_type), 1)
            animals = target.remove_animals(self.trade_animal_type, count)
            for animal in animals:
                initiator.add_animal(animal)
            winner = "initiator (tie)"

        self._log_action("resolve_trade", {
            "winner": winner,
            "offer": offer_value,
            "counter": counter_value
        })

        self._end_cow_trade()

    def _end_cow_trade(self):
        """End the cow trade and move to next turn."""
        self.trade_initiator = None
        self.trade_target = None
        self.trade_animal_type = None
        self.trade_offer = []
        self.trade_counter_offer = []
        self._next_turn()

    def _next_turn(self):
        """Move to the next player's turn."""
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        self.round_number += 1

        if self.is_game_over():
            self.phase = GamePhase.GAME_OVER
        else:
            self.phase = GamePhase.PLAYER_TURN

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Game is over when no animals left and no valid trades possible
        if self.animal_deck:
            return False

        # Check if any cow trades are possible
        for player in self.players:
            for animal_type in AnimalType.get_all_types():
                if player.has_animal_type(animal_type):
                    # Skip if current player has complete set
                    if player.has_complete_set(animal_type):
                        continue
                    for other_player in self.players:
                        if other_player.player_id != player.player_id:
                            if other_player.has_animal_type(animal_type):
                                # Can trade if other player doesn't have complete set
                                if not other_player.has_complete_set(animal_type):
                                    return False

        return True

    def get_scores(self) -> Dict[int, int]:
        """Get final scores for all players."""
        return {player.player_id: player.calculate_score()
                for player in self.players}

    def get_winner(self) -> Optional[Player]:
        """Get the winner of the game."""
        if not self.is_game_over():
            return None

        scores = self.get_scores()
        max_score = max(scores.values())
        winner_id = [pid for pid, score in scores.items() if score == max_score][0]
        return self.players[winner_id]

    def _log_action(self, action_type: str, details: Dict[str, Any]):
        """Log an action for ML training purposes."""
        self.action_history.append({
            "round": self.round_number,
            "action": action_type,
            "details": details
        })

    def get_state(self) -> Dict[str, Any]:
        """Get current game state (useful for ML agents)."""
        return {
            "phase": self.phase.value,
            "current_player": self.current_player_idx,
            "round": self.round_number,
            "animals_remaining": len(self.animal_deck),
            "current_animal": self.current_animal.animal_type.display_name if self.current_animal else None,
            "auction_high_bid": self.auction_high_bid,
            "players": [
                {
                    "id": p.player_id,
                    "money": p.get_total_money(),
                    "animals": p.get_animal_counts(),
                    "score": p.calculate_score()
                }
                for p in self.players
            ]
        }

    def __repr__(self) -> str:
        return f"Game(Phase: {self.phase.value}, Round: {self.round_number}, Current: Player {self.current_player_idx})"

