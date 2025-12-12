import random
from typing import List, Dict, Optional, Any
from enum import Enum

from .Animal import AnimalCard, AnimalType
from .Money import MoneyCard, MoneyDeck
from .Player import Player
from .actions import GameAction, Actions


class GamePhase(Enum):
    SETUP = "setup" # brauchen wir noch?

    PLAYER_TURN_CHOICE = 0  # nur umbenannt, von PLAYER_TURN

    AUCTION_BIDDING = 1
    AUCTION_BIDDING_FLY = 1 # in controller.py eingebaut
    AUCTIONEER_DECISION = 2 # in controller.py eingebaut

    COW_TRADE_CHOOSE_ANIMAL = 3
    COW_TRADE_OFFER = 4
    COW_TRADE_RESPONSE = 5
    
    GAME_OVER = "game_over"


class Game:
    """Main game class for Kuhhandel."""

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

        # Step-based auction tracking
        self.auction_current_bidder_idx: Optional[int] = None
        self.auction_bidders_passed: set = set()  # Track who has passed this round

        # trade state
        self.trade_initiator: Optional[int] = None
        self.trade_target: Optional[int] = None
        self.trade_animal_type: Optional[AnimalType] = None
        self.trade_offer: int = 0
        self.trade_counter_offer: int = 0
        self.trade_offer_card_count: int = 0

        # Track last completed trade result for reward calculation
        # (winner_player_id, loser_player_id, animals_transferred, offer, counter_offer, net_payment)
        self.last_trade_result: Optional[Dict[str, Any]] = None

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

        self.phase = GamePhase.PLAYER_TURN_CHOICE
        self.current_player_idx = 0

    def get_current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.current_player_idx]

    def get_valid_actions(self, player_id: Optional[int] = None) -> List[GameAction]:
        """Get valid actions for the current game state."""


        if player_id is None:
            player_id = self.current_player_idx

        player = self.players[player_id]
        actions = []

        if self.phase == GamePhase.PLAYER_TURN_CHOICE:
            # When deck is empty, cow trades are MANDATORY if incomplete sets exist
            deck_empty = len(self.animal_deck) == 0

            if not deck_empty:
                actions.append(Actions.start_auction())

            # Check if cow trade is possible - player can choose to start a cow trade
            # by selecting an opponent (if any valid targets exist)
            can_start_cow_trade = False
            for animal_type in AnimalType.get_all_types():
                if player.has_animal_type(animal_type):
                    if player.has_complete_set(animal_type):
                        continue
                    for other_player in self.players:
                        if other_player.player_id != player_id:
                            if other_player.has_animal_type(animal_type):
                                if not other_player.has_complete_set(animal_type):
                                    can_start_cow_trade = True
                                    break
                    if can_start_cow_trade:
                        break

            # Add cow trade opponent choices
            if can_start_cow_trade:
                # Get all valid opponents (players who share at least one incomplete animal type)
                valid_opponents = set()
                for animal_type in AnimalType.get_all_types():
                    if player.has_animal_type(animal_type) and not player.has_complete_set(animal_type):
                        for other_player in self.players:
                            if other_player.player_id != player_id:
                                if other_player.has_animal_type(animal_type):
                                    if not other_player.has_complete_set(animal_type):
                                        valid_opponents.add(other_player.player_id)

                for opponent_id in valid_opponents:
                    actions.append(Actions.cow_trade_choose_opponent(opponent_id))

            # If deck is empty and no actions available, but game isn't over,
            # the player must pass (should usually not happen if logic is correct)
            if deck_empty and not actions and not self.is_game_over():
                pass

        elif self.phase == GamePhase.AUCTION_BIDDING:
            # Only non-auctioneers can bid during this phase
            if player_id != self.current_player_idx:
                # Other players can bid or pass
                actions.append(Actions.pass_action())

                # Generate valid bids
                # Valid bid > current high bid
                # Step size 10 is standard convention
                start_bid = self.auction_high_bid + 10

                total_money = self.players[player_id].get_total_money()

                money_steps_mod10 = range(start_bid%10,total_money%10+1)

                for amount in money_steps_mod10:
                    actions.append(Actions.bid(amount=10*amount))

        elif self.phase == GamePhase.AUCTIONEER_DECISION:
            # Only auctioneer (current player) can make decision
            if player_id == self.current_player_idx:
                # Auctioneer can always pass (sell to highest bidder)
                actions.append(Actions.pass_as_auctioneer())

                # Can only buy if there was a bid
                if self.auction_high_bidder is not None:
                    # Can only buy if auctioneer can afford to match the highest bid
                    if player.get_total_money() >= self.auction_high_bid:
                        actions.append(Actions.buy_as_auctioneer())

        elif self.phase == GamePhase.COW_TRADE_CHOOSE_ANIMAL:
            # Only current player can choose animal type
            if player_id == self.current_player_idx:
                target = self.players[self.trade_target]
                for animal_type in AnimalType.get_all_types():
                    if player.has_animal_type(animal_type) and not player.has_complete_set(animal_type):
                        if target.has_animal_type(animal_type) and not target.has_complete_set(animal_type):
                            actions.append(Actions.cow_trade_choose_animal(animal_type))

        elif self.phase == GamePhase.COW_TRADE_OFFER:
            # Only current player can make offer
            if player_id == self.current_player_idx:
                # Generate all possible offer amounts based on player's money
                total_money = player.get_total_money()
                # Offer can be any amount from 0 to total money (step of 10)
                for amount in range(0, total_money + 1, 10):
                    actions.append(Actions.cow_trade_offer(amount))

        elif self.phase == GamePhase.COW_TRADE_RESPONSE:
            # Only trade target can respond
            if player_id == self.trade_target:
                # Target makes counter offer - can be any amount from 0 to their total money
                total_money = player.get_total_money()
                for amount in range(0, total_money + 1, 10):
                    actions.append(Actions.counter_offer(amount=amount))

        return actions

    def start_auction(self) -> Optional[AnimalCard]:
        """Start an auction by drawing the top animal card."""
        #TODO check if this should raise an error if deck is empty
        if not self.animal_deck:
            return None

        self.current_animal = self.animal_deck.pop(0)
        self.phase = GamePhase.AUCTION_BIDDING
        self.auction_bids.clear()
        self.auction_high_bidder = None
        self.auction_high_bid = 0

        # Initialize step-based auction tracking
        # First bidder is the player after the auctioneer
        self.auction_current_bidder_idx = (self.current_player_idx + 1) % self.num_players

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
        if self.phase != GamePhase.AUCTION_BIDDING:
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
        if self.phase != GamePhase.AUCTIONEER_DECISION:
            return False

        if self.auction_high_bidder is None:
            # No bids, auctioneer gets it for free
            self.get_current_player().add_animal(self.current_animal)
            self.end_auction()
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
            self.end_auction()
            return True

        return False

    def auctioneer_passes(self) -> bool:
        """Auctioneer passes, high bidder gets the animal."""
        if self.phase != GamePhase.AUCTIONEER_DECISION:
            return False

        if self.auction_high_bidder is None:
            # No bids, auctioneer gets it for free
            self.get_current_player().add_animal(self.current_animal)
            self.end_auction()
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
            self.end_auction()
            return True

        return False

    def _make_payment(self, payer: Player, receiver: Player, amount: int) -> bool:
        """Make a payment from one player to another."""
        # Find cards that sum to at least the amount
        payment_cards = payer.select_payment_cards(amount)
        if not payment_cards:
            return False

        payer.remove_money(payment_cards)
        receiver.add_money(payment_cards)
        return True


    def end_auction_bidding(self):
        """End the auction bidding phase."""
        self.auction_current_bidder_idx = None
        self.phase = GamePhase.AUCTIONEER_DECISION

    def end_auction(self):
        """End the auction and move to next turn."""
        self.current_animal = None
        self.auction_bids.clear()
        self.auction_high_bidder = None
        self.auction_high_bid = 0
        self.auction_current_bidder_idx = None
        self.auction_bidders_passed.clear()
        self._next_turn()

    def get_current_decision_player(self) -> int:
        """Get the player who needs to make a decision right now.
        
        During auction bidding, this is the current bidder (not the auctioneer).
        During other phases, this is the current player.
        """
        if self.phase == GamePhase.AUCTION_BIDDING:
            return self.auction_current_bidder_idx
        elif self.phase == GamePhase.COW_TRADE_RESPONSE:
            return self.trade_target
        else:
            return self.current_player_idx

    def advance_auction_bidder(self):
        """Move to the next bidder in the auction rotation."""
        auctioneer = self.current_player_idx
        current = self.auction_current_bidder_idx
        
        # Find next non-auctioneer player
        next_bidder = (current + 1) % self.num_players
        if next_bidder == auctioneer:
            next_bidder = (next_bidder + 1) % self.num_players
        
        self.auction_current_bidder_idx = next_bidder

    def reset_auction_passes(self):
        """Reset pass tracking when a new bid is placed."""
        self.auction_bidders_passed.clear()

    def is_auction_round_complete(self) -> bool:
        """Check if all non-auctioneer players have passed."""
        auctioneer = self.current_player_idx
        non_auctioneer_count = self.num_players - 1
        
        # Check if all non-auctioneers have passed
        if len(self.auction_bidders_passed) >= non_auctioneer_count:
            return True
        
        # Also check if no one can afford to outbid
        min_bid = self.auction_high_bid + 10
        for i in range(self.num_players):
            if i != auctioneer and i not in self.auction_bidders_passed:
                if self.players[i].get_total_money() >= min_bid:
                    return True
        
        return False

    def choose_cow_trade_opponent(self, target_id: int):
        self.trade_initiator = self.current_player_idx
        self.trade_target = target_id
        self.phase = GamePhase.COW_TRADE_CHOOSE_ANIMAL


    def choose_cow_trade_animal(self, animal_type: AnimalType):
        self.trade_animal_type = animal_type
        self.phase = GamePhase.COW_TRADE_OFFER

    def choose_cow_trade_offer(self, amount: int):
        self.trade_offer = amount

        # Set how many cards are layed down for the offer
        if amount > 0:
            initiator = self.players[self.trade_initiator]
            offer_cards = initiator.select_payment_cards(amount)
            self.trade_offer_card_count = len(offer_cards) if offer_cards else 0

        self.phase = GamePhase.COW_TRADE_RESPONSE

    def choose_cow_trade_counter_offer(self, amount: int):
        self.trade_counter_offer = amount

    def execute_cow_trade(self) -> bool:
        """Execute cow trade using game state after counter offer has been made.

        Trade rules:
        - If one player has only 1 animal: trade is for 1 animal (even if other has more)
        - If both have 2 or more: trade is for min(initiator_count, target_count)
        - Winner takes that many animals from loser
        """
        if self.phase != GamePhase.COW_TRADE_RESPONSE:
            return False

        initiator = self.players[self.trade_initiator]
        target = self.players[self.trade_target]

        # Validate trade is possible
        if not initiator.has_animal_type(self.trade_animal_type):
            return False
        if not target.has_animal_type(self.trade_animal_type):
            return False
        if target.has_complete_set(self.trade_animal_type):
            return False

        #TODO maybe check if player has enough money if not return false?

        self._log_action("start_cow_trade", {
            "initiator": initiator.player_id,
            "target": self.trade_target,
            "animal": self.trade_animal_type.display_name,
            "offer": self.trade_offer,
            "counter_offer": self.trade_counter_offer
        })


        # Exchange money - select cards from amounts and transfer
        if self.trade_offer > 0:
            offer_cards = initiator.select_payment_cards(self.trade_offer)
            initiator.remove_money(offer_cards)
            target.add_money(offer_cards)

        if self.trade_counter_offer > 0:
            counter_cards = target.select_payment_cards(self.trade_counter_offer)
            target.remove_money(counter_cards)
            initiator.add_money(counter_cards)

        # Determine trade amount: min of what both players have
        initiator_count = initiator.get_animal_count(self.trade_animal_type)
        target_count = target.get_animal_count(self.trade_animal_type)
        trade_amount = min(initiator_count, target_count)

        # Determine winner and transfer animals
        animals = []
        winner = "initiator"  # Default in case of tie
        winner_player_id = self.trade_initiator
        if self.trade_offer >= self.trade_counter_offer:
            # Initiator wins - takes trade_amount animals from target
            # TIE -> initiator wins
            animals = target.remove_animals(self.trade_animal_type, trade_amount)
            for animal in animals:
                initiator.add_animal(animal)
        else:
            # Target wins - takes trade_amount animals from initiator
            animals = initiator.remove_animals(self.trade_animal_type, trade_amount)
            for animal in animals:
                target.add_animal(animal)
            winner = "target"
            winner_player_id = self.trade_target

        # Store trade result for reward calculation (persists after _end_cow_trade)
        self.last_trade_result = {
            "winner_player_id": winner_player_id,
            "loser_player_id": self.trade_target if winner == "initiator" else self.trade_initiator,
            "animals_transferred": len(animals),
            "offer": self.trade_offer,
            "counter_offer": self.trade_counter_offer,
            "net_payment": self.trade_offer - self.trade_counter_offer,  # positive = initiator paid more
        }

        self._log_action("resolve_trade", {
            "winner": winner,
            "offer": self.trade_offer,
            "counter": self.trade_counter_offer,
            "animals_transferred": len(animals)
        })

        self._end_cow_trade()

        return True


    def _end_cow_trade(self):
        """End the cow trade and move to next turn."""
        self.trade_initiator = None
        self.trade_target = None
        self.trade_animal_type = None
        self.trade_offer = 0
        self.trade_counter_offer = 0
        self.trade_offer_card_count = 0
        self._next_turn()

    def _next_turn(self):
        """Move to the next player's turn."""
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        self.round_number += 1

        if self.is_game_over():
            self.phase = GamePhase.GAME_OVER
        else:
            self.phase = GamePhase.PLAYER_TURN_CHOICE

            # If deck is empty, automatically skip players with no valid actions
            # (those who only have complete sets)
            if not self.animal_deck:
                max_skips = self.num_players  # Safety to prevent infinite loop
                skips = 0
                while skips < max_skips:
                    valid_actions = self.get_valid_actions()
                    if valid_actions or self.is_game_over():
                        break
                    # Skip this player - they have only complete sets
                    self.current_player_idx = (self.current_player_idx + 1) % self.num_players
                    self.round_number += 1
                    skips += 1

                # Double-check if game is now over after skipping
                if self.is_game_over():
                    self.phase = GamePhase.GAME_OVER

    def is_game_over(self) -> bool:
        """Check if the game is over.

        Game ends when all animals are distributed in complete sets of 4.
        No animals left in deck AND all animals must be in complete sets.
        """
        if self.animal_deck:
            return False

        # Check if all animals are in complete sets
        for player in self.players:
            animal_counts = player.get_animal_counts()
            for animal_type, count in animal_counts.items():
                if count != 4:
                    # Incomplete set found - game must continue
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

