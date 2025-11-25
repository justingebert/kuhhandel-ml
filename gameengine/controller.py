from abc import ABC, abstractmethod
from typing import List, Optional
import random

from .game import Game, GamePhase, ActionType, AnimalType
from .actions import GameAction


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


class GameController:
    """
    Manages the flow of the game, enforcing turn order and phase logic.
    Acts as the 'Controller' in MVC.
    """

    def __init__(self, game: Game, agents: List[Agent]):
        if len(agents) != game.num_players:
            raise ValueError(f"Game requires {game.num_players} agents, got {len(agents)}")
        
        self.game = game
        self.agents = agents
        
        # Assign player IDs to agents
        for i, agent in enumerate(self.agents):
            agent.set_player_id(i)

    def run(self, max_turns: int = 1000):
        """Run the full game loop."""
        turn_count = 0
        while not self.game.is_game_over() and turn_count < max_turns:
            turn_count += 1
            self.step()
            
        return self.game.get_scores()

    def step(self):
        """Execute a single step of the game logic."""
        if self.game.phase == GamePhase.PLAYER_TURN:
            self._handle_player_turn()
        elif self.game.phase == GamePhase.AUCTION:
            self._handle_auction_phase()
        elif self.game.phase == GamePhase.COW_TRADE:
            self._handle_cow_trade_phase()
        elif self.game.phase == GamePhase.GAME_OVER:
            return

    def _handle_player_turn(self):
        """Handle the main turn decision (Auction vs Trade)."""
        current_player_idx = self.game.current_player_idx
        agent = self.agents[current_player_idx]
        
        valid_actions = self.game.get_valid_actions()
        if not valid_actions:
            # Should be handled by game logic (skipping), but safety check
            self.game._next_turn()
            return

        action = agent.get_action(self.game, valid_actions)

        if action.type == ActionType.START_AUCTION:
            self.game.start_auction()
        
        elif action.type == ActionType.START_COW_TRADE:
            self.game.start_cow_trade(
                action.target_id,
                action.animal_type,
                action.money_cards
            )

    def _handle_auction_phase(self):
        """
        Handle the dynamic auction loop.
        Iterates through players asking for bids until everyone passes.
        """
        from .actions import BidAction
        
        auctioneer_id = self.game.current_player_idx
        highest_bid_changed = True
        
        # 1. Bidding Loop
        while highest_bid_changed:
            highest_bid_changed = False
            
            # Ask each player (except auctioneer)
            for i in range(self.game.num_players):
                bidder_id = (auctioneer_id + 1 + i) % self.game.num_players
                if bidder_id == auctioneer_id:
                    continue
                
                # Check if player can even bid higher
                min_bid = self.game.auction_high_bid + 10
                bidder_player = self.game.players[bidder_id]
                if bidder_player.get_total_money() < min_bid:
                    continue

                agent = self.agents[bidder_id]
                # We construct a special 'valid_actions' for the agent in this context
                # Ideally Game would provide this, but for now we infer
                valid_actions = [ActionType.BID, ActionType.PASS]
                
                action = agent.get_action(self.game, valid_actions)
                
                if action.type == ActionType.BID:
                    if action.amount >= min_bid:
                        success = self.game.place_bid(bidder_id, action.amount)
                        if success:
                            highest_bid_changed = True
                            # Notify others? (Optional)

        # 2. Auctioneer Decision
        auctioneer_agent = self.agents[auctioneer_id]
        valid_actions = self.game.get_valid_actions(auctioneer_id) # PASS or BUY_AS_AUCTIONEER
        
        action = auctioneer_agent.get_action(self.game, valid_actions)
        
        if action.type == ActionType.BUY_AS_AUCTIONEER:
            self.game.auctioneer_buys()
        else:
            self.game.auctioneer_passes()

    def _handle_cow_trade_phase(self):
        """Handle the response to a cow trade."""
        from .actions import CounterOfferAction
        
        target_id = self.game.trade_target
        agent = self.agents[target_id]
        
        valid_actions = self.game.get_valid_actions(target_id)
        action = agent.get_action(self.game, valid_actions)
        
        if action.type == ActionType.COUNTER_OFFER:
            self.game.counter_trade_offer(action.money_cards)
        else:
            # Default to accept if not countering or invalid
            self.game.accept_trade_offer()
