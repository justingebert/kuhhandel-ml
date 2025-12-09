from typing import List

from .agent import Agent
from .game import Game, GamePhase
from .actions import ActionType


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
        if self.game.phase == GamePhase.PLAYER_TURN_CHOICE:
            self._handle_player_turn()
        elif self.game.phase == GamePhase.AUCTION_BIDDING:
            self._handle_auction_bidding()
        elif self.game.phase == GamePhase.AUCTIONEER_DECISION:
            self._handle_auctioneer_decision()
        elif self.game.phase == GamePhase.COW_TRADE_CHOOSE_ANIMAL:
            self._handle_cow_trade_choose_animal()
        elif self.game.phase == GamePhase.COW_TRADE_OFFER:
            self._handle_cow_trade_offer()
        elif self.game.phase == GamePhase.COW_TRADE_RESPONSE:
            self._handle_cow_trade_response()
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
        
        elif action.type == ActionType.COW_TRADE_CHOOSE_OPPONENT:
            self.game.choose_cow_trade_opponent(action.target_id)

    def _handle_auction_bidding(self):
        """
        Handle ONE bid decision during auction.
        Called repeatedly until all players have passed.
        """
        # Check if auction round is complete
        if self.game.is_auction_round_complete():
            self.game.end_auction_bidding()
            return

        bidder_id = self.game.auction_current_bidder_idx
        auctioneer_id = self.game.current_player_idx

        # Skip if this bidder can't afford the minimum bid
        min_bid = self.game.auction_high_bid + 10
        bidder_player = self.game.players[bidder_id]
        if bidder_player.get_total_money() < min_bid:
            self.game.auction_bidders_passed.add(bidder_id)
            self.game.advance_auction_bidder()
            return

        # Skip if this bidder has already passed this round
        if bidder_id in self.game.auction_bidders_passed:
            self.game.advance_auction_bidder()
            return

        # Get action from current bidder
        agent = self.agents[bidder_id]
        valid_actions = self.game.get_valid_actions(bidder_id)
        
        action = agent.get_action(self.game, valid_actions)

        if action.type == ActionType.AUCTION_BID:
            bid_amount = action.amount
            if bid_amount >= min_bid:
                success = self.game.place_bid(bidder_id, bid_amount)
                if success:
                    # New bid placed - reset passes so others can bid again
                    self.game.reset_auction_passes()
        else:
            # Player passed
            self.game.auction_bidders_passed.add(bidder_id)

        # Move to next bidder
        self.game.advance_auction_bidder()


    def _handle_auctioneer_decision(self):
        """
        Handle the auctioneer's decision to buy the animal or pass.
        """
        auctioneer_id = self.game.current_player_idx
        auctioneer_agent = self.agents[auctioneer_id]
        valid_actions = self.game.get_valid_actions(auctioneer_id)
        
        action = auctioneer_agent.get_action(self.game, valid_actions)
        
        if action.type == ActionType.BUY_AS_AUCTIONEER:
            self.game.auctioneer_buys()
        else:
            self.game.auctioneer_passes()

    def _handle_cow_trade_choose_animal(self):
        agent = self.agents[self.game.current_player_idx]

        valid_actions = self.game.get_valid_actions(self.game.current_player_idx)
        action = agent.get_action(self.game, valid_actions)

        self.game.choose_cow_trade_animal(action.animal_type)

    def _handle_cow_trade_offer(self):
        agent = self.agents[self.game.current_player_idx]

        valid_actions = self.game.get_valid_actions(self.game.current_player_idx)
        action = agent.get_action(self.game, valid_actions)

        self.game.choose_cow_trade_offer(action.amount)


    def _handle_cow_trade_response(self):
        agent = self.agents[self.game.trade_target]
        valid_actions = self.game.get_valid_actions(self.game.trade_target)
        action = agent.get_action(self.game, valid_actions)
        self.game.choose_cow_trade_counter_offer(action.amount)

        self.game.execute_cow_trade()
