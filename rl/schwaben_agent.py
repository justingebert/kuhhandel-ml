import random
import numpy as np
from typing import List

from gameengine import Game
from gameengine.actions import GameAction, AuctionBidAction, CowTradeOfferAction, CowTradeCounterOfferAction, ActionType
from gameengine.agent import Agent


class RandomSchwabenAgent(Agent):
    """
    Au Agent that acts similar to a Random Agent but follows a 'Schwaben' distribution
    for money-related actions, preferring smaller indices (cheaper actions).
    """

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        """
        Select an action from valid_actions.
        For money-related actions, lower indices (cheaper) are much more likely.
        For strategy/selection actions, uniform random is used.
        """
        can_start_auction = any(a.type == ActionType.START_AUCTION for a in valid_actions)
        if can_start_auction:
            return valid_actions[0] # Start Auction

        first_action = valid_actions[0]
        

        if first_action.type in [ # hier review fortsetzen
            ActionType.START_COW_TRADE,
            ActionType.COW_TRADE_CHOOSE_OPPONENT,
            ActionType.COW_TRADE_CHOOSE_ANIMAL
        ] or (
            # If we have a mix of Start Auction and Cow Trade, treat as uniform
            any(a.type == ActionType.START_AUCTION for a in valid_actions) and 
            any(a.type == ActionType.COW_TRADE_CHOOSE_OPPONENT for a in valid_actions) 
        ):
             return random.choice(valid_actions)

        # 2. Money Related Cases (Weighted)
        # - Auction Bidding (Pass vs Bid amounts)
        # - Cow Trade Offer
        # - Cow Trade Counter Offer
        # - Auctioneer Decision (Buy vs Accept/Pass)
        
        # Confirm we are in a weighted scenario
        is_money_decision = False
        
        # Check specific types
        # Note: In Auction Bidding, valid_actions includes Pass (Index 0 usually if sorted) and Bids.
        # We need to sort them by "cost".
        
        # Helper to get sort key
        def get_cost_key(action: GameAction) -> int:
            if isinstance(action, AuctionBidAction):
                return action.amount
            if isinstance(action, CowTradeOfferAction):
                return action.amount
            if isinstance(action, CowTradeCounterOfferAction):
                return action.amount
            
            # For Auctioneer decision:
            # Pass/Accept = 0 cost (actually "free" or "get money")
            # Buy = pays money
            if action.type == ActionType.PASS_AS_AUCTIONEER: # "Accept/Pass"
                return 0
            if action.type == ActionType.BUY_AS_AUCTIONEER:
                return 1 # Higher cost than passing
                
            if action.type == ActionType.AUCTION_PASS:
                return -1 # "Cheapest" action (doing nothing)
                
            return 0

        # Attempt to sort valid actions by cost
        # If actions are not money related, keys will be 0, effectively uniform random if we handle it right,
        # but we already handled explicit uniform cases above.
        
        sorted_actions = sorted(valid_actions, key=get_cost_key)
        
        n = len(sorted_actions)
        if n == 1:
            return sorted_actions[0]
            
        # Create Bose-Einstein-like / Geometric / Power-law weights
        # User requested: "verteilung ähnlich einer boseverteilung ... aktionen mit kleinerem index deutlich warscheinlicher"
        # We'll use 1/(index + 1)^2 distribution
        indices = np.arange(n)
        weights = 1.0 / ((indices + 1) ** 2)
        
        # Normalize
        probs = weights / np.sum(weights)
        
        # Select
        chosen_action = np.random.choice(sorted_actions, p=probs)
        return chosen_action
