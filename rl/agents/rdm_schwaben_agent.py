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
        if len(valid_actions) == 1: 
            return valid_actions[0]

        if valid_actions[0].type == ActionType.START_AUCTION:
            return valid_actions[0] #Start Auction
        
        if valid_actions[0].type in [ #Select Opponent/Animal
            ActionType.COW_TRADE_CHOOSE_OPPONENT,
            ActionType.COW_TRADE_CHOOSE_ANIMAL
        ]:
             return random.choice(valid_actions)
        
        indices = np.arange(len(valid_actions))
        weights = 1.0 / (indices + 1)
        
        probs = weights / np.sum(weights)
        
        chosen_action = np.random.choice(valid_actions, p=probs)
        return chosen_action