from typing import Optional, List

import gymnasium

from gameengine import Game, AnimalType
from gameengine.actions import GameAction, ActionType, Actions
from gameengine.agent import Agent
from gameengine.game import GamePhase


class RLAgent(Agent):
    """Agent wrapper for RL model - converts int actions to GameActions.
    
    The RLAgent stores the integer action chosen by the RL policy and
    converts it to a GameAction when the controller asks for it.
    """

    def __init__(self, name: str, env: 'gymnasium.Env'):
        super().__init__(name)
        self.env = env
        self.last_action_int: Optional[int] = None

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        """Convert stored integer action to GameAction based on current game phase."""
        action_idx = self.last_action_int
        decoded_action = self.decode_action(action_idx)
        return decoded_action
        
    def decode_action(self, action_idx: int) -> GameAction:
        """Decode integer action index to GameAction."""
        # Import action constants from env
        from rl.env import (
            ACTION_START_AUCTION, ACTION_COW_CHOOSE_OPP_BASE, ACTION_COW_CHOOSE_OPP_END,
            ACTION_AUCTION_BID_BASE, AUCTION_BID_END, MONEY_STEP,
            ACTION_AUCTIONEER_ACCEPT, ACTION_AUCTIONEER_BUY,
            ACTION_COW_CHOOSE_ANIMAL_BASE, ACTION_COW_CHOOSE_ANIMAL_END,
            ACTION_COW_OFFER_BASE, ACTION_COW_OFFER_END,
            ACTION_COW_RESP_COUNTER_BASE, COW_RESP_COUNTER_END
        )
        
        if game.phase == GamePhase.PLAYER_TURN_CHOICE:
            if action_idx == ACTION_START_AUCTION:
                return Actions.start_auction()
            elif ACTION_COW_CHOOSE_OPP_BASE <= action_idx <= ACTION_COW_CHOOSE_OPP_END:
                opponent_offset = action_idx - ACTION_COW_CHOOSE_OPP_BASE
                return Actions.cow_trade_choose_opponent(opponent_offset)
        
        elif game.phase == GamePhase.AUCTION_BIDDING:
            if ACTION_AUCTION_BID_BASE <= action_idx <= AUCTION_BID_END:
                bid_level = action_idx - ACTION_AUCTION_BID_BASE
                if bid_level == 0:
                    return Actions.pass_action()
                else:
                    bid_amount = bid_level * MONEY_STEP
                    return Actions.bid(bid_amount)
        
        elif game.phase == GamePhase.AUCTIONEER_DECISION:
            if action_idx == ACTION_AUCTIONEER_ACCEPT:
                return Actions.pass_as_auctioneer()
            elif action_idx == ACTION_AUCTIONEER_BUY:
                return Actions.buy_as_auctioneer()
        
        elif game.phase == GamePhase.COW_TRADE_CHOOSE_ANIMAL:
            if ACTION_COW_CHOOSE_ANIMAL_BASE <= action_idx <= ACTION_COW_CHOOSE_ANIMAL_END:
                animal_idx = action_idx - ACTION_COW_CHOOSE_ANIMAL_BASE
                animal_type = AnimalType.get_all_types()[animal_idx]
                return Actions.cow_trade_choose_animal(animal_type)
        
        elif game.phase == GamePhase.COW_TRADE_OFFER:
            if ACTION_COW_OFFER_BASE <= action_idx <= ACTION_COW_OFFER_END:
                offer_level = action_idx - ACTION_COW_OFFER_BASE
                offer_amount = offer_level * MONEY_STEP
                return Actions.cow_trade_offer(offer_amount)
        
        elif game.phase == GamePhase.COW_TRADE_RESPONSE:
            if ACTION_COW_RESP_COUNTER_BASE <= action_idx <= COW_RESP_COUNTER_END:
                counter_level = action_idx - ACTION_COW_RESP_COUNTER_BASE
                counter_amount = counter_level * MONEY_STEP
                return Actions.counter_offer(counter_amount)
        
        # Fallback: return first valid action if decoding fails
        if valid_actions:
            return valid_actions[0]
        
        raise ValueError(f"Cannot decode action {action_idx} for phase {game.phase}")
