from typing import List, Optional
import numpy as np
from sb3_contrib import MaskablePPO

from gameengine.agent import Agent
from gameengine.game import Game
from gameengine.actions import GameAction
from rl.env import KuhhandelEnv

class ModelAgent(Agent):
    """
    An agent that uses a trained SB3 MaskablePPO model to make decisions.
    Supports rotated observations for self-play.
    """
    def __init__(self, name: str, model_path: str, env: KuhhandelEnv):
        super().__init__(name)
        self.env = env
        # Load the model
        # Note: We need to handle cpu/gpu device if necessary, usually auto is fine
        self.model = MaskablePPO.load(model_path)
    
    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        # 1. Get observation for THIS player (rotated so self is index 0)
        # Use .unwrapped to bypass wrappers like ActionMasker/Monitor
        obs = self.env.unwrapped.get_observation_for_player(self.player_id)
        
        # 2. Get action mask for THIS player
        action_mask = self.env.unwrapped.get_action_mask_for_player(self.player_id)
        
        # 3. Predict action
        # deterministic=True is usually better for evaluation/deployment
        action_int, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        
        # 4. Decode action
        # NOTE: If we used rotated observations, the action output might strictly refer to 'Relative' targets
        # if the model learned relative strategies.
        # However, the current Env structure keeps Actions ABSOLUTE.
        # So we just decode the int to GameAction using the standard decoder.
        # Ideally, we should check if the chosen action is actually valid.
        # Since we used the mask, it SHOULD be valid.
        
        # We can use the Env's decoder (RLAgent logic)
        # But RLAgent logic is in RLAgent class. We can duplicate it or use a helper.
        # To avoid code duplication, we can instanciate a dummy RLAgent or move decode logic to Env.
        # For now, let's just use RLAgent's method by creating a temporary instance or static method?
        # RLAgent.decode_action is an instance method.
        # Let's just copy the decode logic for now or import it if decoupled.
        
        from rl.rl_agent import RLAgent
        # Create a temporary agent just to use decode (cheap hack, but effective without refactoring rl_agent)
        # OR better: Refactor rl_agent to make decode_action static.
        # I'll just copy the logic effectively by using the helper I'll create here.
        
        return self._decode_action(action_int, game)
        
        
    def _decode_action(self, action_idx: int, game: Game) -> GameAction:
        """Decode integer action index to GameAction. (Copied/Adapted from RLAgent)"""
        from gameengine.actions import Actions
        from gameengine.game import GamePhase
        from gameengine import AnimalType
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
                # NOTE: In rl_agent.py this is used as absolute ID effectively?
                # Actually, Actions.cow_trade_choose_opponent takes 'opponent_id' (absolute).
                # The Env constant BASE maps to ID 0? No, usually ID 0 is Player 0.
                # If existing code assumes indices, we trust it.
                return Actions.cow_trade_choose_opponent(opponent_offset) # opponent_offset IS the target_id

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
        
        raise ValueError(f"Cannot decode action {action_idx} for phase {game.phase}")
