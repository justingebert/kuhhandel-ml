from typing import List

from sb3_contrib import MaskablePPO

from gameengine.actions import GameAction
from gameengine.agent import Agent
from gameengine.game import Game
from rl.env import KuhhandelEnv


class ModelAgent(Agent):
    """
    An agent that uses a trained SB3 MaskablePPO model to make decisions.
    """

    def __init__(self, name: str, model_path: str, env: KuhhandelEnv, model_instance=None):
        super().__init__(name)
        self.env = env
        if model_instance:
            self.model = model_instance
        else:
            self.model = MaskablePPO.load(model_path, device='cpu')
    
    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        if len(valid_actions) == 1:
            return valid_actions[0]

        obs = self.env.unwrapped.get_observation_for_player(self.player_id)

        action_mask = self.env.unwrapped.get_action_mask_for_player(self.player_id)

        action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
        
        return self.env.decode_action(action, game)