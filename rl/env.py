from typing import Optional, List

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Box

from gameengine import AnimalType, MoneyDeck
from gameengine.agent import Agent
from gameengine.controller import GameController
from gameengine.game import Game, GamePhase
from rl.rl_agent import RLAgent
from tests.demo_game import RandomAgent


class KuhhandelEnv(gym.Env):

    def __init__(self, num_players: int = 3, opponent_generator=None):
        super().__init__()

        self.num_players = num_players
        self.opponent_generator = opponent_generator
        self.action_space = Discrete(N_ACTIONS)

        # Observation Space Refactoring:
        # Large integer values (money) are now continuous normalized floats (Box).
        # This prevents huge embedding layers and improves stability.
        
        self.observation_space = Dict({
            "game_phase": Discrete(len(GamePhase)),
            "current_player": Discrete(N_PLAYERS),

            # per-player animals: 0..4 (unchanged, small discrete is fine, or could be Box)
            # keeping MultiDiscrete as it is categorical/count data.
            "animals": MultiDiscrete(
                np.full(N_PLAYERS * N_ANIMALS, 5, dtype=np.int64)
            ),

            # Money: Flattened arrays. 
            # OWN money: specific card counts.
            "money_own": MultiDiscrete(
                np.full(len(MONEY_VALUES), max_cards_per_value, dtype=np.int64)
            ),
            
            # OPPONENT money: Total count (small integer 0..30?). 
            # Keeping MultiDiscrete is fine.
            "money_opponents": MultiDiscrete(
                np.full((N_PLAYERS - 1), MoneyDeck.AMOUNT_MONEYCARDS, dtype=np.int64)
            ),

            # deck + donkeys
            "deck_size": Discrete(41),
            "donkeys_revealed": Discrete(5),

            # Auction Info:
            "auction_animal_type": Discrete(N_ANIMALS + 1),
            # High Bid: Money value -> Box [0, 1]
            "auction_high_bid": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
            # Cow Trade Info:
            "trade_initiator": Discrete(N_PLAYERS + 1),
            "trade_target": Discrete(N_PLAYERS + 1),
            "trade_animal_type": Discrete(N_ANIMALS + 1),
            # Card counts are visible, exact values are hidden
            "trade_offer_card_count": Discrete(MoneyDeck.AMOUNT_MONEYCARDS + 1),  # 0 = no offer
        })

        self.game: Optional[Game] = None
        self.controller: Optional[GameController] = None
        self.agents: List[Agent] = []
        self.rl_agent_id = 0  # RL agent is always player 0

        self.episode_step = 0
        self.max_steps = 500

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.game = Game(num_players=self.num_players, seed=seed)
        self.game.setup()

        self.agents = []
        # RL agent at position 0
        self.agents.append(RLAgent("RL_Learner", self))
        
        # Dynamic opponents
        if self.opponent_generator:
            others = self.opponent_generator(self.num_players - 1, self)
            self.agents.extend(others)
        else:
            # Fallback: Random agents for opponents
            for i in range(1, self.num_players):
                self.agents.append(RandomAgent(f"Random_{i}"))

        self.controller = GameController(self.game, self.agents)

        self.episode_step = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: int):
        """
        Execute one step in the environment.

        Args:
            action: int - Discrete action index chosen by the RL agent.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.episode_step += 1

        # Store the action in the RL agent so it can be used when controller asks
        rl_agent = self.agents[self.rl_agent_id]
        rl_agent.last_action_int = action

        # execute the RL agent's action
        self.controller.step()

        # Continue playing until the RL agent needs to make another decision
        self._play_until_next_decision()

        terminated = self.game.is_game_over()
        truncated = self.episode_step >= self.max_steps

        reward = self._compute_reward(terminated)

        obs = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info

    def _play_until_next_decision(self):
        """Advance the game until the RL agent needs to make a decision."""
        max_steps = 100  # Safety limit to prevent infinite loops

        for _ in range(max_steps):
            if self.game.is_game_over():
                break

            # Check if the RL agent needs to make the next decision
            decision_player = self.game.get_current_decision_player()
            if decision_player == self.rl_agent_id:
                break

            # Let the controller handle one step for the other player
            self.controller.step()


    def _compute_reward(self, terminated: bool) -> float:
        if not terminated:
            return 0.0

        #end scores since the game is terminated
        scores = self.game.get_scores()
        if scores[self.rl_agent_id] == max(scores.values()):
            return 1.0
        return 0.0


    def get_observation_for_player(self, player_id: int) -> dict:
        """
        Generate observation from the perspective of player_id.
        ROTATION: The observation is rotated so that player_id is seen as 'index 0'.
        """
        all_animal_types = AnimalType.get_all_types()

        def rotate_idx(idx):
            return (idx - player_id) % self.num_players

        # Animals
        animals = np.zeros(N_PLAYERS * N_ANIMALS, dtype=np.int64)
        for p_idx, player in enumerate(self.game.players):
            rel_idx = rotate_idx(p_idx)
            counts = player.get_animal_counts()
            for animal_idx, animal_type in enumerate(all_animal_types):
                flat_idx = rel_idx * N_ANIMALS + animal_idx
                animals[flat_idx] = counts.get(animal_type, 0)

        # Money Own
        money = np.zeros(len(MONEY_VALUES), dtype=np.int64)
        observer_player = self.game.players[player_id]
        histogram = observer_player.get_money_histogram(MONEY_VALUES)
        for value_idx, count in enumerate(histogram.values()):
             money[value_idx] = count
        
        # Money Opponents
        money_opponents = np.zeros(N_PLAYERS-1, dtype=np.int64)
        for offset in range(1, self.num_players):
            target_abs_idx = (player_id + offset) % self.num_players
            target_player = self.game.players[target_abs_idx]
            money_opponents[offset - 1] = len(target_player.money)

        # Rotated IDs
        curr_player_rel = rotate_idx(self.game.current_player_idx)
        
        auction_animal_type = AnimalType.get_all_types().index(self.game.current_animal.animal_type) if self.game.current_animal else N_ANIMALS
        trade_animal_type = AnimalType.get_all_types().index(self.game.trade_animal_type) if self.game.trade_animal_type else N_ANIMALS
        
        # Determine card counts based on who made the offer
        trade_offer_card_count = 0
        
        if self.game.trade_initiator is not None:
            # Card counts are always visible
            trade_offer_card_count = self.game.trade_offer_card_count
            
        observation = {
            "game_phase": self.game.phase.value,
            "current_player": curr_player_rel,
            "animals": animals,
            "money_own": money,
            "money_opponents": money_opponents,
            "deck_size": len(self.game.animal_deck),
            "donkeys_revealed": self.game.donkeys_revealed,
            "auction_animal_type": auction_animal_type,
            "auction_high_bid": self.game.auction_high_bid or 0,
            "trade_initiator": self.game.trade_initiator if self.game.trade_initiator is not None else N_PLAYERS,
            "trade_target": self.game.trade_target if self.game.trade_target is not None else N_PLAYERS,
            "trade_animal_type": trade_animal_type,
            "trade_offer_card_count": trade_offer_card_count,
        }

        return observation

    def _get_observation(self) -> dict:
        return self.get_observation_for_player(self.rl_agent_id)

    def get_action_mask_for_player(self, player_id: int) -> np.ndarray:
        """
        Return a binary mask of valid actions for specific player.
        """
        from gameengine.actions import ActionType

        mask = np.zeros(N_ACTIONS, dtype=np.int8)

        # Get valid actions from the game for the player
        valid_actions = self.game.get_valid_actions(player_id)

        # Get player's total money for constraining bids/offers
        player_money = self.game.players[player_id].get_total_money()
        max_bid_level = player_money // MONEY_STEP

        # Get current highest bid for auction constraints
        current_high_bid = self.game.auction_high_bid or 0
        min_bid_level = (current_high_bid // MONEY_STEP) + 1  # Must bid higher than current

        for action in valid_actions:
            if action.type == ActionType.START_AUCTION:
                mask[ACTION_START_AUCTION] = 1

            elif action.type == ActionType.COW_TRADE_CHOOSE_OPPONENT:
                target_id = action.target_id
                action_idx = ACTION_COW_CHOOSE_OPP_BASE + target_id
                if ACTION_COW_CHOOSE_OPP_BASE <= action_idx <= ACTION_COW_CHOOSE_OPP_END:
                    mask[action_idx] = 1

            elif action.type == ActionType.AUCTION_PASS:
                mask[ACTION_AUCTION_BID_BASE] = 1

            elif action.type == ActionType.AUCTION_BID:
                for bid_level in range(min_bid_level, max_bid_level + 1):
                    action_idx = ACTION_AUCTION_BID_BASE + bid_level
                    if ACTION_AUCTION_BID_BASE <= action_idx <= AUCTION_BID_END:
                        mask[action_idx] = 1

            elif action.type == ActionType.PASS_AS_AUCTIONEER:
                mask[ACTION_AUCTIONEER_ACCEPT] = 1

            elif action.type == ActionType.BUY_AS_AUCTIONEER:
                mask[ACTION_AUCTIONEER_BUY] = 1

            elif action.type == ActionType.COW_TRADE_CHOOSE_ANIMAL:
                animal_idx = AnimalType.get_all_types().index(action.animal_type)
                action_idx = ACTION_COW_CHOOSE_ANIMAL_BASE + animal_idx
                if ACTION_COW_CHOOSE_ANIMAL_BASE <= action_idx <= ACTION_COW_CHOOSE_ANIMAL_END:
                    mask[action_idx] = 1

            elif action.type == ActionType.COW_TRADE_OFFER:
                for offer_level in range(0, max_bid_level + 1):
                    action_idx = ACTION_COW_OFFER_BASE + offer_level
                    if ACTION_COW_OFFER_BASE <= action_idx <= ACTION_COW_OFFER_END:
                        mask[action_idx] = 1

            elif action.type == ActionType.COUNTER_OFFER:
                for counter_level in range(0, max_bid_level + 1):
                    action_idx = ACTION_COW_RESP_COUNTER_BASE + counter_level
                    if ACTION_COW_RESP_COUNTER_BASE <= action_idx <= COW_RESP_COUNTER_END:
                        mask[action_idx] = 1
        
        # SAFETY CHECK to prevent NaNs/Simplex Errors
        if np.sum(mask) == 0:
            if self.game.phase == GamePhase.AUCTION_BIDDING:
                mask[ACTION_AUCTION_BID_BASE] = 1 # Pass
            else:
                mask[0] = 1 # Default
                
        return mask

    def get_action_mask(self) -> np.ndarray:
        return self.get_action_mask_for_player(self.rl_agent_id)


# ----- GAME CONSTANTS -----
N_PLAYERS = 3
N_ANIMALS = len(AnimalType.get_all_types())  # 10
MONEY_VALUES = sorted(MoneyDeck.INITIAL_DISTRIBUTION.keys())

max_cards_per_value = max(MoneyDeck.INITIAL_DISTRIBUTION.values()) + 1

# offers/bids as multiples of 10 up to MAX_MONEY
MONEY_STEP = 10
MAX_MONEY = 940*5  # might adjust
N_MONEY_LEVELS = MAX_MONEY // MONEY_STEP + 1  # 0..MAX_MONEY

# ----- ACTION INDICES -----
# Turn-level actions
ACTION_START_AUCTION = 0

# Cow trade: choose opponent
ACTION_COW_CHOOSE_OPP_BASE = ACTION_START_AUCTION + 1
ACTION_COW_CHOOSE_OPP_END = ACTION_COW_CHOOSE_OPP_BASE + N_PLAYERS - 1

# Auction bidding (non-auctioneer):
# action k: bid (k * MONEY_STEP), k = 0..N_MONEY_LEVELS-1
# k=0 means "pass"
ACTION_AUCTION_BID_BASE = ACTION_COW_CHOOSE_OPP_END + 1
AUCTION_BID_END = ACTION_AUCTION_BID_BASE + N_MONEY_LEVELS - 1

# Auctioneer decision
ACTION_AUCTIONEER_ACCEPT = AUCTION_BID_END + 1
ACTION_AUCTIONEER_BUY = ACTION_AUCTIONEER_ACCEPT + 1

# Cow trade: choose animal type (0..N_ANIMALS-1)
ACTION_COW_CHOOSE_ANIMAL_BASE = ACTION_AUCTIONEER_BUY + 1
ACTION_COW_CHOOSE_ANIMAL_END = ACTION_COW_CHOOSE_ANIMAL_BASE + N_ANIMALS - 1

# Cow trade: A's offer amount
# action k : offer k * MONEY_STEP, k=0..N_MONEY_LEVELS-1
ACTION_COW_OFFER_BASE = ACTION_COW_CHOOSE_ANIMAL_END + 1
ACTION_COW_OFFER_END = ACTION_COW_OFFER_BASE + N_MONEY_LEVELS - 1

# Cow trade: B's response (counter-offer k * MONEY_STEP, k=0 means counter with 0)
ACTION_COW_RESP_COUNTER_BASE = ACTION_COW_OFFER_END + 1
COW_RESP_COUNTER_END = ACTION_COW_RESP_COUNTER_BASE + N_MONEY_LEVELS - 1

N_ACTIONS = COW_RESP_COUNTER_END + 1  # +1 because indices are 0-based
