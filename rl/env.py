from typing import Optional, List

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiDiscrete

from gameengine import AnimalType, MoneyDeck
from gameengine.agent import Agent
from gameengine.controller import GameController
from gameengine.game import Game, GamePhase
from rl.rl_agent import RLAgent
from tests.demo_game import RandomAgent


class KuhhandelEnv(gym.Env):

    def __init__(self, num_players: int = 3):
        super().__init__()

        self.num_players = num_players
        self.action_space = Discrete(N_ACTIONS)

        # broken down observation space for first runs
        # Note: MultiDiscrete needs flattened 1D arrays
        self.observation_space = Dict({
            "game_phase": Discrete(len(GamePhase)),
            "current_player": Discrete(N_PLAYERS),

            # per-player animals: 0..4 of each type (flattened: N_PLAYERS * N_ANIMALS)
            "animals": MultiDiscrete(
                np.full(N_PLAYERS * N_ANIMALS, 5, dtype=np.int64)
            ),

            # per-player money histogram (flattened: N_PLAYERS * len(MONEY_VALUES))
            # also for now the agent knows the exact values the opponents have
            "money": MultiDiscrete(
                np.full(N_PLAYERS * len(MONEY_VALUES), max_cards_per_value, dtype=np.int64)
            ),

            # deck + donkeys
            "deck_size": Discrete(41),
            "donkeys_revealed": Discrete(5),

            # auction info
            "auction_animal_type": Discrete(N_ANIMALS + 1),  # 0..N_ANIMALS-1, N_ANIMALS = none
            "auction_high_bid": Discrete(MAX_MONEY + 1),  # 0..MAX_MONEY

            # cow trade info
            "trade_initiator": Discrete(N_PLAYERS + 1),  # +1 for "none"
            "trade_target": Discrete(N_PLAYERS + 1),
            "trade_animal_type": Discrete(N_ANIMALS + 1),
            "trade_offer_value": Discrete(MAX_MONEY + 1),
            "trade_counter_offer_value": Discrete(MAX_MONEY + 1), #this is not known information but will be used for first model - might remove later
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
        self.agents.append(RLAgent("RL_Agent", self))
        # Random agents for opponents
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


    def _get_observation(self) -> dict:
        all_animal_types = AnimalType.get_all_types()

        # Flattened animals array: [player0_animal0, player0_animal1, ..., player2_animal9]
        animals = np.zeros(N_PLAYERS * N_ANIMALS, dtype=np.int64)
        for player_idx, player in enumerate(self.game.players):
            counts = player.get_animal_counts()
            for animal_idx, animal_type in enumerate(all_animal_types):
                flat_idx = player_idx * N_ANIMALS + animal_idx
                animals[flat_idx] = counts.get(animal_type, 0)

        # Flattened money array: [player0_value0, player0_value1, ..., player2_value5]
        money = np.zeros(N_PLAYERS * len(MONEY_VALUES), dtype=np.int64)
        for player_idx, player in enumerate(self.game.players):
            histogram = player.get_money_histogram(MONEY_VALUES)
            for value_idx, count in enumerate(histogram.values()):
                flat_idx = player_idx * len(MONEY_VALUES) + value_idx
                money[flat_idx] = count

        auction_animal_type = AnimalType.get_all_types().index(self.game.current_animal.animal_type) if self.game.current_animal else N_ANIMALS
        trade_animal_type = AnimalType.get_all_types().index(self.game.trade_animal_type) if self.game.trade_animal_type else N_ANIMALS

        observation = {
            "game_phase": self.game.phase.value,
            "current_player": self.game.current_player_idx,
            "animals": animals,
            "money": money,
            "deck_size": len(self.game.animal_deck),
            "donkeys_revealed": self.game.donkeys_revealed,
            "auction_animal_type": auction_animal_type,
            "auction_high_bid": self.game.auction_high_bid or 0,
            "trade_initiator": self.game.trade_initiator or N_PLAYERS,
            "trade_target": self.game.trade_target or N_PLAYERS,
            "trade_animal_type": trade_animal_type,
            "trade_offer_value": self.game.trade_offer or 0,
            "trade_counter_offer_value": self.game.trade_counter_offer or 0,
        }

        return observation

    def get_action_mask(self) -> np.ndarray:
        """
        Return a binary mask of valid actions.
        1 = action is valid, 0 = action is invalid.
        """
        from gameengine.actions import ActionType

        mask = np.zeros(N_ACTIONS, dtype=np.int8)

        # Get valid actions from the game for the RL agent
        valid_actions = self.game.get_valid_actions(self.rl_agent_id)

        # Get player's total money for constraining bids/offers
        player_money = self.game.players[self.rl_agent_id].get_total_money()
        max_bid_level = player_money // MONEY_STEP

        # Get current highest bid for auction constraints
        current_high_bid = self.game.auction_high_bid or 0
        min_bid_level = (current_high_bid // MONEY_STEP) + 1  # Must bid higher than current

        for action in valid_actions:
            if action.type == ActionType.START_AUCTION:
                mask[ACTION_START_AUCTION] = 1

            elif action.type == ActionType.COW_TRADE_CHOOSE_OPPONENT:
                # Map opponent ID to action index
                target_id = action.target_id
                action_idx = ACTION_COW_CHOOSE_OPP_BASE + target_id
                if ACTION_COW_CHOOSE_OPP_BASE <= action_idx <= ACTION_COW_CHOOSE_OPP_END:
                    mask[action_idx] = 1

            elif action.type == ActionType.AUCTION_PASS:
                # Pass is always valid during bidding
                mask[ACTION_AUCTION_BID_BASE] = 1

            elif action.type == ActionType.AUCTION_BID:
                # Only enable bids that are:
                # 1. Higher than current highest bid
                # 2. Within player's money
                for bid_level in range(min_bid_level, max_bid_level + 1):
                    action_idx = ACTION_AUCTION_BID_BASE + bid_level
                    if ACTION_AUCTION_BID_BASE <= action_idx <= AUCTION_BID_END:
                        mask[action_idx] = 1

            elif action.type == ActionType.PASS_AS_AUCTIONEER:
                mask[ACTION_AUCTIONEER_ACCEPT] = 1

            elif action.type == ActionType.BUY_AS_AUCTIONEER:
                mask[ACTION_AUCTIONEER_BUY] = 1

            elif action.type == ActionType.COW_TRADE_CHOOSE_ANIMAL:
                # Map animal type to action index
                animal_idx = AnimalType.get_all_types().index(action.animal_type)
                action_idx = ACTION_COW_CHOOSE_ANIMAL_BASE + animal_idx
                if ACTION_COW_CHOOSE_ANIMAL_BASE <= action_idx <= ACTION_COW_CHOOSE_ANIMAL_END:
                    mask[action_idx] = 1

            elif action.type == ActionType.COW_TRADE_OFFER:
                # Only enable offers within player's money (0 to max_bid_level)
                for offer_level in range(0, max_bid_level + 1):
                    action_idx = ACTION_COW_OFFER_BASE + offer_level
                    if ACTION_COW_OFFER_BASE <= action_idx <= ACTION_COW_OFFER_END:
                        mask[action_idx] = 1

            elif action.type == ActionType.COUNTER_OFFER:
                # Only enable counter offers within player's money (0 to max_bid_level)
                for counter_level in range(0, max_bid_level + 1):
                    action_idx = ACTION_COW_RESP_COUNTER_BASE + counter_level
                    if ACTION_COW_RESP_COUNTER_BASE <= action_idx <= COW_RESP_COUNTER_END:
                        mask[action_idx] = 1

        return mask


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
