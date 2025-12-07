from typing import Optional, List

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiDiscrete

from gameengine import AnimalType, MoneyDeck
from gameengine.Money import calculate_total_value
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
        self.observation_space = Dict({
            "game_phase": Discrete(len(GamePhase)),
            "current_player": Discrete(N_PLAYERS),

            # per-player animals: 0..4 of each type TODO maybe flatten
            "animals": MultiDiscrete(
                np.full((N_PLAYERS, N_ANIMALS), 5, dtype=np.int64)
            ),

            # per-player money histogram TODO maybe flatten
            # also for now the agent now the exact values the opponent have
            "money": MultiDiscrete(
                np.full((N_PLAYERS, len(MONEY_VALUES)), max_cards_per_value, dtype=np.int64)
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
            "trade_counter_offer_value": Discrete(MAX_MONEY + 1), #this is not kniown inforamtion but will be used for first model - might remvoe later
        })

        self.game: Optional[Game] = None
        self.controller: Optional[GameController] = None
        self.agents: List[Agent] = []
        self.rl_agent_id = 0  # RL agent is always player 0

        self.episode_step = 0
        self.max_steps = 500

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.game = Game(num_players=self.num_players, seed=seed).setup()

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

        self._apply_action(action)

        self._play_until_next_decision()

        terminated = self.game.is_game_over()
        truncated = self.episode_step >= self.max_steps

        reward = self._compute_reward(terminated)

        obs = self._get_observation()
        info = {}

        return obs, reward, terminated, truncated, info

    #TODO this first needs a refactor of the game phases (see TODO in game.py ), this will act as similar to the controller
    # here we will decode the
    def _apply_action(self, action_index: int):
        """Map a discrete action index to a concrete game operation"""
        if self.game.phase == GamePhase.PLAYER_TURN:
            if action_index == ACTION_START_AUCTION:
                pass
            if action_index == ACTION_START_COW_TRADE:
                pass
        elif self.game.phase == GamePhase.AUCTION:
            if action_index < ACTION_AUCTION_BID_BASE:
                pass
        elif self.game.phase == GamePhase.COW_TRADE:
            pass
        else:
            raise ValueError(f"Invalid game phase: {self.game.phase}")

    def _play_until_next_decision(self):
        steps_taken = 0
        max_steps_per_action = 20  # Safety limit

        while steps_taken < max_steps_per_action:
            if self.game.is_game_over():
                break

            self.controller.step()
            steps_taken += 1

            if self.game.current_player_idx == self.rl_agent_id or self.game.is_game_over():
                break


    def _compute_reward(self, terminated: bool) -> float:
        if not terminated:
            return 0.0

        #end scores since the game is terminated
        scores = self.game.get_scores()
        if scores[self.rl_agent_id] == max(scores.values()):
            return 1.0
        return 0.0


    def _get_observation(self) -> dict:

        animals = np.zeros((N_PLAYERS, N_ANIMALS), dtype=np.int64)
        for player_idx, player in enumerate(self.game.players):
            counts = player.get_animal_counts()
            animals[player_idx, :] = counts.values()

        money = np.zeros((N_PLAYERS, len(MONEY_VALUES)), dtype=np.int64)
        for player_idx, player in enumerate(self.game.players):
            histogram = player.get_money_histogram(MONEY_VALUES)
            money[player_idx, :] = list(histogram.values())

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
            "trade_offer_value": calculate_total_value(self.game.trade_offer),
            "trade_counter_offer_value": calculate_total_value(self.game.trade_counter_offer),
        }

        return observation


# ----- GAME CONSTANTS -----
N_PLAYERS = 3
N_ANIMALS = len(AnimalType.get_all_types())  # 10
MONEY_VALUES = sorted(MoneyDeck.INITIAL_DISTRIBUTION.keys())

max_cards_per_value = max(MoneyDeck.INITIAL_DISTRIBUTION.values()) + 1

# offers/bids as multiples of 10 up to MAX_MONEY
MONEY_STEP = 10
MAX_MONEY = 1000  # might adjust
N_MONEY_LEVELS = MAX_MONEY // MONEY_STEP + 1  # 0..MAX_MONEY

# ----- ACTION INDICES -----
# Turn-level actions
ACTION_START_AUCTION = 0
ACTION_START_COW_TRADE = 1
# Auction bidding (non-auctioneer):
# action 3 + k: bid (k * MONEY_STEP), k = 0..N_MONEY_LEVELS-1
# k=0 means "pass"
ACTION_AUCTION_BID_BASE = 3
AUCTION_BID_END = ACTION_AUCTION_BID_BASE + N_MONEY_LEVELS
# Auctioneer decision
ACTION_AUCTIONEER_ACCEPT = AUCTION_BID_END
ACTION_AUCTIONEER_BUY = ACTION_AUCTIONEER_ACCEPT + 1
# Cow trade: choose opponent (2 opponents in 3-player game)
ACTION_COW_CHOOSE_OPP_BASE = ACTION_AUCTIONEER_BUY + 1
ACTION_COW_CHOOSE_OPP_END = ACTION_COW_CHOOSE_OPP_BASE + N_PLAYERS - 1
# Cow trade: choose animal type (0..N_ANIMALS-1)
ACTION_COW_CHOOSE_ANIMAL_BASE = ACTION_COW_CHOOSE_OPP_END
ACTION_COW_CHOOSE_ANIMAL_END = ACTION_COW_CHOOSE_ANIMAL_BASE + N_ANIMALS
# Cow trade: A's offer amount
# action 80 + k : offer k * MONEY_STEP, k=0..N_MONEY_LEVELS-1
ACTION_COW_OFFER_BASE = ACTION_COW_CHOOSE_ANIMAL_END
ACTION_COW_OFFER_END = ACTION_COW_OFFER_BASE + N_MONEY_LEVELS

# Cow trade: B's response
ACTION_COW_RESP_ACCEPT = ACTION_COW_OFFER_END
# action: counter-offer k * MONEY_STEP
ACTION_COW_RESP_COUNTER_BASE = ACTION_COW_RESP_ACCEPT + 1
COW_RESP_COUNTER_END = ACTION_COW_RESP_COUNTER_BASE + N_MONEY_LEVELS

N_ACTIONS = COW_RESP_COUNTER_END