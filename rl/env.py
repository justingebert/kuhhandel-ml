from typing import Optional, List

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiDiscrete

from gameengine import AnimalType, MoneyDeck
from gameengine.actions import Actions, GameAction
from gameengine.agent import Agent
from gameengine.controller import GameController
from gameengine.game import Game, GamePhase
from rl.random_agent import RandomAgent
from rl.rl_agent import RLAgent


class KuhhandelEnv(gym.Env):

    def __init__(self, num_players: int = 3, opponent_generator=None):
        super().__init__()

        self.num_players = num_players
        self.opponent_generator = opponent_generator
        self.action_space = Discrete(N_ACTIONS)

        self.observation_space = Dict({
            "game_phase": Discrete(len(GamePhase)),
            "current_player": Discrete(N_PLAYERS),

            # per-player animals: 0..4 of each type (flattened: N_PLAYERS * N_ANIMALS)
            "animals": MultiDiscrete(
                np.full(N_PLAYERS * N_ANIMALS, 5, dtype=np.int64)
            ),

            # own money histogram
            "money_own": MultiDiscrete(
                np.full(len(MONEY_VALUES), max_cards_per_value, dtype=np.int64)
            ),

            "money_opponents": MultiDiscrete(
                np.full((N_PLAYERS - 1), MoneyDeck.AMOUNT_MONEYCARDS, dtype=np.int64)
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
            # Card counts are visible, exact values are hidden
            "trade_offer_card_count": Discrete(MoneyDeck.AMOUNT_MONEYCARDS + 1),  # 0 = no offer
        })

        self.game: Optional[Game] = None
        self.controller: Optional[GameController] = None
        self.agents: List[Agent] = []
        self.rl_agent_id = 0  # RL agent is always player 0

        self.episode_step = 0
        self.max_steps = 500
        self.last_quartet_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.game = Game(num_players=self.num_players, seed=seed)
        self.game.setup()

        self.agents = []
        self.agents.append(RLAgent("RL_Learner", self))

        if self.opponent_generator:
            others = self.opponent_generator(self, self.num_players - 1)
            self.agents.extend(others)
        else:
            # Fallback: Random agents for opponents
            for i in range(1, self.num_players):
                self.agents.append(RandomAgent(f"Random_{i}"))

        self.controller = GameController(self.game, self.agents)

        self.episode_step = 0
        self.last_quartet_count = 0

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

        # Clear last trade result from previous step
        self.game.last_trade_result = None

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
        
        if terminated:
            scores = self.game.get_scores()
            if scores:
                max_s = max(scores.values())
                info["winners"] = [p for p, s in scores.items() if s == max_s]

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
        reward = 0.0

        # Quartet Bonus: Reward building quartets while the deck is still active
        rl_player = self.game.players[self.rl_agent_id]
        current_quartets = sum(1 for c in rl_player.get_animal_counts().values() if c == 4)
        quartet_diff = current_quartets - self.last_quartet_count

        if len(self.game.animal_deck) > 0 and quartet_diff != 0:
            reward += quartet_diff * 0.3
        
        self.last_quartet_count = current_quartets

        # Reward/penalty for cow trade outcomes based on economic efficiency
        if self.game.last_trade_result is not None:
            result = self.game.last_trade_result
            animals_transferred = result.get("animals_transferred")
            net_payment = result.get("net_payment")

            if result["winner_player_id"] == self.rl_agent_id:
                base_reward = 0.1 * animals_transferred
                # If paid 0, get full bonus; if paid 500+, get no bonus - normalized
                efficiency_bonus = 0.1 * max(0, (500 - net_payment) / 500)
                reward += base_reward + efficiency_bonus

            elif result["loser_player_id"] == self.rl_agent_id:
                base_penalty = -0.1 * animals_transferred
                # If received 500+, can offset the penalty entirely
                money_compensation = 0.15 * min(1.0, net_payment / 500)
                reward += base_penalty + money_compensation

        if not terminated:
            return reward

        # End scores since the game is terminated
        scores = self.game.get_scores()
        if scores[self.rl_agent_id] == max(scores.values()):
            return reward + 5.0

        return reward


    def get_observation_for_player(self, player_id: int) -> dict:
        """
        Generate observation from the perspective of player_id.
        ROTATION: The observation is rotated so that player_id is seen as 'index 0'.
        """
        all_animal_types = AnimalType.get_all_types()

        # Flattened animals array: [player0_animal0, player0_animal1, ..., player2_animal9]
        animals = np.zeros(N_PLAYERS * N_ANIMALS, dtype=np.int64)
        for player_idx, player in enumerate(self.game.players):
            counts = player.get_animal_counts()
            for animal_idx, animal_type in enumerate(all_animal_types):
                flat_idx = player_idx * N_ANIMALS + animal_idx
                animals[flat_idx] = counts.get(animal_type, 0)

        money = np.zeros(len(MONEY_VALUES), dtype=np.int64)
        observer_player = self.game.players[player_id]
        histogram = observer_player.get_money_histogram(MONEY_VALUES)
        for value_idx, count in enumerate(histogram.values()):
             money[value_idx] = count

        money_opponents = np.zeros(N_PLAYERS-1, dtype=np.int64)
        for player_idx, player in enumerate(self.game.players):
            if player_idx != player_id:
                cards = len(player.money)
                opponent_idx = player_idx - 1
                money_opponents[opponent_idx] = cards

        auction_animal_type = AnimalType.get_all_types().index(self.game.current_animal.animal_type) if self.game.current_animal else N_ANIMALS
        trade_animal_type = AnimalType.get_all_types().index(self.game.trade_animal_type) if self.game.trade_animal_type else N_ANIMALS

        observation = {
            "game_phase": self.game.phase.value,
            "current_player": self.game.current_player_idx,
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
            "trade_offer_card_count": self.game.trade_offer_card_count,
        }

        return observation

    def _get_observation(self) -> dict:
        return self.get_observation_for_player(self.rl_agent_id)

    def get_action_mask_for_player(self, player_id: int) -> np.ndarray:
        """
        Return a binary mask of valid actions for specific player.
        1 = action is valid, 0 = action is invalid.
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

    def get_action_mask(self) -> np.ndarray:
        return self.get_action_mask_for_player(self.rl_agent_id)

    @staticmethod
    def decode_action(action_idx: int, game: Game) -> GameAction:
        """Decode integer action index to GameAction."""

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

        raise ValueError(f"Cannot decode action {action_idx} for phase {game.phase}")


# ----- GAME CONSTANTS -----
N_PLAYERS = 3
N_ANIMALS = len(AnimalType.get_all_types())  # 10
MONEY_VALUES = sorted(MoneyDeck.INITIAL_DISTRIBUTION.keys())

max_cards_per_value = max(MoneyDeck.INITIAL_DISTRIBUTION.values()) + 1

# offers/bids as multiples of 10 up to MAX_MONEY
MONEY_STEP = 10
MAX_MONEY = 940*N_PLAYERS  # edited so that max_money depends on number of players
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
