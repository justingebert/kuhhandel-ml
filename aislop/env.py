"""
Gymnasium environment wrapper for Kuhhandel game.

This environment allows training RL agents using Stable-Baselines3.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional

from gameengine import Game, GamePhase, ActionType, AnimalType
from gameengine.controller import GameController
from gameengine.agent import Agent
from gameengine.actions import Actions, GameAction

class RLAgent(Agent):

    """Agent wrapper for RL model - converts int actions to GameActions."""

    def __init__(self, name: str, env: 'KuhhandelEnv'):
        super().__init__(name)
        self.env = env
        self.last_action_int: Optional[int] = None

    def get_action(self, game: Game, valid_actions: List[ActionType]) -> GameAction:
        """This will be called by the controller - we return the decoded action."""
        if self.last_action_int is None:
            # Shouldn't happen, but safety fallback
            return Actions.pass_action()

        return self.env._decode_action(self.last_action_int, game)


class KuhhandelEnv(gym.Env):
    """
    Gymnasium environment for Kuhhandel.
    
    The environment manages a game with one RL agent and N-1 random opponents.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, num_players: int = 3, render_mode: Optional[str] = None):
        super().__init__()
        
        if num_players < 3 or num_players > 5:
            raise ValueError("num_players must be between 3 and 5")
        
        self.num_players = num_players
        self.render_mode = render_mode
        
        # RL agent is always player 0
        self.rl_agent_id = 0
        
        # Define action space (discrete actions that will be decoded)
        # We'll use a simple encoding for now
        self.action_space = spaces.Discrete(100)
        
        # Define observation space (we'll build this step by step)
        # For now, a simple flat vector
        obs_size = self._calculate_obs_size()
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32),
            'action_mask': spaces.MultiBinary(100)
        })
        
        # Game state
        self.game: Optional[Game] = None
        self.controller: Optional[GameController] = None
        self.agents: List[Agent] = []
        self.episode_step = 0
        self.max_steps = 500
        
    def _calculate_obs_size(self) -> int:
        """Calculate the size of the observation vector."""
        # Player state: money (7 bins) + animals (10 types x 5 states) = 57
        # Public info: phase (5), deck size (1), auction bid (1), current player (num_players) = 7 + num_players
        # Opponent info: (7 + 50) * (num_players - 1)
        player_state_size = 7 + 50  # 57
        public_info_size = 7 + self.num_players
        opponent_info_size = 57 * (self.num_players - 1)
        
        return player_state_size + public_info_size + opponent_info_size
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Create new game
        self.game = Game(num_players=self.num_players, seed=seed)
        self.game.setup()
        
        # Create agents
        # Import here to avoid circular dependency
        from tests.demo_game import RandomAgent
        
        self.agents = []
        # RL agent at position 0
        self.agents.append(RLAgent("RL_Agent", self))
        # Random agents for opponents
        for i in range(1, self.num_players):
            self.agents.append(RandomAgent(f"Random_{i}"))
        
        # Create controller
        self.controller = GameController(self.game, self.agents)
        
        self.episode_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action from the RL agent
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.episode_step += 1
        
        # Store the action for the RL agent to use
        self.agents[self.rl_agent_id].last_action_int = action
        
        # Execute game steps until it's the RL agent's turn again or game ends
        initial_turn = self.game.current_player_idx
        steps_taken = 0
        max_steps_per_action = 20  # Safety limit
        
        while steps_taken < max_steps_per_action:
            if self.game.is_game_over():
                break
            
            # Execute one controller step
            self.controller.step()
            steps_taken += 1
            
            # If it's back to RL agent's turn (or game over), stop
            if self.game.current_player_idx == self.rl_agent_id or self.game.is_game_over():
                break
        
        # Get observation and reward
        obs = self._get_observation()
        reward = self._calculate_reward()
        terminated = self.game.is_game_over()
        truncated = self.episode_step >= self.max_steps
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation for the RL agent."""
        obs_vector = self._encode_observation()
        action_mask = self._get_action_mask()
        
        return {
            'observation': obs_vector,
            'action_mask': action_mask
        }
    
    def _encode_observation(self) -> np.ndarray:
        """Encode the game state as a feature vector."""
        features = []
        
        player = self.game.players[self.rl_agent_id]
        
        # 1. Player's money (binned)
        money = player.get_total_money()
        money_bins = [0, 50, 100, 200, 500, 1000, 2000]
        money_features = [1.0 if money >= threshold else 0.0 for threshold in money_bins]
        features.extend(money_features)
        
        # 2. Player's animals (count per type, normalized)
        animal_counts = player.get_animal_counts()
        for animal_type in AnimalType.get_all_types():
            count = animal_counts.get(animal_type, 0)
            # One-hot for 0, 1, 2, 3, 4
            for i in range(5):
                features.append(1.0 if count == i else 0.0)
        
        # 3. Public game info
        phase_features = [0.0] * 5
        phase_map = {
            GamePhase.PLAYER_TURN: 0,
            GamePhase.AUCTION: 1,
            GamePhase.COW_TRADE: 2,
            GamePhase.GAME_OVER: 3,
            GamePhase.SETUP: 4
        }
        if self.game.phase in phase_map:
            phase_features[phase_map[self.game.phase]] = 1.0
        features.extend(phase_features)
        
        features.append(len(self.game.animal_deck) / 40.0)  # Deck size normalized
        features.append(self.game.auction_high_bid / 1000.0 if self.game.auction_high_bid else 0.0)
        
        # Current player (one-hot)
        for i in range(self.num_players):
            features.append(1.0 if self.game.current_player_idx == i else 0.0)
        
        # 4. Opponent info (just card counts for now)
        for i in range(self.num_players):
            if i == self.rl_agent_id:
                continue
            
            opponent = self.game.players[i]
            
            # Opponent money (binned)
            opp_money = opponent.get_total_money()
            opp_money_features = [1.0 if opp_money >= threshold else 0.0 for threshold in money_bins]
            features.extend(opp_money_features)
            
            # Opponent animals
            opp_animal_counts = opponent.get_animal_counts()
            for animal_type in AnimalType.get_all_types():
                count = opp_animal_counts.get(animal_type, 0)
                for i in range(5):
                    features.append(1.0 if count == i else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _get_action_mask(self) -> np.ndarray:
        """Get a mask of valid actions (1 = valid, 0 = invalid)."""
        mask = np.zeros(100, dtype=np.int8)
        
        valid_actions = self.game.get_valid_actions(self.rl_agent_id)
        
        # Map ActionTypes to action indices
        # Action encoding:
        # 0-9: Bid ($10, $20, ..., $100)
        # 10: Pass
        # 11-30: Start trade (not implemented yet, mark as invalid)
        # 31-50: Counter offer (not implemented yet)
        # 51: Accept offer (not implemented yet)
        
        if ActionType.BID in valid_actions:
            player = self.game.players[self.rl_agent_id]
            money = player.get_total_money()
            min_bid = self.game.auction_high_bid + 10
            
            # Enable bids that the player can afford
            for i in range(10):
                bid_amount = (i + 1) * 10  # $10, $20, ..., $100
                if bid_amount >= min_bid and bid_amount <= money:
                    mask[i] = 1
        
        if ActionType.PASS in valid_actions:
            mask[10] = 1
        
        if ActionType.BUY_AS_AUCTIONEER in valid_actions:
            # Map to pass action for now (auctioneer can choose to buy or pass)
            mask[10] = 1
        
        if ActionType.START_AUCTION in valid_actions:
            # Start auction = pass (for now, simplified)
            mask[10] = 1
        
        # TODO: Implement trade actions (11-51)
        
        # Safety: if no actions are valid, enable pass
        if mask.sum() == 0:
            mask[10] = 1
        
        return mask
    
    def _decode_action(self, action_int: int, game: Game) -> GameAction:
        """Decode an integer action to a GameAction."""
        valid_actions = game.get_valid_actions(self.rl_agent_id)
        
        # Decode based on action int
        if 0 <= action_int <= 9:
            # Bid
            if ActionType.BID in valid_actions:
                bid_amount = (action_int + 1) * 10
                return Actions.bid(amount=bid_amount)
        
        if action_int == 10:
            # Pass or contextual action
            if ActionType.PASS in valid_actions:
                return Actions.pass_action()
            if ActionType.BUY_AS_AUCTIONEER in valid_actions:
                # For now, auctioneer always passes (doesn't buy)
                return Actions.pass_action()
            if ActionType.START_AUCTION in valid_actions:
                return Actions.start_auction()
        
        # TODO: Decode trade actions (11-51)
        
        # Fallback
        return Actions.pass_action()
    
    def _calculate_reward(self) -> float:
        """Calculate the reward for the current state."""
        if not self.game.is_game_over():
            # No reward during the game for now (sparse reward)
            return 0.0
        
        # Game is over - return final reward based on rank
        scores = self.game.get_scores()
        rl_score = scores[self.rl_agent_id]
        
        # Normalize by max possible score (rough estimate)
        # Max is ~10000 if you get 3 high-value complete sets
        normalized_score = rl_score / 10000.0
        
        # Check if we won
        max_score = max(scores.values())
        if rl_score == max_score:
            # Winner bonus
            return 100.0 + normalized_score
        else:
            # Return normalized score
            return normalized_score
    
    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary information."""
        info = {
            'episode_step': self.episode_step,
            'game_phase': self.game.phase.value if self.game else 'none',
        }
        
        if self.game and self.game.is_game_over():
            scores = self.game.get_scores()
            info['final_scores'] = scores
            info['rl_agent_score'] = scores[self.rl_agent_id]
            info['winner'] = max(scores, key=scores.get)
        
        return info
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"\nStep: {self.episode_step}")
            print(f"Phase: {self.game.phase.value}")
            print(f"Current Player: {self.game.current_player_idx}")
            
            for i, player in enumerate(self.game.players):
                prefix = "RL" if i == self.rl_agent_id else "AI"
                print(f"{prefix} Player {i}: Money={player.get_total_money()}, "
                      f"Animals={len(player.animals)}, Score={player.calculate_score()}")
    
    def close(self):
        """Clean up resources."""
        pass
