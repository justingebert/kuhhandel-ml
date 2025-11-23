import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple

from gameengine import Game, GamePhase, ActionType, AnimalType


class GameStateEncoder:
    """Encode game state into a feature vector."""

    def __init__(self, num_players: int):
        self.num_players = num_players
        self.num_animal_types = len(AnimalType.get_all_types())

    def encode(self, game: Game, player_idx: int) -> np.ndarray:
        """
        Encode game state from perspective of player_idx.
        Returns a fixed-size feature vector.
        """
        features = []

        # Player's own state
        player = game.players[player_idx]
        features.append(player.get_total_money() / 1000.0)  # Normalize money

        # Player's animals (count per type)
        animal_counts = player.get_animal_counts()
        for animal_type in AnimalType.get_all_types():
            count = animal_counts.get(animal_type, 0)
            features.append(count / 4.0)  # Max 4 per set

        # Other players' states
        for i in range(self.num_players):
            if i != player_idx:
                other = game.players[i]
                features.append(other.get_total_money() / 1000.0)
                other_counts = other.get_animal_counts()
                for animal_type in AnimalType.get_all_types():
                    count = other_counts.get(animal_type, 0)
                    features.append(count / 4.0)

        # Game state
        features.append(len(game.animal_deck) / 28.0)  # Deck size
        features.append(game.round_number / 100.0)

        # Current phase (one-hot)
        phase_features = [0.0] * 3
        if game.phase == GamePhase.PLAYER_TURN:
            phase_features[0] = 1.0
        elif game.phase == GamePhase.AUCTION:
            phase_features[1] = 1.0
        elif game.phase == GamePhase.COW_TRADE:
            phase_features[2] = 1.0
        features.extend(phase_features)

        # Auction state (if active)
        features.append(game.auction_high_bid / 1000.0 if game.auction_high_bid else 0.0)

        return np.array(features, dtype=np.float32)

    def get_state_size(self) -> int:
        """Calculate total feature vector size."""
        # Own: money + animals
        own_features = 1 + self.num_animal_types
        # Others: (money + animals) * (num_players - 1)
        other_features = (1 + self.num_animal_types) * (self.num_players - 1)
        # Game: deck size, round, phase (3), auction bid
        game_features = 1 + 1 + 3 + 1
        return own_features + other_features + game_features


class PolicyNetwork(nn.Module):
    """Neural network for action selection."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class RLAgent:
    """Reinforcement Learning agent for the game."""

    def __init__(self, player_idx: int, state_size: int, num_actions: int = 50):
        self.player_idx = player_idx
        self.policy_net = PolicyNetwork(state_size, num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # Discount factor

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor).squeeze()

            # Mask invalid actions
            masked_q = q_values.clone()
            mask = torch.ones_like(masked_q) * float('-inf')
            mask[valid_actions] = 0
            masked_q = masked_q + mask

            return torch.argmax(masked_q).item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Q-learning update
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.policy_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def action_to_game_action(action_idx: int, game: Game, player_idx: int):
    """Map action index to actual game action."""
    # Simple mapping: 0-9 for bids, 10 for pass, 11+ for trades
    if action_idx < 10:
        # Bid actions (multiples of 10)
        bid_amount = (action_idx + 1) * 10
        return ('bid', bid_amount)
    elif action_idx == 10:
        return ('pass', None)
    else:
        return ('trade', action_idx - 11)


def train_rl_agents(num_episodes: int = 1000, num_players: int = 3):
    """Train RL agents by self-play."""

    encoder = GameStateEncoder(num_players)
    state_size = encoder.get_state_size()
    agents = [RLAgent(i, state_size) for i in range(num_players)]

    print(f"Training {num_players} agents for {num_episodes} episodes...")
    print(f"State size: {state_size}")

    for episode in range(num_episodes):
        game = Game(num_players=num_players, seed=episode)
        game.setup()

        episode_rewards = [0] * num_players
        turn_count = 0
        max_turns = 500

        while not game.is_game_over() and turn_count < max_turns:
            turn_count += 1
            current_player = game.current_player_idx
            agent = agents[current_player]

            # Get state and valid actions
            state = encoder.encode(game, current_player)
            valid_actions_game = game.get_valid_actions()

            # Map game actions to indices (simplified)
            valid_actions = list(range(20))  # Simplified action space

            # Select and execute action
            action = agent.select_action(state, valid_actions)

            # Execute action in game (simplified - would need full mapping)
            reward = 0
            try:
                if ActionType.START_AUCTION in valid_actions_game:
                    game.start_auction()
                    reward = 0.1
            except:
                reward = -0.5  # Invalid action penalty

            # Get next state
            next_state = encoder.encode(game, current_player)
            done = game.is_game_over()

            # Store experience
            agent.remember(state, action, reward, next_state, done)
            episode_rewards[current_player] += reward

            # Train agent
            agent.replay()

        # Final rewards based on game outcome
        if game.is_game_over():
            scores = game.get_scores()
            max_score = max(scores.values())
            for player_idx in range(num_players):
                if scores[player_idx] == max_score:
                    episode_rewards[player_idx] += 100  # Win bonus

        # Logging
        if episode % 100 == 0:
            avg_reward = sum(episode_rewards) / num_players
            print(f"Episode {episode}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agents[0].epsilon:.3f}")

    return agents


if __name__ == "__main__":
    trained_agents = train_rl_agents(num_episodes=1000, num_players=3)

    # Save trained models
    for i, agent in enumerate(trained_agents):
        torch.save(agent.policy_net.state_dict(), f'agent_{i}_model.pth')

    print("Training complete! Models saved.")
