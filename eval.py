"""
Evaluation script for trained Kuhhandel agent.

Loads a trained model and evaluates its performance.
"""
import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym

from aislop.env import KuhhandelEnv


def mask_fn(env: gym.Env) -> any:
    """Extract action mask from the environment."""
    return env.unwrapped._get_action_mask()


def evaluate(model_path: str = "kuhhandel_bot.zip", num_episodes: int = 100):
    """
    Evaluate a trained model's performance.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to evaluate
    """
    print(f"Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)
    
    print("Creating evaluation environment...")
    env = KuhhandelEnv(num_players=3, render_mode=None)
    env = ActionMasker(env, mask_fn)
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 60)
    
    episode_rewards = []
    episode_lengths = []
    wins = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if 'winner' in info and info['winner'] == 0:
            wins += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Avg Reward: {np.mean(episode_rewards[-10:]):.2f}, "
                  f"Win Rate: {wins}/{episode + 1}")
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"Win Rate: {wins}/{num_episodes} ({100 * wins / num_episodes:.1f}%)")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'win_rate': wins / num_episodes,
        'mean_length': np.mean(episode_lengths)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Kuhhandel agent")
    parser.add_argument("--model", type=str, default="kuhhandel_bot.zip", help="Path to model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()
    
    evaluate(model_path=args.model, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
