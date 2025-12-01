"""
Training script for Kuhhandel RL agent using Stable-Baselines3.

This trains a PPO agent with action masking to play Ku hhandel.
"""
import argparse
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from aislop.env import KuhhandelEnv


def mask_fn(env: gym.Env) -> any:
    """Extract action mask from the environment."""
    # The action mask is already in the observation dict
    return env.unwrapped._get_action_mask()


def train(total_timesteps: int = 100_000, save_freq: int = 10_000):
    """
    Train a PPO agent on Kuhhandel.
    
    Args:
        total_timesteps: Total number of environment steps to train for
        save_freq: How often to save checkpoints
    """
    print("Creating environment...")
    env = KuhhandelEnv(num_players=3, render_mode=None)
    env = Monitor(env)  # Wrapper for logging
    env = ActionMasker(env, mask_fn)  # Wrapper for action masking
    
    print("Creating PPO model...")
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy for exploration
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("\nModel architecture:")
    print(model.policy)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="./checkpoints/",
        name_prefix="kuhhandel_ppo"
    )
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        progress_bar=True
    )
    
    print("\nTraining complete!")
    print("Saving final model...")
    model.save("kuhhandel_bot")
    print("✅ Model saved as 'kuhhandel_bot.zip'")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Kuhhandel RL agent")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Checkpoint frequency")
    args = parser.parse_args()
    
    train(total_timesteps=args.timesteps, save_freq=args.save_freq)


if __name__ == "__main__":
    main()
