import argparse
import os

from sb3_contrib import MaskablePPO

from rl.agents.user_agent import UserAgent
from rl.env import KuhhandelEnv
from rl.agents.model_agent import ModelAgent
from rl.agents.random_agent import RandomAgent

from sb3_contrib.common.maskable import distributions as maskable_dist
from rl.train_selfplay import robust_apply_masking
maskable_dist.MaskableCategorical.apply_masking = robust_apply_masking

def play_terminal():
    parser = argparse.ArgumentParser(description="Play Kuhhandel against AI in the terminal.")
    parser.add_argument("--players", type=int, default=3, help="Number of players (3-5)")
    parser.add_argument("--model_path", type=str, default="rl/models/kuhhandel_ppo_latest.zip", help="Path to opponent model")
    args = parser.parse_args()

    num_players = args.players
    model_path = args.model_path

    # Check if model exists
    model_instance = None
    if os.path.exists(model_path):
        print(f"Loading opponent model from {model_path}...")
        try:
            model_instance = MaskablePPO.load(model_path, device='cpu')
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to Random Agents.")
    else:
        print(f"Model not found at {model_path}. Using Random Agents.")

    def opponent_generator(env, count):
        agents = []
        for i in range(count):
            name = f"Opponent {i+1}"
            if model_instance:
                agents.append(ModelAgent(name, model_path, env, model_instance=model_instance))
            else:
                agents.append(RandomAgent(name))
        return agents

    env = KuhhandelEnv(num_players=num_players, opponent_generator=opponent_generator)
    
    # Reset to setup the game
    print("Setting up game...")
    env.reset()
    
    # Inject UserAgent as Player 0
    user_agent = UserAgent("You")
    user_agent.set_player_id(0)
    
    # Replace the RLAgent (placeholder) with UserAgent
    env.agents[0] = user_agent
    
    print("\n" + "="*50)
    print("STARTING GAME")
    print("="*50)
    
    # Run the game loop
    scores = env.controller.run()
    
    print("\n" + "="*50)
    print("GAME OVER")
    print("Final Scores:")
    for player_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        player_name = env.agents[player_id].name
        print(f"  {player_name} (Player {player_id}): {score}")
    print("="*50)

if __name__ == "__main__":
    play_terminal()
