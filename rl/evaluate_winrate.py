import os
import argparse
import glob
import numpy as np
from sb3_contrib import MaskablePPO

from rl.env import KuhhandelEnv
from rl.model_agent import ModelAgent
from tests.demo_game import RandomAgent

def create_specific_opponents(n_opponents, env_ref, opponent_type, opponent_model_path=None):
    """
    Generator that creates specific opponents for evaluation.
    """
    opponents = []
    for i in range(n_opponents):
        if opponent_type == "random":
            opponents.append(RandomAgent(f"Random_{i}"))
        elif opponent_type == "model":
            if not opponent_model_path:
                raise ValueError("Must provide model path for model opponents")
            try:
                # Load or reuse model? For eval it's safer to load once outside and pass instance?
                # For simplicity here, we load inside or use a simple cache if needed.
                # Since we reset ~100 times, caching is good.
                # But here we just assume the class or user handles it. 
                # Let's load fresh for safety or implement simple caching here.
                opponents.append(ModelAgent(f"Opponent_{i}", opponent_model_path, env=env_ref))
            except Exception as e:
                print(f"Error loading opponent model: {e}")
                opponents.append(RandomAgent(f"Fallback_{i}"))
    return opponents

def evaluate(main_model_path, opponent_type="random", opponent_model_path=None, n_games=100):
    print(f"\nEvaluating Main Model: {main_model_path}")
    print(f"Against Opponents: {opponent_type} ({opponent_model_path if opponent_model_path else ''})")
    print(f"Games: {n_games}")
    
    # Setup Env
    env = KuhhandelEnv(num_players=3)
    
    # Setup Opponent Generator
    def eval_generator(n, env_ref):
        return create_specific_opponents(n, env_ref, opponent_type, opponent_model_path)
    
    env.opponent_generator = eval_generator
    
    # Load Main Model
    try:
        main_model = MaskablePPO.load(main_model_path, env=None) # We don't need env attached for predict if obs space matches
    except Exception as e:
        print(f"Failed to load main model: {e}")
        return

    wins = 0
    scores = []
    
    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get Action Mask
            action_masks = env.get_action_mask()
            
            # Predict
            action, _ = main_model.predict(obs, action_masks=action_masks, deterministic=True)
            
            # Step
            obs, reward, done, truncated, info = env.step(action)
            
        # Game Over
        # Reward is 1.0 if won, 0.0 otherwise (based on current env implementation)
        if reward > 0:
            wins += 1
            
        # Optional: track exact score?
        # The env doesn't return score in info by default yet, but we can access env.game
        p0_score = env.game.players[0].calculate_score()
        scores.append(p0_score)
        
        if (i+1) % 10 == 0:
            print(f"Game {i+1}/{n_games} finished. Current Winrate: {wins/(i+1):.2%}")

    winrate = wins / n_games
    avg_score = np.mean(scores)
    
    print("\n" + "="*30)
    print(f"RESULTS ({n_games} games)")
    print(f"Winrate: {winrate:.2%}")
    print(f"Avg Score (Main): {avg_score:.1f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, required=True, help="Path to main model zip")
    parser.add_argument("--opp", type=str, default="random", help="'random' or path to opponent model zip")
    parser.add_argument("--n", type=int, default=100, help="Number of games")
    
    args = parser.parse_args()
    
    opp_type = "model" if args.opp != "random" else "random"
    opp_path = args.opp if args.opp != "random" else None
    
    evaluate(args.main, opp_type, opp_path, args.n)
