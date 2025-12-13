import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym

from rl.env import KuhhandelEnv

def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

def evaluate(model_path: str, n_games: int = 100):
    """
    Evaluate a trained model against random opponents.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = KuhhandelEnv()
    env = ActionMasker(env, mask_valid_action)
    
    wins = 0
    total_score = 0
    opponent_scores = 0
    
    print(f"Starting evaluation over {n_games} games...")
    
    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated):
            action_masks = mask_valid_action(env)
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, done, truncated, info = env.step(action)
            
        # Game over
        scores = env.unwrapped.game.get_scores()
        winner = env.unwrapped.game.get_winner()
        
        # RL agent is always player 0
        rl_score = scores[0]
        opp_score_avg = sum(s for pid, s in scores.items() if pid != 0) / (len(scores) - 1)
        
        total_score += rl_score
        opponent_scores += opp_score_avg
        
        if winner and winner.player_id == 0:
            wins += 1
            
        if (i+1) % 10 == 0:
            print(f"Game {i+1}/{n_games} | Win Rate: {wins/(i+1):.2%} | Avg Score: {total_score/(i+1):.1f}")

    print("\n" + "="*50)
    print(f"FINAL RESULTS ({n_games} games)")
    print(f"Win Rate: {wins/n_games:.2%}")
    print(f"Average Score (RL): {total_score/n_games:.1f}")
    print(f"Average Score (Opponent): {opponent_scores/n_games:.1f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/kuhhandel_ppo_final", help="Path to model file")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    args = parser.parse_args()
    
    evaluate(args.model, args.games)
