
import logging
import sys
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
import numpy as np
import random
from rl.env import KuhhandelEnv
from rl.train_selfplay import LATEST_MODEL_PATH
from gameengine.actions import ActionType

DEBUG_SEED = random.randint(0, 2**32 - 1)

def run_game():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    logger = logging.getLogger()
    
    print(f"Loading Model from {LATEST_MODEL_PATH}...")
    try:
        model = MaskablePPO.load(LATEST_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using Random Agent fallback (mask-aware) for demo.")
    model = None 

    env = KuhhandelEnv(num_players=3)
    obs, _ = env.reset(seed=DEBUG_SEED)
    
    done = False
    turn_count = 0
    
    print("\n--- STARTING FULL GAME DEBUG LOG ---")
    
    while not done:
        turn_count += 1
        current_player = env.game.current_player_idx
        phase = env.game.phase
        
        # Determine valid actions mask
        action_mask = env.get_action_mask()
        
        # Predict action
        if model:
            # We must use predict with mask
            action, _state = model.predict(obs, action_masks=action_mask, deterministic=True)
        else:
            # Random fallback
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                print(f"FATAL: No valid actions for Player {current_player} in Phase {phase}!")
                print("--- DEBUG STATE DUMP ---")
                print(f"Deck Empty: {len(env.game.animal_deck) == 0}")
                print(f"Game Over: {env.game.is_game_over()}")
                for p in env.game.players:
                   print(f"P{p.player_id}: {p.get_animal_counts()}")
                break
                
        # Decode action to human readable
        # Accessing private decoding method or we can just look at game logs
        # Better: Execute step and check game.action_history
        
        # Capture history length before step to see what's new
        prev_history_len = len(env.game.action_history)
        
        # Execute
        obs, rewards, truncated, terminated, info = env.step(action)
        done = truncated or terminated
        
        # LOGGING
        # Iterate over all NEW actions that happened during this step
        current_history_len = len(env.game.action_history)
        if current_history_len > prev_history_len:
            for i in range(prev_history_len, current_history_len):
                entry = env.game.action_history[i]
                print(f"[AI Step {turn_count} | Round {entry['round']}] Action: {entry['action']} | Details: {entry['details']}")
        else:
            # Fallback if no actions logic (shouldn't happen usually)
            pass
        
        if turn_count > 1000:
            print("Force stopping after 1000 turns (Infinite Loop safeguard)")
            break

    print("\n--- GAME OVER ---")
    print(f"Winner: {env.game.get_winner().player_id} (Score: {env.game.get_winner().calculate_score()})")
    print("Scores:")
    for p in env.game.players:
        print(f"P{p.player_id}: {p.calculate_score()}")

if __name__ == "__main__":
    run_game()
