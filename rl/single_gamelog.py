import logging
import random
import sys

from sb3_contrib import MaskablePPO

from rl.env import KuhhandelEnv
from rl.train_selfplay import LATEST_MODEL_PATH

DEBUG_SEED = random.randint(0, 2**32 - 1)

def run_game():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    logging.getLogger()
    
    print(f"Loading Model from {LATEST_MODEL_PATH}...")
    try:
        model = MaskablePPO.load(LATEST_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    env = KuhhandelEnv(num_players=3)
    obs, _ = env.reset(seed=DEBUG_SEED)
    
    done = False
    turn_count = 0
    
    print("\n--- STARTING FULL GAME DEBUG LOG ---")
    
    while not done:
        turn_count += 1

        action, _state = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        
        # Capture history length before step to see what's new
        prev_history_len = len(env.game.action_history)

        obs, rewards, truncated, terminated, info = env.step(action)
        done = truncated or terminated

        # Iterate over all NEW actions that happened during this step
        current_history_len = len(env.game.action_history)
        for i in range(prev_history_len, current_history_len):
            entry = env.game.action_history[i]
            print(f"[Step {turn_count} | Round {entry['round']}] Action: {entry['action']} | Details: {entry['details']}")

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
