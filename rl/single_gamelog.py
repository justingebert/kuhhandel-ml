import logging
import random
import sys
import functools

from sb3_contrib import MaskablePPO

from rl.env import KuhhandelEnv
from rl.agents.model_agent import ModelAgent
from rl.train_selfplay import LATEST_MODEL_PATH

from sb3_contrib.common.maskable import distributions as maskable_dist
from rl.train_selfplay import robust_apply_masking
maskable_dist.MaskableCategorical.apply_masking = robust_apply_masking

DEBUG_SEED = random.randint(0, 2**32 - 1)

def create_model_opponents(env_ref, n):
    """
    Create n ModelAgent opponents that use the same model.
    This shares the model instance to avoid loading it multiple times.
    """
    # Load model once and share it
    model = MaskablePPO.load(LATEST_MODEL_PATH, device='cpu')
    if hasattr(model, "rollout_buffer"):
        model.rollout_buffer = None
    
    opponents = []
    for i in range(n):
        opponents.append(ModelAgent(f"Model_{i+1}", model_path=None, env=env_ref, model_instance=model))
    
    return opponents

def run_game():
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    logging.getLogger()
    
    print(f"Loading Model from {LATEST_MODEL_PATH}...")
    try:
        model = MaskablePPO.load(LATEST_MODEL_PATH, device='cpu')
        if hasattr(model, "rollout_buffer"):
            model.rollout_buffer = None
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # Create environment with model opponents (all 3 players use the same model)
    env = KuhhandelEnv(num_players=3)
    env.opponent_generator = functools.partial(create_model_opponents)
    
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
