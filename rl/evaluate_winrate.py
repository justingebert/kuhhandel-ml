import argparse
import multiprocessing
from pathlib import Path
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl.env import KuhhandelEnv
from rl.model_agent import ModelAgent
from tests.demo_game import RandomAgent

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "models" / "kuhhandel_ppo_latest"

def mask_valid_action(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

def create_specific_opponents(n_opponents, env_ref, opponent_type, opponent_model_path):
    opponents = []
    
    for i in range(n_opponents):
        if opponent_type == "random":
            opponents.append(RandomAgent(f"Random_{i}"))
        elif opponent_type == "model":
            try:
                opponents.append(ModelAgent(f"Opponent_{i}", opponent_model_path, env=env_ref))
            except Exception as e:
                # If model path is None but type is model, fallback to random? 
                # No, that should be handled by caller.
                raise RuntimeError(f"Error loading opponent model: path:{opponent_model_path} Error: {e}")
    
    return opponents

def make_eval_env(rank, opponent_type, opponent_model_path):
    def _init():
        env = KuhhandelEnv(num_players=3)
        
        def opponent_generator(env_ref, n):
            return create_specific_opponents(n, env_ref, opponent_type, opponent_model_path)
            
        env.opponent_generator = opponent_generator
        env = ActionMasker(env, mask_valid_action)
        return env
    
    return _init

def run_batch(main_model_path, opp_type, opp_path, n_games, label="Batch"):
    """
    Runs a batch of games and returns stats.
    """

    # Don't use more envs than games requested
    n_envs = min(n_games, multiprocessing.cpu_count(), 16)

    print(f"\n--- {label} ---")
    print(f"Main Agent: {'Random' if main_model_path is None else main_model_path}")
    print(f"Opponents:  {opp_type} ({'Random' if opp_path is None else opp_path})")
    print(f"Using {n_envs} parallel environments")

    
    env_fns = [make_eval_env(i, opp_type, opp_path) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    
    wins = 0
    games_completed = 0
    
    try:
        if main_model_path:
            main_model = MaskablePPO.load(main_model_path, env=vec_env)
        else:
            main_model = None

        obs = vec_env.reset()
        
        while games_completed < n_games:
            # We must manually fetch masks from all envs.
            action_masks = [vec_env.env_method("get_action_mask", indices=[i])[0] for i in range(n_envs)]
            action_masks = np.stack(action_masks)
            
            if main_model:
                action, _ = main_model.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                 # Random agent logic respecting masks
                 actions = []
                 for mask in action_masks:
                     valid_actions = np.flatnonzero(mask)
                     if len(valid_actions) > 0:
                        actions.append(np.random.choice(valid_actions))
                     else:
                        actions.append(0) 
                 action = np.array(actions)
            
            obs, rewards, dones, infos = vec_env.step(action)
            
            for i, done in enumerate(dones):
                if done:
                    games_completed += 1
                    
                    # Logic: Did Main Agent (Player 0) win?
                    # We check info['winners'] which contains a list of player IDs [0, 1, 2]
                    info = infos[i]
                    if "winners" in info:
                        if 0 in info["winners"]:
                            wins += 1
                        
                    if games_completed % 10 == 0 or games_completed == n_games:
                        print(f"Progress: {games_completed}/{n_games} | Wins (P0): {wins}", end="\r")
                    
                    if games_completed >= n_games:
                        break
                        
    finally:
        vec_env.close()

    winrate = wins / games_completed if games_completed > 0 else 0
    print(f"\nResult: {winrate:.2%} Winrate (as Player 0)")
    
    return {
        "games": games_completed,
        "wins_p0": wins,
        "winrate_p0": winrate
    }

def evaluate_fair(model_a_path, model_b_path, n_total_games):
    """
    Evaluates Model A vs Model B by playing two halves with swapped roles.
    1. A vs B (A is P0)
    2. B vs A (A is P1, P2) -> Wait, run_batch only tracks P0 wins!
    
    So here is the interpretation:
    Run 1: A vs B,B.  Result = How good is A as a Solo Player against B.
    Run 2: B vs A,A.  Result = How good is B as a Solo Player against A.
    
    If A is better than B, then:
    - WR(A vs B) should be HIGH.
    - WR(B vs A) should be LOW.
    """
    
    n_half = n_total_games // 2
    
    # --- Phase 1: A vs B ---
    # Model A is Main (Player 0). Opponent is Model B.
    # If model_b_path is None, opp_type is random.
    opp_type_1 = "model" if model_b_path else "random"
    
    stats_1 = run_batch(
        main_model_path=model_a_path,
        opp_type=opp_type_1, 
        opp_path=model_b_path, 
        n_games=n_half, 
        label="Phase 1 (Model A as P0)"
    )
    
    # --- Phase 2: B vs A ---
    # Model B is Main (Player 0). Opponent is Model A.
    # If model_a_path is None (Random A), opp_type is random.
    opp_type_2 = "model" if model_a_path else "random"
    
    stats_2 = run_batch(
        main_model_path=model_b_path,
        opp_type=opp_type_2, 
        opp_path=model_a_path, 
        n_games=n_half, 
        label="Phase 2 (Model B as P0 vs Model A)"
    )
    
    # --- Aggregation ---
    wr_a_solo = stats_1["winrate_p0"]
    wr_b_solo = stats_2["winrate_p0"]
    
    # Winrate of A against B (Solo)
    print("\n" + "="*40)
    print(f"FAIR EVALUATION RESULTS ({n_total_games} games)")
    print("="*40)
    print(f"Model A: {model_a_path if model_a_path else 'Random'}")
    print(f"Model B: {model_b_path if model_b_path else 'Random'}")
    print("-" * 40)
    print(f"1. When A is Solo (vs 2xB): A wins {wr_a_solo:.1%}")
    print(f"2. When B is Solo (vs 2xA): B wins {wr_b_solo:.1%} (=> A team wins {1-wr_b_solo:.1%})")
    print("-" * 40)
    
    # Final Score: Average of "A winning as Solo" and "A winning as Team (meaning B lost)"
    # This is a bit heuristic but gives a single number.
    combined_score = (wr_a_solo + (1.0 - wr_b_solo)) / 2
    print(f"Combined Score (A Performance): {combined_score:.1%}")
    print("="*40)


if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows support
    gym.logger.min_level = gym.logger.ERROR
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to main model zip")
    # Opponent defaults to random if not specified
    parser.add_argument("--opp", type=str, default="random", help="'random' or path to opponent model zip")
    parser.add_argument("--n", type=int, default=100, help="Total number of games")
    
    args = parser.parse_args()
    
    path_a = args.main 
    path_b = args.opp if args.opp != "random" else None
    
    evaluate_fair(path_a, path_b, args.n)