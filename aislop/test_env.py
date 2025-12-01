"""
Test script to verify the Gymnasium environment works correctly.
"""
from stable_baselines3.common.env_checker import check_env
from aislop.env import KuhhandelEnv

def test_env_basic():
    """Test basic environment functionality."""
    print("Creating environment...")
    env = KuhhandelEnv(num_players=3)
    
    print("Running environment checks...")
    try:
        check_env(env, warn=True)
        print("✅ Environment passes all checks!")
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        return False
    
    print("\nTesting reset...")
    obs, info = env.reset(seed=42)
    print(f"Observation keys: {obs.keys()}")
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Action mask shape: {obs['action_mask'].shape}")
    print(f"Valid actions count: {obs['action_mask'].sum()}")
    
    print("\nTesting step...")
    # Take a valid action (pass)
    action = 10
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    
    print("\nRunning a short episode...")
    env.reset(seed=123)
    total_reward = 0
    steps = 0
    max_steps = 100
    
    while steps < max_steps:
        # Sample a valid action
        valid_actions = obs['action_mask'].nonzero()[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]
        else:
            action = 10  # Default to pass
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"Episode finished after {steps} steps")
            print(f"Total reward: {total_reward}")
            if 'final_scores' in info:
                print(f"Final scores: {info['final_scores']}")
            break
    
    print("\n✅ All basic tests passed!")
    return True

if __name__ == "__main__":
    test_env_basic()
