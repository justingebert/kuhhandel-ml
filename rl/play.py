from rl.train import mask_valid_action
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

env = KuhhandelEnv()
env = ActionMasker(env, mask_valid_action)

wins = 0
total_score = 0
opponent_scores = 0

model_path = ""
model = MaskablePPO.load(model_path)

obs, _ = env.reset()
done = False
truncated = False

while not (done or truncated):
    action_masks = mask_valid_action(env)
    action, _ = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)

winner = env.unwrapped.game.get_winner()
print(f"Winner: {winner}")