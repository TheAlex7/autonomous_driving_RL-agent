import csv
from stable_baselines3 import PPO
from autodriving2d.envs import CityDrive
import pygame

MODEL_PATH = "models/archived_not_random/ppo_citydrive+320k.zip"
NUM_EPISODES = 20
NUM_TRAINING_STEPS = 320_000  # you MUST fill this based on model filename

OUTPUT_FILE = "evaluation_results.csv"

env = CityDrive(render_mode=None, randomize_start=False)
model = PPO.load(MODEL_PATH)

results = []

for episode in range(NUM_EPISODES):
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    goal_reached = False

    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        total_reward += r
        steps += 1
        
        if terminated or truncated:
            # If your environment marks goal in info:
            if "lap_finished" in info and info["lap_finished"] == True:
                goal_reached = True

            break

    # Store data for the row
    results.append([
        goal_reached,
        total_reward,
        NUM_TRAINING_STEPS,
        "continuous",   # PPO model → continuous
        "PPO"
    ])

env.close()

# Write CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["goal_reached", "episode_score", "num_training_steps",
                     "action_type", "algorithm"])
    writer.writerows(results)

print("Saved results to:", OUTPUT_FILE)

