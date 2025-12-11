import csv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from autodriving2d.envs import CityDrive


model = PPO.load("models/archived_not_random/ppo_citydrive+320k")

env = CityDrive(render_mode=None, randomize_start=False)


with open("results_dqn.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["goal_reached", "episode_score", "num_training_steps",
                     "action_space", "algorithm"])
    
    for episode in range(80):  # or however many you need
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        goal = False

        while True:
            a, _ = model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, info = env.step(a)
            total_reward += r
            steps += 1

            if terminated:
                goal = True
            if terminated or truncated or steps > 2000:
                break

        writer.writerow([goal, total_reward, steps, "continuous", "PPO"])

env.close()

