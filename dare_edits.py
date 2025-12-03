from stable_baselines3 import DQN
from autodriving2d.envs import CityDrive
import pygame

env = CityDrive(render_mode="human")
model = DQN.load("dqn_citydrive")

quit = False
while not quit:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
                break

        a, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or terminated or truncated:
            print(f"\naction {a}")
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated or quit:
            break

env.close()
