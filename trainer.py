import gymnasium as gym
from stable_baselines3 import PPO
from autodriving2d.envs import CityDrive

# Create environment
env = CityDrive(render_mode=None)

model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=3e-4)

# Train
print("Training...")
model.learn(total_timesteps=2_000_000)
model.save("dqn_citydrive")
print("Done.")
env.close()
