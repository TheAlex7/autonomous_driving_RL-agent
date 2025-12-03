import gymnasium as gym
from stable_baselines3 import DQN
from autodriving2d.envs import CityDrive

# Create environment
env = CityDrive(render_mode=None)

# DQN model
model = DQN(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
    exploration_fraction=0.1,
    exploration_final_eps=0.05
)

# Train
print("trainin..")
model.learn(total_timesteps=300000)
model.save("dqn_citydrive")
env.close()
