import gymnasium as gym
from stable_baselines3 import PPO
from autodriving2d.envs import CityDrive
from stable_baselines3.common.vec_env import SubprocVecEnv
import os

def make_env():
    return CityDrive(render_mode=None)

if __name__ == "__main__":
    steps = 100_000
    initial_model = "models/ppo_initial2"
    initial_model_path = initial_model + ".zip"

    # Making sure new model wont override others
    base_path = "models/ppo_citydrive" 
    for iter in range(100):
        out_path = base_path+str(iter+1)+".zip"
        if not os.path.isfile(out_path):
            break

    # Multiprocess training to speed up dev
    env = make_env() # SubprocVecEnv([make_env for _ in range(4)]) 

    if not os.path.isfile(initial_model_path):
        # new initial model if it doesn't exist already
        model = PPO("MultiInputPolicy", 
                    env, 
                    verbose=1, 
                    n_steps = 2048,        # per update
                    batch_size = 256,      # mini-batch size
                    n_epochs = 10,         # number of SGD passes per update
                    learning_rate = 3e-4,  # Adam optimizer
                    gamma = 0.99,          # discount factor
                    gae_lambda = 0.95,
                    clip_range = 0.2,
                    ent_coef = 0.01        # encourages exploration
                    )
        steps = 300_000 # should be trained on 300,000 time steps
        out_path = "models/ppo_initial"
    else:
        model = PPO.load(initial_model, env=env)

    # Train
    print("Training...")
    model.learn(total_timesteps=steps)
    model.save(out_path)
    print("Done. Saved to:", out_path)
    env.close()
