import gymnasium as gym
from stable_baselines3 import PPO
from autodriving2d.envs import CityDrive
from stable_baselines3.common.vec_env import SubprocVecEnv
import os

def make_env():
    return CityDrive(render_mode=None)

if __name__ == "__main__":
    steps = 100_000
    initial_model_path = "models/ppo_initial.zip"

    # Making sure new model wont override others
    base_path = "models/ppo_citydrive" 
    for iter in range(100):
        out_path = base_path+str(iter+1)+".zip"
        if not os.path.isfile(out_path):
            break

    # Multiprocess training to speed up dev
    env = SubprocVecEnv([make_env for _ in range(5)]) 

    if not os.path.isfile(initial_model_path):
        # new initial model if it doesn't exist already
        model = PPO("MultiInputPolicy", 
                    env, 
                    verbose=1, 
                    learning_rate=3e-4
                    )
        out_path = "models/ppo_initial"
        steps = 300_000
    else:
        model = PPO.load(initial_model_path, env=env)

    # Train
    print("Training...")
    model.learn(total_timesteps=steps)
    model.save(out_path)
    print("Done.")
    env.close()
