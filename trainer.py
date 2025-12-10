import gymnasium as gym
from stable_baselines3 import PPO
from autodriving2d.envs import CityDrive
from stable_baselines3.common.vec_env import SubprocVecEnv
import os

def make_env():
    return CityDrive(render_mode=None)

if __name__ == "__main__":
    for _ in range(1): # trains for a total of 200 * 100,000 = 20M steps
        STEPS = 100_000
        INITIAL_MODEL = "models/ppo_initial"
        INITIAL_MODEL_PATH = INITIAL_MODEL + ".zip"

        # Making sure new model wont override others and continue iterating
        prev_path = None
        for iter in range(1000):
            additional_steps = (iter+1) * 100
            if additional_steps < 1000:
                additional_steps = str(additional_steps)+"k"
            else:
                additional_steps = f"{additional_steps/1000}m".replace(".0","").replace(".","_") # rid of trailing 0 and rid of period in name
            base_path = f"models/ppo_citydrive+{additional_steps}" 
            out_path = base_path+".zip"
            if not os.path.isfile(out_path):
                if iter == 0:
                    prev_path = INITIAL_MODEL
                break
            else:
                prev_path = base_path

        # Multiprocess training to speed up dev
        env = make_env()# SubprocVecEnv([make_env for _ in range(8)]) 

        if not os.path.isfile(INITIAL_MODEL_PATH):
            # new initial model if it doesn't exist already
            model = PPO("MultiInputPolicy", 
                        env, 
                        verbose=1, 
                        batch_size = 4096,      # mini-batch size
                        learning_rate = 3e-3,  # Adam optimizer
                        ent_coef = 0.05        # encourages exploration
                        # n_steps = 2048,        # per update
                        # n_epochs = 10,         # number of SGD passes per update
                        # gamma = 0.99,          # discount factor
                        # gae_lambda = 0.95,
                        # clip_range = 0.2,
                        )
            STEPS = 50_000 # initial should be trained on 300,000 time steps
            base_path = INITIAL_MODEL
        else:
            model = PPO.load(prev_path, env=env)

        # Train
        print("Training...")
        model.learn(total_timesteps=STEPS)
        model.save(base_path)
        print("Done. Saved to:", out_path)
        env.close()
