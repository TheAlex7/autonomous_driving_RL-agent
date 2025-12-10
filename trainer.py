import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import RecurrentPPO
from autodriving2d.envs import CityDrive
import torch


def make_env():
    def _init():
        env = CityDrive(render_mode=None)
        return env
    return _init


if __name__ == "__main__":
    num_envs = 12   # use 8–12 on your 20-core CPU

    env_fns = [make_env() for _ in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=512,
        n_epochs=4,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        device=device,
        policy_kwargs=dict(
            lstm_hidden_size=256,
            n_lstm_layers=1
        )
    )

    print(model.policy.device)
    print("training...")
    model.learn(total_timesteps=300000)
    model.save("ppo_citydrive")
    vec_env.close()
