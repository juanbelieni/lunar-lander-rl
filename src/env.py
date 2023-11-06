
import gymnasium as gym
import numpy as np
from args import args


def create_env(human=False, randomize=False):
    env = gym.make(
        "LunarLander-v2",
        render_mode="human" if human else None,
        max_episode_steps=args.steps,
        enable_wind=False if not randomize or np.random.rand() < 0.5 else True,
        wind_power=np.random.uniform(0, 20) if randomize else 15,
        turbulence_power=np.random.uniform(0, 2) if randomize else 1.5
    )

    env.reset()

    return env


def create_envs(n: int, randomize=False):
    envs = gym.vector.AsyncVectorEnv([
        lambda: create_env(randomize=randomize)
        for _ in range(args.envs)
    ])

    envs.reset()

    return envs
