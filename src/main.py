import gymnasium as gym
import numpy as np


def create_env(human=False):
    env = gym.make(
        "LunarLander-v2",
        render_mode="human" if human else None,
        continuous=False,
    )

    env.reset()

    return env


env = create_env(human=True)

while True:
    action = np.random.randint(0, 4)
    env.step(action)
    env.render()

exit(0)
