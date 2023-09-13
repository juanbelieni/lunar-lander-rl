import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
# import torch.functional as F


def create_env(human=False):
    env = gym.make(
        "LunarLander-v2",
        render_mode="human" if human else None,
        continuous=False,
    )

    env.reset()

    return env


def create_model(dropout=0.1):
    model = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 4),
        nn.Softmax(dim=1),
    )

    return model


def play_game(env, model, p=0):
    state, _ = env.reset()
    done = False
    truncated = False

    actions = []
    rewards = []

    while not done and not truncated:
        state = torch.FloatTensor(state).reshape(1, 8)
        action = model(state)

        if np.random.rand() < p:
            action = np.random.randint(4)
            state, reward, done, truncated, _ = env.step(action)
        else:
            state, reward, done, truncated, _ = env.step(
                action.argmax().item(),
            )

            actions.append(action)
            rewards.append(reward)

    return actions, rewards


env = create_env()
human_env = create_env(human=True)
model = create_model()
optmizer = torch.optim.AdamW(model.parameters(), lr=0.001)

i = 0

while True:
    optmizer.zero_grad()

    loss = 0

    for _ in range(20):
        actions, rewards = play_game(env, model)
        T = len(actions)
        for t, (action, reward) in enumerate(zip(actions, rewards)):
            loss -= action.max() * (reward + rewards[-1] * (T - t) ** -2)

    loss.backward()
    optmizer.step()

    print(f"{loss = }")

    i += 1
    if i % 10 == 0:
        play_game(human_env, model)

exit(0)
