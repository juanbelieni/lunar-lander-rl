import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
from datetime import datetime


class Agent(nn.Module):
    def __init__(
        self,
        gamma=0.99,
        alpha=0.95,
        entropy_coef=0.01,
        actor_lr=0.001,
        critic_lr=0.005
    ):
        super().__init__()

        self.reset_id()

        self.gamma = gamma
        self.alpha = alpha
        self.entropy_coef = entropy_coef
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        self.critic = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_lr
        )

    def forward(self, states: torch.Tensor):
        return self.actor(states)

    def select_action(self, states: torch.Tensor, tau=0.9):
        q = self.forward(states)

        probs = torch.nan_to_num(
            F.gumbel_softmax(q, tau=tau, dim=-1),
            nan=0.25
        )

        actions = torch.distributions.Categorical(probs).sample()

        return q, probs, actions

    def get_losses(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ):
        actions = actions.long().unsqueeze(-1)

        q, probs, _ = self.select_action(states)
        q = torch.gather(q, dim=1, index=actions)[:, 0]

        q_target, probs_target, _ = self.select_action(next_states)

        delta = torch.sum(q_target * probs_target, dim=-1)
        delta = rewards + self.gamma * masks * delta - q

        q_target = q + self.alpha * delta

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q, q_target.detach())

        return loss

    def update_parameters(self, actor_loss: torch.Tensor):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.version += 1

    def log(
        self,
        rewards_mean,
        loss,
    ):
        path = f"states/{self.name}"
        if not os.path.exists(path):
            os.makedirs(path)

        with open(f"{path}/log.csv", "a") as file:
            file.write(
                f"{self.version},{rewards_mean},{loss}\n")

    def save(self):
        path = f"states/{self.name}"
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self, f"{path}/{self.version:05d}.pt")

    def reset_id(self):
        self.name = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        self.version = 0
