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
        action_logits = self.actor(states)
        state_values = self.critic(states)
        return action_logits, state_values

    def select_action(self, states: torch.Tensor):
        action_logits, state_values = self.forward(states)
        action_dist = torch.distributions.Categorical(logits=action_logits)

        actions = action_dist.sample()
        action_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        state_values = self.critic(states)
        actions = torch.argmax(action_logits, dim=-1)

        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        state_values: torch.Tensor,
        entropies: torch.Tensor,
        masks: torch.Tensor,
    ):
        T = len(rewards)

        actions = actions.long().unsqueeze(-1)

        q, _ = self.forward(states)
        q = torch.gather(q, dim=2, index=actions)[:, :, 0]

        # qsa_next, _ = self.forward(states[1:])
        q_target = torch.zeros(T, args.envs)

        # acc = 0

        for t in reversed(range(T - 1)):
            delta = rewards[t] + self.gamma * q[t + 1] - q[t]
            # acc = q[t] + masks[t] * self.alpha * delta
            q_target[t] = q[t] + masks[t] * self.alpha * delta

        # qsa_next = torch.gather(qsa_next, dim=2, index=actions[1:])[:, :, 0]
        # qsa_next = masks[1:] * qsa_next
        # qsa_target = rewards[1:] + masks[1:] * self.lam * qsa_next
        # qsa_target = qsa_target[:-1]

        # q = masks * q
        # q_target = masks * q_target

        # print(q_target)

        loss_fn = nn.SmoothL1Loss()
        actor_loss = loss_fn(q, q_target.detach())

        # exit(0)

        # advantages = torch.zeros(T, args.envs).to(args.device)

        # gae = 0.0

        # for t in reversed(range(T - 1)):
        # #     td_error = rewards[t] + gamma * masks[t] * action_log_probs[t + 1] - value_preds[t]

        #     # gae = td_error + self.gamma * self.lam * masks[t] * gae
        #     gae +=
        #     advantages[t] = gae

        # actor_loss = -((advantages.detach() * action_log_probs).mean() +
        #                self.entropy_coef * entropies.mean())

        # critic_loss = advantages.pow(2).mean()
        critic_loss = torch.tensor(1)

        return actor_loss, critic_loss

    def update_parameters(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic_optimizer.step()

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
