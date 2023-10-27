import torch
import torch.nn as nn
from args import args


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
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
            lr=0.001
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=0.001
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

        return (actions, action_log_probs, state_values, entropy)

    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        state_values: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
    ):
        T = len(rewards)
        advantages = torch.zeros(T, args.envs).to(args.device)

        gae = 0.0

        for t in reversed(range(T - 1)):
            td_error = rewards[t] + 0.99 * masks[t] * \
                state_values[t + 1] - state_values[t]

            gae = td_error + 0.999 * 0.95 * masks[t] * gae
            advantages[t] = gae

        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() -
            0.01 * entropy.mean()
        )

        critic_loss = advantages.pow(2).mean()

        return actor_loss, critic_loss

    def update_parameters(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
