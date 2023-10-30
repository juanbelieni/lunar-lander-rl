import gymnasium as gym
import torch
from model import Agent
from args import args
from env import create_env, create_envs


human_env = create_env(human=True)
envs = create_envs(n=args.envs, randomize=False)

agent = Agent().to(args.device)

envs_wrapper = gym.wrappers.RecordEpisodeStatistics(
    envs,
    deque_size=args.envs * args.steps
)

actor_losses = []
entropies = []

for epoch in range(args.epochs):
    print(f"Training epoch {epoch}")

    ep_rewards = torch.zeros(args.steps, args.envs).to(args.device)
    ep_action_log_probs = torch.zeros(args.steps, args.envs).to(args.device)
    ep_state_values = torch.zeros(args.steps, args.envs).to(args.device)
    masks = torch.zeros(args.steps, args.envs).to(args.device)

    if epoch == 0:
        states, _ = envs_wrapper.reset(seed=42)

    for step in range(args.steps):
        actions, action_log_probs, state_values, entropy = agent.select_action(
            torch.from_numpy(states).to(args.device)
        )

        states, rewards, terminated, _, _ = envs_wrapper.step(
            actions.cpu().numpy()
        )

        ep_rewards[step] = torch.tensor(rewards, device=args.device)
        ep_action_log_probs[step] = action_log_probs
        ep_state_values[step] = torch.squeeze(state_values)

        masks[step] = torch.tensor([not term for term in terminated])

    actor_loss, critic_loss = agent.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_state_values,
        entropy,
        masks
    )

    print(f"Loss: actor  = {actor_loss:.2f}")
    print(f"      critic = {critic_loss:.2f}")

    agent.update_parameters(actor_loss, critic_loss)

    if (epoch + 1) % 8 == 0:
        with torch.no_grad():
            state, _ = human_env.reset()
            done = False
            truncated = False

            while not done and not truncated:
                state = torch.FloatTensor(state).reshape(1, 8).to(args.device)
                actions, _, _, _ = agent.select_action(state)
                action = actions.cpu().numpy()[0]
                state, _, terminated, truncated, _ = human_env.step(action)
