import gymnasium as gym
import torch
from model import Agent
from args import args


def create_env(human=False):
    env = gym.make(
        "LunarLander-v2",
        render_mode="human" if human else None,
        max_episode_steps=args.steps,
    )

    env.reset()

    return env


def play_game(env, agent, p=0):
    state, _ = env.reset()
    done = False
    truncated = False

    while not done and not truncated:
        state = torch.FloatTensor(state).reshape(1, 8).to(args.device)
        actions, _, _, _ = agent.select_action(state)
        state, _, terminated, truncated, _ = env.step(
            actions.cpu().numpy()[0]
        )


envs = gym.vector.AsyncVectorEnv([
    lambda: create_env()
    for _ in range(args.envs)
])

human_env = create_env(human=True)

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
        states, info = envs_wrapper.reset(seed=42)

    for step in range(args.steps):
        actions, action_log_probs, state_values, entropy = agent.select_action(
            torch.from_numpy(states).to(args.device)
        )

        states, rewards, terminated, truncated, infos = envs_wrapper.step(
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

    if epoch % 10 == 9:
        with torch.no_grad():
            play_game(human_env, agent)
