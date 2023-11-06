import gymnasium as gym
import numpy as np
import torch
from model import Agent
from args import args
from env import create_env, create_envs

match args.command:
    case "train":
        envs = create_envs(n=args.envs, randomize=args.randomize)

        if args.load:
            agent: Agent = torch.load(args.load)
            agent.reset_id()
        else:
            agent = Agent()

        agent = agent.to(args.device)

        envs_wrapper = gym.wrappers.RecordEpisodeStatistics(
            envs,
            deque_size=args.envs * args.steps
        )

        actor_losses = []
        entropies = []

        for epoch in range(args.epochs):
            print(f"Training epoch {epoch}")

            S, E = args.steps, args.envs

            ep_rewards = torch.zeros(S, E).to(args.device)
            ep_action_log_probs = torch.zeros(S, E).to(args.device)
            ep_state_values = torch.zeros(S, E).to(args.device)
            ep_entropies = torch.zeros(S, E).to(args.device)
            ep_masks = torch.zeros(S, E).to(args.device)

            if epoch == 0:
                states, _ = envs_wrapper.reset(seed=42)

            for step in range(args.steps):
                actions, action_log_probs, state_values, entropies = agent.select_action(
                    torch.from_numpy(states).to(args.device)
                )

                actions = actions.cpu().numpy()
                actions[np.random.rand(E) < 0.05] = np.random.randint(4)

                states, rewards, terminated, _, _ = envs_wrapper.step(actions)

                ep_rewards[step] = torch.tensor(rewards, device=args.device)
                ep_action_log_probs[step] = action_log_probs
                ep_state_values[step] = torch.squeeze(state_values)
                ep_entropies[step] = entropies

                ep_masks[step] = torch.tensor(
                    [not term for term in terminated])

            actor_loss, critic_loss = agent.get_losses(
                ep_rewards,
                ep_action_log_probs,
                ep_state_values,
                ep_entropies,
                ep_masks
            )

            print(f"Loss: actor  = {actor_loss:.2f}")
            print(f"      critic = {critic_loss:.2f}")

            agent.log(
                rewards_mean=ep_rewards.mean().item(),
                state_values_mean=ep_state_values.mean().item(),
                action_log_probs_mean=ep_state_values.mean().item(),
                entropies_mean=ep_entropies.mean().item(),
                actor_loss=actor_loss.item(),
                critic_loss=critic_loss.item(),
            )

            agent.update_parameters(actor_loss, critic_loss)

            if agent.version % args.save_interval == 0:
                agent.save()
                print("Saving model.")

            print("")

    case 'play':
        env = create_env(human=True)
        agent = torch.load(args.path).to(args.device)

        while True:
            with torch.no_grad():
                state, _ = env.reset()
                done = False
                truncated = False

                while not done and not truncated:
                    state = torch.FloatTensor(
                        state).reshape(1, 8).to(args.device)
                    actions, _, _, _ = agent.select_action(state)
                    action = actions.cpu().numpy()[0]
                    state, _, terminated, truncated, _ = env.step(action)
