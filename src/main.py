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

        epsilon = 1

        for epoch in range(args.epochs):
            print(f"Training epoch {epoch}")

            S, E = args.steps, args.envs

            ep_states = torch.zeros(S, E, 8).to(args.device)
            ep_actions = torch.zeros(S, E).to(args.device)
            ep_rewards = torch.zeros(S, E).to(args.device)
            ep_action_log_probs = torch.zeros(S, E).to(args.device)
            ep_state_values = torch.zeros(S, E).to(args.device)
            ep_entropies = torch.zeros(S, E).to(args.device)
            ep_masks = torch.ones(S, E).to(args.device)

            # if epoch == 0:
            states, _ = envs_wrapper.reset(seed=42)
            states = torch.from_numpy(states).to(args.device)

            for step in range(args.steps):
                ep_states[step] = states
                actions, action_log_probs, state_values, entropies = agent.select_action(
                    states
                )

                actions = actions.cpu().numpy()
                actions[np.random.rand(E) < epsilon] = np.random.randint(4)

                states, rewards, terminated, _, _ = envs_wrapper.step(actions)
                states = torch.from_numpy(states).to(args.device)

                ep_actions[step] = torch.from_numpy(actions).to(args.device)
                ep_rewards[step] = torch.tensor(rewards).to(args.device)
                ep_action_log_probs[step] = action_log_probs
                ep_state_values[step] = torch.squeeze(state_values)
                ep_entropies[step] = entropies

                if step < args.steps - 1:
                    ep_masks[step + 1] = (
                        ep_masks[step] *
                        torch.tensor([not t for t in terminated])
                    )

            actor_loss, critic_loss = agent.get_losses(
                ep_states,
                ep_actions,
                ep_rewards,
                ep_action_log_probs,
                ep_state_values,
                ep_entropies,
                ep_masks
            )

            # print(ep_masks)

            print(f"Loss    = {actor_loss:.2f}")
            print(
                f"Reward  = {(ep_masks * ep_rewards).sum(axis=0).mean().item()}")
            print(f"Epsilon = {epsilon}")

            agent.log(
                rewards_mean=(ep_masks * ep_rewards).sum(axis=0).mean().item(),
                loss=actor_loss.item(),
            )

            agent.update_parameters(actor_loss, critic_loss)

            if agent.version % args.save_interval == 0:
                agent.save()
                print("Saving model.")

            epsilon = max(epsilon * 0.995, 0.15)

            print("")

    case 'play':
        env = create_env(human=True)
        agent = torch.load(args.path).to(args.device)

        while True:
            with torch.no_grad():
                state, _ = env.reset()
                terminated = False
                truncated = False

                while not terminated and not truncated:
                    state = torch.FloatTensor(
                        state).reshape(1, 8).to(args.device)
                    actions, _, _, _ = agent.select_action(state)
                    action = actions.cpu().numpy()[0]
                    state, reward, terminated, truncated, _ = env.step(action)

                    print(
                        f"Terminated = {terminated}, Truncated = {truncated}, Reward = {reward}")

                # input("")
