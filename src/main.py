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
            deque_size=100
        )

        actor_losses = []
        entropies = []

        epsilon = 1

        states, _ = envs_wrapper.reset(seed=42)

        for _ in range(args.steps):
            states, _, _, _, _ = envs_wrapper.step(
                np.random.randint(4, size=args.envs))

        states = torch.from_numpy(states).to(args.device)

        for epoch in range(args.epochs):
            print(f"Training epoch {epoch}")

            ep_states = torch.FloatTensor([])
            ep_next_states = torch.FloatTensor([])
            ep_actions = torch.FloatTensor([])
            ep_rewards = torch.FloatTensor([])
            ep_masks = torch.FloatTensor([])

            for _ in range(5):
                _, _, actions = agent.select_action(states)
                actions = actions.cpu().numpy()

                next_states, rewards, terminated, _, _ = envs_wrapper.step(
                    actions)
                next_states = torch.from_numpy(next_states).to(args.device)

                actions = torch.from_numpy(actions).to(args.device)
                rewards = torch.tensor(rewards).to(args.device)
                masks = torch.tensor([not t for t in terminated])

                ep_states = torch.concat([ep_states, states])
                ep_next_states = torch.concat([ep_next_states, next_states])
                ep_actions = torch.concat([ep_actions, actions])
                ep_rewards = torch.concat([ep_rewards, rewards])
                ep_masks = torch.concat([ep_masks, masks])

                states = next_states

            actor_loss = agent.get_losses(
                states,
                next_states,
                actions,
                rewards,
                masks
            )

            print(f"Loss    = {actor_loss:.2f}")
            print(f"Reward  = {np.mean(list(envs_wrapper.return_queue))}")

            agent.update_parameters(actor_loss)

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
                    _, _, actions = agent.select_action(state)
                    action = actions.cpu().numpy()[0]
                    state, reward, terminated, truncated, _ = env.step(action)

                    print(
                        f"Terminated = {terminated}, Truncated = {truncated}, Reward = {reward}")

                # input("")
