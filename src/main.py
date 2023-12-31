import gymnasium as gym
import numpy as np
import torch
from model import Agent
from args import args
from env import create_env, create_envs
from gym.wrappers.monitoring import video_recorder
from datetime import datetime

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
            deque_size=50
        )

        envs_wrapper.reset(seed=42)

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

            for _ in range(2):
                _, _, actions = agent.select_action(states)
                actions = actions.cpu().numpy()

                next_states, rewards, terminated, _, _ = envs_wrapper.step(
                    actions,
                )
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

            loss = agent.get_losses(
                ep_states,
                ep_next_states,
                ep_actions,
                ep_rewards,
                ep_masks
            )

            print(f"Reward  = {np.mean(list(envs_wrapper.return_queue))}")
            print(f"Loss    = {loss:.3f}")

            agent.update_parameters(loss)

            agent.log(
                rewards_mean=np.mean(list(envs_wrapper.return_queue)),
                loss=loss
            )

            if agent.version % args.save_interval == 0:
                agent.save()
                print("Saving model.")

            print("")

    case 'play':
        env = create_env(human=True)
        agent = torch.load(args.path).to(args.device)

        if args.render_mode == "record":
            now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            video_path = f"video/gameplay_{now}.mp4"
            vid = video_recorder.VideoRecorder(
                env, path=video_path, enabled=video_path is not None)
            
        for _ in range(10):
            with torch.no_grad():
                state, _ = env.reset()
                terminated = False
                truncated = False
                rewards = 0

                while not terminated and not truncated:
                    state = torch.FloatTensor(state).reshape(1, 8).to(args.device)
                    _, _, actions = agent.select_action(state)
                    action = actions.cpu().numpy()[0]
                    state, reward, terminated, truncated, _ = env.step(action)
                    rewards += reward
                    
                    if args.render_mode == "record":
                        frame = env.render()
                        vid.capture_frame()
            print(
                f"Terminated = {terminated}, Truncated = {truncated}, Reward = {rewards}")
            
        if args.render_mode == "record":
            vid.close()  
