import gymnasium as gym
import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import argparse 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn_agent import DQNAgent



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exploration_p', type=str,default='egreedy', action='store')
    parser.add_argument('--no-replay-buffer',  action='store_true')
    parser.add_argument('--no-target-network', action='store_true')
    parser.add_argument('--learning-rate', type=float,default=1e-4)
    parser.add_argument('--batch-size',type=int, default=128)
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default= 0.1)
    parser.add_argument('--target-update',type=int, default=500)
    parser.add_argument('--memory-size',type=int,default=10e4)

    args = parser.parse_args()

    env = gym.make("CartPole-v1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TAU = 0.005
    # EPS_START = 0.9
    # EPS_END = 0.05
    # EPS_DECAY = 1000
    batch_size = args.batch_size
    gamma = args.gamma
    policy = args.exploration_p
    lr = args.learning_rate
    memory_size = int(args.memory_size)
    epsilon = args.epsilon
    temperature = args.temperature

    n_actions = env.action_space.n
    state, info = env.reset()
    agent  = DQNAgent(len(state), n_actions,lr, memory_size=memory_size)
    target_net_update_cnt = 0
    num_episodes = 500
    episode_durations = []

    if args.no_replay_buffer:
        batch_size = 1
        memory_size = 1
    ep_to_export = []

    print('Starting...')
    for run in range(5):
        episode_durations.clear()
        for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # for t in count():
            duration = 0
            done = False
            
            while not done:
                action = agent.select_action(state=state,epsilon=0.1,env=env,policy=policy,temp=temperature)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                agent.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                agent.optimize_model(batch_size=batch_size, GAMMA=gamma)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()

                if args.no_target_network:
                    agent.target_net.load_state_dict(policy_net_state_dict)
                else:
                    # https://ai.stackexchange.com/questions/21485/how-and-when-should-we-update-the-q-target-in-deep-q-learning
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

                duration += 1
                # if done:
            episode_durations.append(duration)
            print(f'Run{run},episode {i_episode}, reward {duration}')

        ep_to_export.append(episode_durations)
    # plt.plot(episode_durations)
    # plt.show()
    export = np.array(ep_to_export)
    export = np.mean(export, axis=0)
    np.save(f'batch_size_{batch_size}_epsilon_{epsilon}_exploration_p_{policy}_gamma_{gamma}_learning_rate_{lr}_memory_size_{memory_size}_no_replay_buffer_{args.no_replay_buffer}_no_target_network_{args.no_target_network}_target_update_{args.target_update}_temperature_{temperature}.npy',np.array(episode_durations))