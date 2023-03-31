import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn_agent import DQNAgent
if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 128
    GAMMA = 1
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    n_actions = env.action_space.n
    state, info = env.reset()
    agent  = DQNAgent(len(state), n_actions,LR)

    num_episodes = 500

    for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.select_action(state,0.1,env)
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
            agent.optimize_model(batch_size=BATCH_SIZE, GAMMA=GAMMA)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break
        print(f'Episode {i_episode}, reward {t}')