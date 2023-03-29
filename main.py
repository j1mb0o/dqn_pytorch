import torch
import gymnasium as gym
import argparse

from dqn_agent import DQNAgent

if __name__ == '__main__':

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    state, info = env.reset()
    print(info)
    print(env.observation_space.shape)    


    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(state)
