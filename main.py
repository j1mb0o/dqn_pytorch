import torch
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    
    agent = DQNAgent(n_observations=4,
                    n_actions=2, 
                    replay_state_shape=env.observation_space.shape,
                    batch_size=32,
                    hidden_layers=3,
                    neurons=128,
                    update_freq=1000
                    )

    n_episodes = 10
    r = []
    for i in range(n_episodes):
        total_reward = 0
        done = False
        trunc = False
        state, _ = env.reset()
        # print(type(state))
        print(f'Begining episode {i}',end='  ')
        while not done and not trunc:

            x = agent.q_net.forward(torch.from_numpy(state))
            action = agent.exploration.act(x, epsilon=0.1)

            state_next, reward, done, trunc, _ = env.step(action)
            agent.memory.store_transition(state,action,reward,state_next, done)
            agent.learn()
            state = state_next
            total_reward += 1

            
        print(f' Rewars: {total_reward}')
        r.append(total_reward)
    plt.plot(r)
    plt.show()
    print(torch.cuda.is_available())