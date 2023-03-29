import numpy as np
import torch
from torch.nn import MSELoss
import torch.optim as optim
import gymnasium as gym

from exploration import Exploration
from replay_buffer import ReplayBuffer
from network import DQN



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self,
                n_observations,
                n_actions, 
                replay_state_shape,
                replay_buffer_size=10e6, 
                exploration_policy='egreedy', 
                learning_rate=10e-3, 
                batch_size= 32, 
                use_replay_buffer= True,
                gamma=1,
                update_freq = 10
                ) -> None:
        
        # Create the Q_net and the target_network
        self.q_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        #  Copy the parameters of Q_net to target_net
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.SGD(self.q_net.parameters(), lr= learning_rate)

        self.memory = ReplayBuffer(int(replay_buffer_size),input_shape= replay_state_shape)
        self.batch_size = batch_size

        self.exploration = Exploration(n_actions=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma

        # Counters that indicate how target network is updated in method update_target_network
        self.update_counter = 0
        self.update_freq = update_freq


    def update_target_network(self):
        if self.update_counter % self.update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


    def memory_sample_to_tensor(self):
        states, actions, rewards, states_next, dones = self.memory.sample_buffer(self.batch_size)  

        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        states_next = torch.Tensor(states_next).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        return  states, actions, rewards, states_next, dones

    def learn(self):
            if self.memory.mem_cntr < self.batch_size:
                return

            self.optimizer.zero_grad()
            # TODO
            # self.replace_target_network()

            states, actions, rewards, states_next, dones = self.memory_sample_to_tensor()
            
            indices = np.arange(self.batch_size)
            q_pred = self.q_net.forward(states)[indices, actions]
            target = self.target_net.forward(states_next).max(dim=1)[0]


            target[dones] = 0.0
            q_target = rewards + self.gamma*target

            print(q_target.shape, q_pred.shape)
            print(q_pred)
            # loss = self.optimizer.loss(q_target, q_pred).to(self.device)
            loss = MSELoss().to(self.device)
            loss = loss(q_target, q_pred).to(self.device)
            loss.backward()
            # loss.backward()
            # self.optimizer.step()
            self.update_counter += 1

    def train(self):
        pass

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    state, info = env.reset()
    
    # print(env.action_space)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(state)
    # print(f'State before: {state}, {type(state)}')
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # print(f'State after: {state}, {type(state)}')
    # print(env.step(1))
    agent = DQNAgent(n_observations=4, replay_state_shape=env.observation_space.shape, n_actions=2, batch_size=2)
    # q_vals = agent.q_net(state).detach()
    agent.memory.store_transition(state,1,1,state, False)
    agent.memory.store_transition(state,1,1,state, True)

    agent.learn()
    # print(agent.q_net(state).detach())
    # print(agent.exploration.act(q_vals,epsilon=1))
