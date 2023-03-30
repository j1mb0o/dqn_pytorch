import numpy as np
import torch
from torch.nn import MSELoss
import torch.optim as optim
import gymnasium as gym

from exploration import Exploration
from replay_buffer import ReplayBuffer
from network import DQN

# TODO: Don't use replay buffer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self,
                n_observations,
                n_actions, 
                replay_state_shape,
                hidden_layers,
                neurons,
                replay_buffer_size=10e6, 
                exploration_policy='egreedy', 
                learning_rate=10e-3, 
                epsilon=0.1,
                batch_size= 32, 
                use_replay_buffer= True,
                gamma=1,
                update_freq = 10
                ) -> None:
        
        # Create the Q_net and the target_network
        self.q_net = DQN(n_observations, n_actions, n_layers_hidden=hidden_layers, n_neurons=neurons)
        self.target_net = DQN(n_observations, n_actions, n_layers_hidden=hidden_layers, n_neurons=neurons)

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
            self.update_target_network()

            states, actions, rewards, states_next, dones = self.memory_sample_to_tensor()
            
            indices = np.arange(self.batch_size)
            q_pred = self.q_net.forward(states)[indices, actions]
            target = self.target_net.forward(states_next).max(dim=1)[0]


            target[dones] = 0.0
            q_target = rewards + self.gamma*target

            # print(q_target.shape, q_pred.shape)
            # print(q_pred)
            # loss = self.optimizer.loss(q_target, q_pred).to(self.device)
            loss = MSELoss()
            loss = loss(q_target, q_pred).to(self.device)

            loss.backward()
            self.optimizer.step()
            self.update_counter += 1


