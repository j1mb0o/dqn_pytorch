import numpy as np
import torch
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

        self.update_counter = 0
    
    def memory_sample_to_tensor(self):
        states, actions, rewards, states_next, dones = self.memory.sample_buffer(self.batch_size)  

        states = torch.Tensor(states).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        states_next = torch.Tensor(states_next).to(self.device)
        # dones = torch.Tensor(dones).to(self.device)

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

            loss = self.q_net.loss(q_target, q_pred).to(self.device)
            loss.backward()
            self.optimizer.step()
            self.learn_step_counter += 1

    def train(self):
        pass

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    state, info = env.reset()
    
    # print(env.action_space)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # print(f'State before: {state}, {type(state)}')
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # print(f'State after: {state}, {type(state)}')
    print(env.step(1))
    agent = DQNAgent(n_observations=4, replay_state_shape=env.observation_space.shape, n_actions=2, batch_size=2)
    # q_vals = agent.q_net(state).detach()
    agent.memory.store_transition([0.1,0.2,0.3,0.4],1,1,[0.1,0.2,0.3,0.4], False)
    agent.memory.store_transition([0.0,0.9,0.3,0.2],1,1,[0.4,0.5,0.6,1.1], False)

    agent.learn()
    # print(agent.q_net(state).detach())
    # print(agent.exploration.act(q_vals,epsilon=1))
