import torch
import torch.optim as optim

from exploration import Exploration
from replay_buffer import ReplayBuffer
from network import DQNNetwork



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, n_observations, n_actions, replay_buffer_size=10e6, exploration_policy='egreedy', learning_rate=10e-3, batch_size= 32, use_replay_buffer= True) -> None:
        
        # Create the Q_net and the target_network
        self.q_net = DQNNetwork(n_observations, n_actions)
        self.target_netw = DQNNetwork(n_observations, n_actions)
        #  Copy the parameters of Q_net to target_net
        self.target_network.load_state_dict(self.target_network.state_dict())
        set.optimizer = optim.SGD(self.q_net.parameters(), lr= learning_rate)

        self.memory = ReplayBuffer(replay_buffer_size)

    def optimize_model(self):
        pass

    def train(self):
        pass