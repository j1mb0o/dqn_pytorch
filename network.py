import torch.nn as nn
import torch.nn.functional as F
import torch

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_layers_hidden=3, n_neurons= 36):
        super(DQN, self).__init__()
        
        # self.layer1 = nn.Linear(n_observations, n_neurons, bias= False)
        # self.layer2 = nn.Linear(n_neurons, n_neurons, bias= False)
        # self.layer3 = nn.Linear(n_neurons, n_actions, bias= False)
        layers = []
        layers.append(nn.Linear(n_observations,n_neurons, bias=False))
        
        for layer in range(n_layers_hidden):
            layers.append(nn.Linear(n_neurons, n_neurons, bias=False))

        layers.append(nn.Linear(n_neurons, n_actions, bias=False))
        
        self.layers = nn.Sequential(*layers)
        
        
    def forward(self, x):
        # x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        for l in self.layers:
            x = F.relu(l(x))
        return x


