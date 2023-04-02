import torch.nn as nn
import torch.nn.functional as F

class DQN3L(nn.Module):

    def __init__(self, n_observations, n_actions,n_neurons):
        super(DQN3L, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.layer3 = nn.Linear(n_neurons, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN5L(nn.Module):

    def __init__(self, n_observations, n_actions,n_neurons):
        super(DQN5L, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_neurons)
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.layer3 = nn.Linear(n_neurons, n_neurons)
        self.layer4 = nn.Linear(n_neurons, n_neurons)
        
        self.layer5 = nn.Linear(n_neurons, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)
        

class DQN7L(nn.Module):

    def __init__(self, n_observations, n_actions,n_neurons):
        super(DQN7L, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_neurons)
        
        self.layer2 = nn.Linear(n_neurons, n_neurons)
        self.layer3 = nn.Linear(n_neurons, n_neurons)
        self.layer3 = nn.Linear(n_neurons, n_neurons)
        self.layer4 = nn.Linear(n_neurons, n_neurons)
        self.layer5 = nn.Linear(n_neurons, n_actions)
        self.layer6 = nn.Linear(n_neurons, n_neurons)

        self.layer7 = nn.Linear(n_neurons, n_neurons)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))        
        return self.layer7(x)

