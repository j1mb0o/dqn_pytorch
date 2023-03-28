import torch.nn as nn
import torch.nn.functional as F
import torch

class DQNNetwork(nn.Module):

    def __init__(self, n_observations, n_actions, n_neurons= 36):
        super(DQNNetwork, self).__init__()
        
        self.layer1 = nn.Linear(n_observations, n_neurons, bias= False)
        self.layer2 = nn.Linear(n_neurons, n_neurons, bias= False)
        self.layer3 = nn.Linear(n_neurons, n_actions, bias= False)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


if __name__ == '__main__':
    dqn = DQNNetwork(4,2)
    print(dqn.forward(torch.Tensor([1,2,3,4])))
    for param_tensor in dqn.state_dict():
        print(param_tensor, "\t", dqn.state_dict()[param_tensor].size())
