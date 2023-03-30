import numpy as np
import torch

class Exploration:
    def __init__(self, n_actions, policy = 'egreedy') -> None:
        #self.input_shape = input_shape
        self.n_actions = n_actions
        self.policy = policy
    
    def act(self,x,epsilon=None,temp=None):

        if self.policy == 'egreedy':
            if epsilon is None:
                raise KeyError('Provide an epsilon')
            if np.random.uniform(0,1) < epsilon:
                return np.random.randint(0, self.n_actions)
            else:
                try:
                    return np.random.choice(torch.where(x == torch.max(x))[0])
                except:
                    return np.argmax(x)
        elif self.policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            # soft = softmax(self.Q_sa[s], temp)
            x = x / temp
            z = x - max(x)
            soft = np.exp(z) / np.sum(np.exp(z))
            return np.random.choice(range(self.n_actions), p=soft)

        else:
            raise KeyError('Select an exploration policy')
            
if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6]
    x = np.array(x)
    # print(np.random.choice(x))
    exp = Exploration(6)
    # print(np.random.choice(6))
    # x.n_actions()
    # action = exp.greedy(x)
    action = exp.act(x, epsilon=0.5, temp=0.1)
    # print(action)
