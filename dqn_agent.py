import random
import torch
import math
import torch.optim as optim
import torch.nn as nn

from replay_buffer import ReplayMemory
from model import DQN3L, DQN5L, DQN7L
from collections import namedtuple

class DQNAgent:
    def __init__(self, n_states, n_actions,LR, memory_size, layers, neurons, optimizer) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # pass
        if layers == 3:
            self.policy_net = DQN3L(n_states, n_actions, neurons).to(self.device)
            self.target_net = DQN3L(n_states, n_actions, neurons).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        elif layers == 5:
            self.policy_net = DQN5L(n_states, n_actions, neurons).to(self.device)
            self.target_net = DQN5L(n_states, n_actions, neurons).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.policy_net = DQN7L(n_states, n_actions, neurons).to(self.device)
            self.target_net = DQN7L(n_states, n_actions, neurons).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=LR)
        else:
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(memory_size)

    
    def select_action(self, state, epsilon, temp,env,policy):
        if policy == 'egreedy':
            sample = random.random()
            if sample > epsilon:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                x = self.policy_net(state)[0]/temp
                z = x - self.policy_net(state).max(1)[0]
                softmax = torch.exp(z) / torch.sum(torch.exp(z))
                
                return torch.multinomial(softmax,1).view(1,1)



    def optimize_model(self, batch_size, GAMMA):
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            # next_state_values[non_final_mask] = self.policy_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
