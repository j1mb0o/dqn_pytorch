#  Inpired from PyTorch's blog https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import numpy as np 

# class ReplayBuffer:
    
#     def __init__(self, size) -> None:
#         self.__buffer = deque([], maxlen= size)
    
#     def __len__(self):
#         return len(self.__buffer)

#     # Adds the transition values to ReplayBuffer, when the buffer will reach its size limit (maxlen) it will remove the items at front
#     def append(self, *args):
#         self.__buffer.append((args))

#     def get_sample(self, batch_size):

        

#         )

#     def get_buffer(self):
#         return self.__buffer

class ReplayBuffer:
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal