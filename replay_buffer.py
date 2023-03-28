#  Inpired from PyTorch's blog https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import deque
import numpy as np 
import random

class ReplayBuffer():
    
    def __init__(self, size) -> None:
        self.__buffer = deque([], maxlen= size)
    
    def __len__(self):
        return len(self.__buffer)

    # Adds the transition values to ReplayBuffer, when the buffer will reach its size limit (maxlen) it will remove the items at front
    def append(self, *args):
        self.__buffer.append((args))

    def get_sample(self, batch_size):
        # print(random.sample(self.__buffer, batch_size))
        return random.sample(self.__buffer, batch_size)

    def get_buffer(self):
        return self.__buffer

