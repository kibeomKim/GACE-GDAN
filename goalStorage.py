import torch

import numpy as np
from random import sample
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('goal', 'label'))


class goalStorage(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.position = 0

    def put(self, state, label):
        goal = np.concatenate([state[0], state[1], state[2], state[3]]) / 255.
        self.memory.append(Transition(goal, label))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return Transition(*(zip(*transitions)))

    def __len__(self):
        return len(self.memory)

    def len(self):
        return len(self.memory)
