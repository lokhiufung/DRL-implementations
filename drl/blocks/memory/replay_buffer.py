import random
import collections

# from drl.core.modules import NonTranableModule

import torch


class ReplayBuffer:
    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self.output_type = collections.namedtuple(
            'Transition',
            ['state', 'reward', 'next_state', 'action', 'done'],
        )
        self.buffer = collections.deque(maxlen=self.capacity)  # once maxlen is rearched, the left sample will be poped out

    def append(self, state, reward, next_state, action, done):
        self.buffer.append(self.output_type(state, reward, next_state, action, done))

    def get_batch(self, batch_size):
        """generator for getting randomly batched samples"""
        batch = random.sample(self.buffer, k=batch_size)
        yield batch

    def __len__(self):
        return len(self.buffer)

