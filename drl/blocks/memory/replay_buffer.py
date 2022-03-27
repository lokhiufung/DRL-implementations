import random
import collections
from typing import List

import torch.nn as nn

from drl.core.transition import Transition


class ReplayBuffer(nn.Module):
    
    output_type = Transition
    
    def __init__(self, capacity=10**6):
        super().__init__()

        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)  # once maxlen is rearched, the left sample will be poped out

    def __len__(self):
        return len(self.buffer)

    def append(self, state, reward, next_state, action, done):
        self.buffer.append(self.output_type(state, reward, next_state, action, done))

    def get_batch(self, batch_size):
        """ Get randomly batched samples"""
        batch = random.sample(self.buffer, k=batch_size)
        return batch

    # def get_n_step_batches(self, batch_size, n_steps=20, gamma=0.999) -> List[collections.namedtuple]:
    #     """get n-steps (decayed) reward from an ordered Transition buffer, i.e, self.buffer.  

    #     :param batch_size: batch_size, for experience replay
    #     :type batch_size: int
    #     :param n_steps: [description], defaults to 20
    #     :type n_steps: int, optional
    #     :param gamma: decay factor, less than 1, i.e reward = reward + gamma**n * reward, defaults to 0.999
    #     :type gamma: float, optional
    #     :return: 
    #     :rtype: List[collections.namedtuple]
    #     """
    #     batch = [] 
        
    #     batch_indexes = random.sample(range(len(self.buffer)), k=batch_size)
        
    #     for index in batch_indexes:
    #         n_step_reward = 0.0
    #         n_steps = 0
    #         for i in range(n_steps):
    #             n_step_reward += (gamma**i) * self.buffer[index+i].reward
    #             n_steps = i + 1  # reminder: get the number of steps taken since the index
    #             if (self.buffer[index].done) or (index + i >= len(self.buffer)):
    #                 # if transit to terminal state or there is no next transition yet, break 
    #                 break
    #         batch.append(
    #             self.output_type(
    #                 state=self.buffer[index].state,
    #                 reward=n_step_reward,
    #                 next_state=self.buffer[index].next_state,
    #                 action=self.buffer[index].action,
    #                 done=self.buffer[index].done,
    #                 n_steps=n_steps
    #             )
    #         )
    #         # single-line implementation
    #         # n_step_reward = sum([gamma * self.buffer[index+i] if (not self.buffer[index].done) and (index + i < len(self.buffer)) else 0.0 for i in range(n_steps)]
    #     return batch


    

class NECReplayBuffer(ReplayBuffer):
    
    OUTPUT_TYPE = collections.namedtuple(
        'Transition',
        ['state', 'q_target', 'action'],
    )

    def append(self, state, q_target, action):
        self.buffer.append(self.output_type(state, q_target, action))


    
    