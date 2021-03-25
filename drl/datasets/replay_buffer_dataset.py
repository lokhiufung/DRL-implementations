import random

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from drl.blocks.memory.replay_buffer import ReplayBuffer


class LowDimReplayBufferDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int=16, batches_per_epoch: int=100):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    @property
    def output_port(self):
        return (
            ('states', ('B', 'C')),
            ('rewards'), ('B', 'R'),
            ('actions'), ('B', 'A'),
            ('next_states'), ('B', 'C'),
            ('dones'), ('B', 'I'),
        )
    
    def __len__(self):
        return self.batches_per_epoch
        
    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            yield self.replay_buffer.get_batch(batch_size=self.batch_size)


    def collate_fn(self, batch):
        batch = batch[0]  # REMINDME: get_batch() returns a list of sampled Trainsition objects
        # print(batch)
        states = np.stack([sample.state for sample in batch])
        rewards = [sample.reward for sample in batch]
        actions = [sample.action for sample in batch]
        next_states = np.stack([sample.next_state for sample in batch])
        dones = [sample.done for sample in batch] 
        
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)
        return states, actions, rewards, next_states, dones
            

class HighDimReplayBufferDataset(LowDimReplayBufferDataset):
    @property
    def output_port(self):
        return (
            ('state', ('B', 'H', 'W', 'C')),
            ('reward'), ('B', 'R'),
            ('action'), ('B', 'A'),
            ('next_state'), ('B', 'H', 'W', 'C'),
            ('done'), ('B', 'I'),
        )


class NStepsLowDimReplayBufferDataset(LowDimReplayBufferDataset):
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int=16, batches_per_epoch: int=100, horizon: int=100, gamma: float=0.999):
        super().__init__(replay_buffer, batch_size, batches_per_epoch)
        self.horizon = horizon
        self.gamma = gamma

    def get_discount_n_step_reward(self, transitions):
        n_step_reward = 0.0
        for i, transition in enumerate(transitions):
            n_step_reward += (self.gamma**i) * transition.reward
            if transition.done:
                break
        return n_step_reward

    def __iter__(self):
        transitions = list(self.replay_buffer.buffer)  # reminder: cannot slice a deque
        size = len(transitions)
        n_steps_transitions = []
        for i in range(0, size, self.horizon):
            # print(type(transitions))
            n_steps_transition = self.replay_buffer.output_type(
                state=transitions[i].state,
                reward=self.get_discount_n_step_reward(transitions[i:min(i+self.horizon, size)]),
                next_state=transitions[min(i+self.horizon, size-1)].next_state,  # reminder: exclusive upper bound
                action=transitions[i].action,
                done=transitions[i].done
            )
            n_steps_transitions.append(n_steps_transition)

        random.shuffle(n_steps_transitions)
        
        for i in range(self.batches_per_epoch):
            yield n_steps_transitions[i:min(i+self.batch_size, len(n_steps_transitions))]


