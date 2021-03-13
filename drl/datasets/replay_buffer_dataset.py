import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from drl.blocks.memory.replay_buffer import ReplayBuffer


class LowDimReplayBufferDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int=16):
        super().__init__()
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    @property
    def output_port(self):
        return (
            ('states', ('B', 'C')),
            ('rewards'), ('B', 'R'),
            ('actions'), ('B', 'A'),
            ('next_states'), ('B', 'C'),
            ('dones'), ('B', 'I'),
        )
    
    def __iter__(self):
        return self.replay_buffer.get_batch(batch_size=self.batch_size)

    # temp
    # def __len__(self):
    #     return self.replay_buffer.capacity

    def collate_fn(self, batch):
        batch = batch[0]  # REMINDME: get_batch() returns a list of sampled Trainsition objects

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
