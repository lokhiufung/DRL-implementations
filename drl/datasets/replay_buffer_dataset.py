import torch
from torch.utils.data import Dataset, IterableDataset

from drl.blocks.memory.replay_buffer import ReplayBuffer


class LowDimReplayBufferDataset(IterableDataset):
    def __init__(self, replay_buffer: ReplayBuffer):
        super().__init__()
        self.replay_buffer = replay_buffer
    
    @property
    def output_port(self):
        return (
            ('state', ('B', 'C')),
            ('reward'), ('B', 'R'),
            ('action'), ('B', 'A'),
            ('next_state'), ('B', 'C'),
            ('done'), ('B', 'I'),
        )
    
    def __iter__(self):
        return iter(self.replay_buffer)

    def collate_fn(self, batch):
        states = [torch.tensor(sample.state, dtype=torch.float) for sample in batch]
        rewards = [torch.tensor(sample.rewards, dtype=torch.float) for sample in batch]
        actions = [torch.tensor(sample.actions, dtype=torch.long) for sample in batch]
        next_states = [torch.tensor(sample.next_state, dtype=torch.float) for sample in batch]
        dones = [torch.tensor(sample.done, dtype=torch.bool) for sample in batch] 
        return (
            torch.cat(states, dim=0),
            torch.cat(rewards, dim=0),
            torch.cat(actions, dim=0),
            torch.cat(next_states, dim=0),
            torch.cat(dones, dim=0),
        )

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
