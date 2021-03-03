import random
import collections

# from drl.core.modules import NonTranableModule

import torch


class ReplayBuffer:
    
    OUTPUT_TYPE = namedtuple(
        'Transition',
        ['state', 'reward', 'next_state', 'action', 'done'],
    )
    OUTPUT_DTYPE = {
        'state': torch.float32,
        'reward': torch.float32,
        'state_next': torch.float32,
        'action': torch.long,
        'done': torch.bool
    }

    def __init__(self, capacity=1e6):
        self.capacity = capacity
        self.output_type = OUTPUT_TYPE
        self.output_dtype = OUTPUT_DTYPE
        self.buffer = deque(maxlen=self.capactity)  # once maxlen is rearched, the left sample will be poped out

    def append(self, states=None, reward=None, next_state=None, action=None, done=None):
        self.buffer.append(self.output_type(state, reward, next_state, action, done))

    def get_batch(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)

        return self._collate_fn(samples=samples)

    def _collate_fn(self, batch):
        output_tensors = []
        for field, dtype in self.output_dtype.items():
            output_tensors.append(torch.tensor([
                getattr(sample, field) for sample in batch, dtype=dtype
            ]))
        return output_tensors

    def __len__(self):
        return len(self.buffer)

