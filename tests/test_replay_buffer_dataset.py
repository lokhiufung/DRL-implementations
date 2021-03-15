import gym
from torch.utils.data import DataLoader

from drl.blocks.memory.replay_buffer import ReplayBuffer
from drl.datasets.replay_buffer_dataset import LowDimReplayBufferDataset


ENV = gym.make('CartPole-v0')

def test_low_dim_replay_buffer_dataset_batch():
    replay_buffer = ReplayBuffer()
    
    state = ENV.reset()
    for _ in range(10):
        action = ENV.action_space.sample()
        next_state, reward, done, _ = ENV.step(action)
        if done:
            state = ENV.reset()
        else:
            state = next_state
        replay_buffer.append(state, reward, next_state, action, done)

    dataset = LowDimReplayBufferDataset(
        replay_buffer,
        batch_size=4
    )
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn)

    batch = next(iter(dataloader))
    assert len(batch) == 5


def test_low_dim_replay_buffer_dataset_dim():
    replay_buffer = ReplayBuffer()
    
    state = ENV.reset()
    for _ in range(10):
        action = ENV.action_space.sample()
        next_state, reward, done, _ = ENV.step(action)
        if done:
            state = ENV.reset()
        else:
            state = next_state
        replay_buffer.append(state, reward, next_state, action, done)

    dataset = LowDimReplayBufferDataset(
        replay_buffer,
        batch_size=4
    )
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn)
    
    batch = next(iter(dataloader))
    print(batch)
    assert batch[0].size() == (4, 4)
    assert batch[1].size() == (4,)
    assert batch[2].size() == (4,)
    assert batch[3].size() == (4, 4)
    assert batch[4].size() == (4,)

test_low_dim_replay_buffer_dataset_batch()