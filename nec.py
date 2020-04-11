import argparse
import random
from itertools import count

import gym
import torch 
from torch import nn
import torch.functional as F

from utils import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str)
    args = parser.parse_args()
    experiment_name = args.name

    HYPARAMS = load_json('./hyparams/nec_hyparams.json')[experiment_name]['hyparams']

    env = gym.make('CartPole-v0')
    agent = NECAgent(
        input_dim=env.observation_space.shape[0],
        encode_dim=32,
        hidden_dim=64,
        output_dim=env.action_space.n,
        capacity=HYPARAMS['capacity'],
        buffer_size=HYPARAMS['buffer_size']
    )

    for episode in range(HYPARAMS['episodes']):
        state = env.reset()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        while True:
            n_steps_q = 0
            start_state = state 
            # N-steps Q estimate
            for step in range(HYPARAMS['horizon']):
                action_tensor, value_tensor, encoded_state_tensor = agent.epsilon_greedy_infer(state_tensor)
                if step == 0:
                    start_action = action_tensor.item()
                    start_encoded_state = encoded_state_tensor.numpy()
                next_state, reward, done, info = env.step(action_tensor.item())
                n_steps_q += (HYPARAMS['gamma']**step) * reward
                if done:
                    break
                state = next_state
            # append to ReplayBuffer and DND
            agent.remember_to_replay_buffer(start_state, start_action)
            agent.remember_to_dnd(encoded_state, n_steps_q)

            agent.replay(batch_size=HYPARAMS['batch_size'])
            if done:
                # update dnd, 
                break

class Encoder(nn.Module):
    def __init__(self, input_dim, encode_dim, hidden_dim=64):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        self.hidden_dim = 64

        # encoder 
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.encode_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)  # once maxlen is rearched, the left sample will be poped out

    def append(self, state, action, n_steps_q):
        self.buffer.append((state, action, n_steps_q))

    def get_batch(self, batch_size):
        # random.sample() for sampling without replacement
        return random.sample(self.buffer, k=batch_size)

    def __len__(self):
        return len(self.buffer)


class DND(object):
    def __init__(self, encode_dim, capacity):
        self.encode_dim = encode_dim
        self.capacity = capacity


class NECAgent(object):
    def __init__(self, input_dim, encode_dim, hidden_dim, output_dim, capacity, buffer_size):
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.capacity = capacity
        self.buffer_size = buffer_size
        
        self.encoder = Encoder(self.input_dim, self.encode_dim, self.hidden_dim)
        # one dnd one one action; query by index of a list
        self.dnd_list = [DND(self.encode_dim, self.capacity) for _ in range(self.output_dim)]
        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)

    def greedy_infer(self, state):
        pass

    def epsilon_greedy_infer(self, state):
        pass

    def replay(self, batch_size):
        pass

    def remember_to_replay_buffer(self, state, action, n_steps_q):
        self.replay_buffer.append(state, action, n_steps_q)

    def remember_to_dnd(self, encoded_state, n_steps_q):
        pass