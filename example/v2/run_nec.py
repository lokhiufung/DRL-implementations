import torch
import torch.nn as nn
import gym

from drl.blocks.memory.dnd import DifferentiableNeuralDictionary
from drl.blocks.memory.replay_buffer import ReplayBuffer
from drl.blocks.encoder.dense_encoder import DenseEncoder
from drl.blocks.heads.value_head import NoOpValueHead


class NECModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = DenseEncoder(2, 4)
        self.dnd = DifferentiableNeuralDictionary(
            n_actions=3,
            dim=128,
        )
        self.output_head = NoOpValueHead()
    
    def forward(self, x):
        x = self.encoder(x)
        values, actions = self.dnd(x, return_all_values=True)
        values = self.output_head(values)
        return values, actions

    def play_step(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            values, actions = self.dnd(x, return_all_values=False)
            values = self.output_head(values)
        return values, actions


def main():
    env = gym.make('CarPole-v2')

    model = NECModel()

    state = env.reset()

    for episode in range(1000):
        done = False
        while not done:
            state, reward, done, _ = env.step(model.play_step(state))
        
            