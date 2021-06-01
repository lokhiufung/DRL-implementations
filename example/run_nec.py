import time

import gym
import torch
import torch.nn as nn
import numpy as np

from drl.core.transition import TransitionBuffer
from drl.blocks.encoder.dense_encoder import DenseEncoder
from drl.blocks.memory.replay_buffer import ReplayBuffer
from drl.blocks.memory.dnd import DifferentiableNeuralDictionary


class NEC(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = DenseEncoder(n_layers=2, input_dim=4)
        self.dnd = DifferentiableNeuralDictionary(
            n_actions=2,
            dim=128,
        )
        self.replay_buffer = ReplayBuffer()

    def forward(self, x):
        key = self.encoder(x)
        values, action, indexes, scores = self.dnd(x, return_all_values=True)
        return key, values, action, indexes, scores

    @torch.no_grad()
    def play_step(self, state):
        key = self.encoder(state)
        values, action, indexes, scores = self.dnd.lookup(key, return_all_values=False, return_tensor=True) 
        return key.cpu().numpy(), values, action, indexes, scores  
    
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append(state, action, reward, next_state, done)


def main():
    n_steps = 2
    grad_steps = 1000
    gamma = 0.99
    transition_buffer = TransitionBuffer()
    env = gym.make('CartPole-v0')
    
    agent = NEC()

    done = False

    for episode in range(30):
        state = env.reset()
        episode_reward = 0.0
        done = False

        start = time.perf_counter()
        # collect transitions
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            transition_buffer.write_to_buffer(state, reward, next_state, action, done)

            state = next_state

        # print('length of transition_buffer: {}'.format(len(transition_buffer)))
        
        # update dnd and replay_buffer
        q_estimates = []
        for i in range(len(transition_buffer) - n_steps):
                
            reward = sum([gamma**step * transition_buffer[i+step].reward for step in range(n_steps)])

            try:
                state = torch.from_numpy(transition_buffer[i+n_steps].state.astype(np.float32))
                state = state.unsqueeze(dim=0)
                key, q_bootstrap, _, indexes, scores = agent.play_step(state)
                q_bootstrap = q_bootstrap.item() 
                # print('agent act')
                # print('q_bootstrap: ', q_bootstrap)
            except AttributeError:
                q_bootstrap = 0.0

            q_estimate = reward + gamma ** n_steps * q_bootstrap

            agent.replay_buffer.append(state, q_estimate, transition_buffer[i].next_state, transition_buffer[i].action, transition_buffer[i].done)

            key = agent.encoder(torch.from_numpy(transition_buffer[i].state.astype(np.float32)))
            # print('key shape: ', key.size())
            q_estimate = torch.tensor(q_estimate, dtype=torch.float)
            # agent.dnd.write_to_buffer(transition_buffer[i].action, key, q_estimate)
            agent.dnd.update_to_buffer()
            q_estimates.append(q_estimate)
            # print(f'[episode {episode}] q_estimate: {q_estimate}')

        # print(len(agent.dnd.dnds[0].key_buffer))
        # print(len(agent.dnd.dnds[1].key_buffer))

        # agent.dnd.write()
        total_time = time.perf_counter() - start

        # # learning
        # for _ in range(grad_steps):
        #     agent.replay()
        print('*********************************')
        print(f'end of episode {episode} | episode time: {total_time}')
        print(f'[episode {episode}] episode_reward: {episode_reward}')
        print('max q_estimate: {}'.format(max(q_estimates)))
        print('*********************************')



main()
                
            