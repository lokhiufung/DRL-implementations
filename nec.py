import os
import argparse
import random
import math
from itertools import count
from collections import deque

# from scipy.spatial import KDTree
import numpy as np
import gym
import torch 
from torch import nn
import torch.nn.functional as F
from torch.optim import RMSprop

from tensorboard_logger import TensorboardLogger
from utils import load_json, get_logger

logger = get_logger('nec', fh_lv='debug', ch_lv='debug')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str)
    args = parser.parse_args()
    experiment_name = args.name

    HYPARAMS = load_json('./hyparams/nec_hyparams.json')[experiment_name]['hyparams']
    logger.debug('experiment_name: {} hyparams: {}'.format(experiment_name, HYPARAMS))
    # make checkpoint path
    experiment_logdir = 'experiments/{}'.format(experiment_name)
    if not os.path.exists(experiment_logdir):
        os.makedirs(experiment_logdir)

    # write to tensorboard 
    tensorboard_logdir = '{}/tensorboard'.format(experiment_logdir)
    if not os.path.exists(tensorboard_logdir):
        os.mkdir(tensorboard_logdir)
    writer = TensorboardLogger(logdir=tensorboard_logdir)

    env = gym.make('CartPole-v0')
    agent = NECAgent(
        input_dim=env.observation_space.shape[0],
        encode_dim=32,
        hidden_dim=64,
        output_dim=env.action_space.n,
        capacity=HYPARAMS['capacity'],
        buffer_size=HYPARAMS['buffer_size'],
        epsilon_start=HYPARAMS['epsilon_start'],
        epsilon_end=HYPARAMS['epsilon_end'],
        decay_factor=HYPARAMS['decay_factor'],
        lr=HYPARAMS['lr'],
        p=HYPARAMS['p'],
        similarity_threshold=HYPARAMS['similarity_threshold'],
        alpha=HYPARAMS['alpha']
    )
    global_steps = 0
    for episode in range(HYPARAMS['episodes']):
        state = env.reset()
        counter = 0
        while True:
            n_steps_q = 0
            start_state = state
            # N-steps Q estimate
            for step in range(HYPARAMS['horizon']):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                action_tensor, value_tensor, encoded_state_tensor = agent.epsilon_greedy_infer(state_tensor)
                if step == 0:
                    start_action = action_tensor.item()
                    start_encoded_state = encoded_state_tensor
                # env.render()
                if global_steps > HYPARAMS['warmup_steps']:
                    action = action_tensor.item()
                    agent.epsilon_decay()
                else:
                    action = env.action_space.sample()
                logger.debug('episode: {} global_steps: {} value: {} action: {} state: {} epsilon: {}'.format(episode, global_steps, value_tensor.item(), action, state, agent.epsilon))
                next_state, reward, done, info = env.step(action)
                counter += 1
                global_steps += 1
                writer.log_training_v2(global_steps, {
                    'train/value': value_tensor.item(),
                })
                n_steps_q += (HYPARAMS['gamma']**step) * reward
                if done:
                    break
                state = next_state
            n_steps_q += (HYPARAMS['gamma']**HYPARAMS['horizon']) * agent.get_target_n_steps_q().item() 
            writer.log_training_v2(global_steps, {
                'sampled/n_steps_q': n_steps_q,
            })
            logger.debug('sample n_steps_q: {}'.format(n_steps_q))
            # append to ReplayBuffer and DND
            agent.remember_to_replay_buffer(start_state, start_action, n_steps_q)
            agent.remember_to_dnd(start_encoded_state, start_action, n_steps_q)

            if global_steps / HYPARAMS['horizon'] > HYPARAMS['batch_size']:
                agent.replay(batch_size=HYPARAMS['batch_size'])
            if done:
                # update dnd
                writer.log_episode(episode + 1, counter)
                logger.info('episode done! episode: {} score: {}'.format(episode, counter))
                logger.debug('dnd[0] len: {}'.format(len(agent.dnd_list[0])))
                logger.debug('dnd[1] len: {}'.format(len(agent.dnd_list[1])))
                break









if __name__ == '__main__':
    main()