import os
from itertools import count

import torch
import gym

import agents
from env_helper import Transition
from utils import get_logger


class Trainer(object):
    def __init__(
        self, agent_name, env_name, agent_config, 
        output_dir, logger=None, fh_lv='error', ch_lv='debug'
        ):
        self.agent = getattr(agents, agent_name)(**agent_config)
        self.env = gym.make(env_name)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir): 
            os.makedirs(self.output_dir)
        self._transition = None
        self._num_traceback = agent_config.get('n_steps', 1) 
        
        if logger is None:
            self.logger = get_logger('trainer', fh_lv=fh_lv, ch_lv=ch_lv)

    def train(self, num_warmup_episode, num_train_episode, render=False):
        # warmup phase
        self.run_warmup_episodes(num_warmup_episode, render=False)
        # save dnd
        self.agent.save(output_dir)
        # training phase
        self.run_train_episodes(num_train_episode, render=render)
        
    def run_train_episodes(self, num_episodes, render=False):
        transition = Transition(num_traceback=self._num_traceback)
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                if render:
                    self.env.render()
                action = self.agent.epsilon_greedy_infer(state, return_tensor=False)
                next_state, reward, done, info = self.env.step(action)
                transition.append(state, reward, action, next_state)
                self.logger.debug('[train] episode[{}] step[{}] state: {} action: {} reward: {} next_state: {}'.format(episode, step, state, action, reward, next_state))
                state = next_state
                # save to memeory
                if transition.is_ready:
                    # get training samples from transotion object
                    start_state, action, summed_reward, end_state = transition.get_aggregated()

                    encoded_state = self.agent.encode_state(start_state, return_tensor=True)
                    agent.remember_to_replay_buffer(state, action, summed_reward)
                    agent.remember_to_dnd(state, action, summed_reward)        

                # learning
                agent.replay(batch_size)
                if done:
                    self.logger.debug('[train] episode[{}] done at step {}'.format(episode, step))
                    break

    def run_warmup_episodes(self, num_episodes, render=False):
        transition = Transition(num_traceback=self._num_traceback)
        for episode in range(num_episodes):
            state = self.env.reset()
            # while True:
            for step in count():
                if render:
                    self.env.render()
                action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                transition.append(state, reward, action, next_state)
                self.logger.debug('[warmup] episode[{}] step[{}] state: {} action: {} reward: {} next_state: {}'.format(episode, step, state, action, reward, next_state))
                state = next_state
                # save to memeory
                if transition.is_ready:
                    # get training samples from transotion object
                    start_state, action, summed_reward, end_state = transition.get_aggregated()
                    
                    encoded_state = self.agent.encode_state(start_state, return_tensor=True)
                    self.agent.remember_to_replay_buffer(start_state, action, summed_reward)
                    self.agent.remember_to_dnd(encoded_state, action, summed_reward)        

                if done:    
                    self.logger.debug('[warmup] episode[{}] done at step {}'.format(episode, step))
                    break

if __name__ == '__main__':
    from utils import load_json

    experiment_name = 'nec-01'
    HYPARAMS = load_json('./hyparams/nec_hyparams.json')[experiment_name]['hyparams']
    agent_config= {
        'input_dim': 4,
        'encode_dim': 32,
        'hidden_dim': 64,
        'output_dim': 2,
        'capacity': HYPARAMS['capacity'],
        'buffer_size': HYPARAMS['buffer_size'],
        'epsilon_start': HYPARAMS['epsilon_start'],
        'epsilon_end': HYPARAMS['epsilon_end'],
        'decay_factor': HYPARAMS['decay_factor'],
        'lr': HYPARAMS['lr'],
        'p': HYPARAMS['p'],
        'similarity_threshold': HYPARAMS['similarity_threshold'],
        'alpha': HYPARAMS['alpha']
    }
    trainer = Trainer(
        agent_name='NECAgent',
        env_name='CartPole-v0',
        agent_config=agent_config,
        output_dir=f'./experiments/{experiment_name}'
    )
    # trainer.run_warmup_episodes(1000, render=False)
    # trainer.agent.save_dnd(
    #     output_dir='./experiments/nec-01'
    # )
    trainer.agent.load_dnd('./experiments/nec-01')
    trainer.run_train_episodes(10)

    # print(len(trainer.agent.dnd_list[0]))
    # print(len(trainer.agent.dnd_list[1]))
