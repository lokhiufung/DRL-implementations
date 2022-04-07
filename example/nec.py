import argparse

import gym

from drl.trainer import Trainer
from drl.agents.nec_agent import NECAgent
from drl.utils import load_json


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', required=True, type=str, help='name of experiment')
    parser.add_argument('--env', required=False, type=str, default='CartPole-v0', help='name of experiment')
    parser.add_argument('--render', action='store_true', help='render gym')
    args = parser.parse_args()
    return args


def main():
    args = parser_args()

    experiment_name = args.name
    is_render = args.render
    env_id = args.env

    hyparams = load_json('./hyparams/nec_hyparams.json')[experiment_name]['hyparams']
    
    env = gym.make(env_id)
    agent = NECAgent(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        lr=hyparams['lr'],
        gamma=hyparams['gamma'],
        epsilon_start=hyparams['epsilon_start'],
        epsilon_end=hyparams['epsilon_end'],
        decay_factor=hyparams['decay_factor'],
        buffer_size=hyparams['buffer_size'],
        batch_size=hyparams['batch_size'],
        n_step_reward=hyparams['n_step_reward'],
        learn_per_step=hyparams['learn_per_step'],
        n_warmup_steps=hyparams['n_warmup_steps'],
        embedding_dim=hyparams['embedding_dim'],
        p=hyparams['p'],
        similarity_threshold=hyparams['similarity_threshold'],
        alpha=hyparams['alpha'],
        mode='train',
        writer=None,
    )
    trainer = Trainer(
        experiment_name=experiment_name,
        experiment_logdir='experiments/{}'.format(experiment_name), 
        env=env,
    )

    trainer.train(
        agent,
        n_episodes=hyparams['n_episodes'],
        is_render=is_render,
    )



if __name__ == '__main__':
    main()