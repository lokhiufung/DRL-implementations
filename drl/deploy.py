import math
import argparse

import torch
import gym

from drl.core.transition import TransitionHistory
from drl.agents import NAME_TO_AGENT
from drl.core.types import AgentType
from drl.core.reward_scaler import RewardScaler
from drl.utils import get_logger, load_json


logger = get_logger(__name__, fh_lv='debug', ch_lv='info')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', required=True, type=str, help='name of experiment')
    parser.add_argument('--env', required=False, type=str, default='CartPole-v0', help='name of experiment')
    parser.add_argument('--render', action='store_true', help='render gym')
    parser.add_argument('--agent', required=True, type=str, help='name of agent')
    parser.add_argument('--ckpt', required=True, type=str, help='checkpoint filepath')
    args = parser.parse_args()
    return args


def run_drl():
    args = parse_args()

    experiment_name = args.name
    is_render = args.render
    env_id = args.env
    ckpt = args.ckpt

    hyparams = load_json('./hyparams/dqn_hyparams.json')[experiment_name]['hyparams']
    n_episodes = hyparams.get('n_episodes', 100)
    del hyparams['n_episodes']  # this is for env runner only

    env = gym.make(env_id)
    agent = NAME_TO_AGENT[args.agent](
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        **hyparams
    )
    agent.load_checkpoint(ckpt)
    
    reward_scaler = None  # TODO: add reward scaler later
    deploy = Deploy(agent, env, reward_scaler)

    deploy.run(
        n_episodes=n_episodes,
        is_render=is_render
    )


class Deploy:
    def __init__(self, agent, env, reward_scaler: RewardScaler=None):
        self.agent = agent
        self.env = env
        self.reward_scaler = reward_scaler

        if reward_scaler is not None:
            self.reward_scaler = reward_scaler
        else:
            from drl.reward_scalers.no_scaler import NoScaler
            self.reward_scaler = NoScaler()

        self.global_steps = 0

    def run(self, n_episodes=100, is_render=False):
        n_step_reward = self.agent.n_step_reward
        history = TransitionHistory(n_transitions=n_step_reward)
        
        if self.agent.agent_type != AgentType.VALUE_BASED:
            raise Exception('Currently only support value based agent.')
        
        highest_score = -math.inf
        logger.debug('Start training! n_episodes={} n_warmup_steps={}'.format(n_episodes, self.agent.n_warmup_steps))
        for episode in range(n_episodes):
            state = self.env.reset()
            history.reset()
            steps = 0
            episode_scores = 0
            logger.debug('start of episode={} length of history={} steps={}'.format(episode+1, len(history), steps))
            while True:
                if is_render:
                    self.env.render()

                state_tensor = torch.from_numpy(state).float().unsqueeze(0)

                if self.agent.agent_type:
                    action_tensor, value_tensor = self.agent.act(state_tensor)
                    value_predicted = self.agent.evaluate(state_tensor)
                
                action = action_tensor.item()
                next_state, reward, done, _ = self.env.step(action)
                
                history.append(
                    state=state,
                    action=action,
                    reward=self.reward_scaler.scale(reward),  # scaling the reward received
                    next_state=next_state,
                    done=done
                )

                self.global_steps += 1
                steps += 1
                episode_scores += reward

                state = next_state
                if done:
                    break
                        
            logger.info('end of episode={} episode_scores={}'.format(episode+1, episode_scores))
        
        self.env.close()



