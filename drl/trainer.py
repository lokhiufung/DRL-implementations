import os

import torch
import gym

from drl.core.agent import ValueBasedAgent
from drl.core.transition import TransitionHistory
from drl.core.types import AgentType
from drl.utils import get_logger
from tensorboard_logger import TensorboardLogger


logger = get_logger(__name__, fh_lv='debug', ch_lv='info')


class Trainer:
    def __init__(self, experiment_name, env, experiment_logdir=None):
        self.experiment_name = experiment_name
        self.experiment_logdir = experiment_logdir
        self.writer = None
        if self.experiment_logdir is not None:
            if not os.path.exists(experiment_logdir):
                os.makedirs(experiment_logdir)
            self.tensorboard_logdir = os.path.join(self.experiment_logdir, 'tensorboard')
            self.writer = TensorboardLogger(
                logdir=self.tensorboard_logdir
            )
            
        if isinstance(env, str):
            self.env = gym.make(env)
        else:
            self.env = env

        # variables for training
        self.global_steps = 0

    def train(self, agent, n_episodes=100, is_render=False):
        n_step_reward = agent.n_step_reward
        history = TransitionHistory(n_transitions=n_step_reward)
        
        if self.writer is not None:
            try:
                self.env.reset()
                state, _, _, _ = self.env.step(self.env.action_space.sample()) # take a random action to start with
                self.writer.add_graph(agent.model, torch.tensor([state], dtype=torch.float32))  # add model graph to tensorboard
            except:
                logger.error('Writer cannot write computational graph.')

        if isinstance(agent, ValueBasedAgent):
            agent_type = AgentType.VALUE_BASED
        else:
            raise Exception('Currently only support value based agent.')
        
        logger.debug('Start training! n_episodes={} n_warmup_steps={}'.format(n_episodes, agent.n_warmup_steps))
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

                if agent_type == AgentType.VALUE_BASED:
                    action_tensor, value_tensor = agent.act(state_tensor)
                    value_predicted = agent.evaluate(state_tensor)
                    if self.writer is not None:
                        self.writer.log_scalar(
                            iteration=self.global_steps,
                            train_data={
                                'step/target_value': value_predicted.item(),
                                'step/policy_value': value_tensor.item(),
                            }
                        )

                if self.global_steps > agent.n_warmup_steps:
                    agent.call_after_warmup(self.global_steps, steps, self.writer)
                
                action = action_tensor.item()
                next_state, reward, done, _ = self.env.step(action)
                history.append(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )

                if len(history) == agent.n_step_reward:
                    agent.call_after_n_step_reward(
                        global_steps=self.global_steps, 
                        history=history,
                    )
                
                self.global_steps += 1
                steps += 1
                episode_scores += reward

                if self.global_steps % agent.learn_per_step == 0 and agent.is_ready_to_learn() and self.global_steps > agent.n_warmup_steps:
                    agent.backprop(global_steps=self.global_steps, writer=self.writer)

                agent.call_end_of_step(global_steps=self.global_steps, steps=steps, writer=self.writer)
            
                state = next_state
                if done:
                    break
            
            agent.call_end_of_episode(episode, writer=self.writer)
            if self.writer is not None:
                self.writer.log_scalar(
                    iteration=episode,
                    train_data={
                        'episode/episode_scores': episode_scores
                    }
                )
            logger.info('end of episode={} episode_scores={}'.format(episode+1, episode_scores))
        
        self.env.close()
