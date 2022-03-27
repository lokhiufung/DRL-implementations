import os
import argparse
import math
import random
from collections import deque

# from scipy.spatial import KDTree
import numpy as np
import gym
import torch 
from torch import nn
import torch.nn.functional as F
from torch.optim import RMSprop

from drl.blocks.memory.dnd import DifferentiableNeuralDictionary
from drl.blocks.memory.replay_buffer import NECReplayBuffer
from tensorboard_logger import TensorboardLogger
from utils import load_json, get_logger


logger = get_logger('nec', fh_lv='debug', ch_lv='debug')


class NECNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, embedding_dim=32, p=50, similarity_threshold=0.5, alpha=1):
        super().__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.dnd = DifferentiableNeuralDictionary(
            n_actions=n_actions,
            dim=self.embedding_dim,  # dim of embedding vector
            n_neighbors=p,
            score_threshold=similarity_threshold,
            alpha=alpha,
        )
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x, return_all_values=False):
        """_summary_

        Args:
            x (_type_): state tensor
        Returns:
            _type_: _description_
        """
        keys = self.encoder(x)
        values, actions, indexes, scores = self.dnd(keys, return_all_values=return_all_values)
        return values, actions, indexes, scores, keys

    # def commit(self, actions, keys, values, scores):
    #     return self.dnd.update_to_buffer(actions, keys, values, scores)        
    
    def get_max_q(self):
        max_q = self.dnd.get_max_value()
        return max_q

    @torch.no_grad()
    def predict(self, x: torch.TensorType):
        """predict the values and actions from a state

        Args:
            x (torch.TensorType): state

        Returns:
            _type_: values, actions
        """
        max_values, actions, indexes, scores, keys = self.forward(x)
        return max_values, actions, indexes, scores, keys

    @torch.no_grad()
    def encode(self, x: torch.TensorType):
        """get the encoded state for inference

        Args:
            x (torch.TensorType): state tensor

        Returns:
            _type_: encoded state tensor
        """
        x = self.encoder(x)
        return x



class NECAgent:
    def __init__(
        self,
        input_dim,
        output_dim,
        embedding_dim, 
        lr, 
        capacity=5*10**5, 
        buffer_size=10**5,
        epsilon_start=0.9,
        epsilon_end=0.05,
        decay_factor=100, 
        p=50, 
        similarity_threshold=0.5, 
        alpha=1, 
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.capacity = capacity

        self.lr = lr
        self.model = NECNetwork(
            input_dim=self.input_dim,
            n_actions=self.output_dim,
            embedding_dim=self.embedding_dim,
            p=p,
            similarity_threshold=similarity_threshold,
            alpha=alpha,
        )
        self.replay_buffer = NECReplayBuffer(capacity=buffer_size)
        self.optimizer = RMSprop(self.model.parameters(), lr=self.lr)  # TODO: consider adding parameters dynamically

        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.decay_factor = decay_factor

        self.cache_for_dnd = None

        self._reset_cache_for_dnd()  # initialize the self.cache_for_dnd here

    def _reset_cache_for_dnd(self):
        self.cache_for_dnd = {
            'actions': [],
            'keys': [],
            'indexes': [],
            'scores': [],
            'values': [],
        }

    def commit_single_to_dnd(self, action, key, value, index=None, score=None):
        """commit a single record to agent's dnd buffer
        """
        self.cache_for_dnd['actions'].append(action)
        self.cache_for_dnd['keys'].append(key)
        self.cache_for_dnd['values'].append(value)
        
        if index is not None or score is not None:
            assert index is not None and score is not None, 'Both index and socre must be provided.'

            self.cache_for_dnd['indexes'].append(index)
            self.cache_for_dnd['scores'].append(score)
    
    def push_to_dnd(self):
        if not self.model.dnd.is_ready():
            # directly write to the dnd buffer if the dnd is not ready
            # TODO: write_to_buffer method should handle batch writing
            for i in range(len(self.cache_for_dnd['actions'])):
                self.model.dnd.write_to_buffer(
                    action=self.cache_for_dnd['actions'][i],
                    key=self.cache_for_dnd['keys'][i],
                    value=self.cache_for_dnd['values'][i],
                )
        else:
            # if the dnd is ready, update the buffer with indexes, scores
            self.model.dnd.update_to_buffer(
                actions=self.cache_for_dnd['actions'],
                keys=self.cache_for_dnd['keys'],
                indexes=self.cache_for_dnd['indexes'],
                scores=self.cache_for_dnd['scores'],
                values=self.cache_for_dnd['values']
            )
        # if success, reset the cache
        self._reset_cache_for_dnd()

    def write_dnd(self):
        self.model.dnd.write()

    def get_max_q(self):
        if self.model.dnd.is_ready():
            max_q = self.model.get_max_q()
            max_q = max_q.item()
            return max_q
        else:
            return 0.0  # reminder: if the dnd is not ready yet, just use 0.0 as the best value estimate

    def replay(self, batch_size):
        if self.model.dnd.is_ready():
            # TODO: check the replay logic
            batch = self.replay_buffer.get_batch(batch_size)

            # reminder: make sure to cast to float before creating the computation graph
            states = torch.tensor([batch[i].state for i in range(batch_size)]).float()
            actions = torch.tensor([batch[i].action for i in range(batch_size)]).long()
            q_targets = torch.tensor([batch[i].q_target for i in range(batch_size)]).float()

            qs, _, _, _, _ = self.model(states, return_all_values=True)
            
            actions = actions.unsqueeze(-1)
            qs = torch.gather(qs, 1, actions).squeeze()
            # print('qs.shape: ', qs.size())
            # TODO: dims of qs and q_targets do not match 
            # print('q_targets.shape: ', q_targets.size())
            loss = F.mse_loss(qs, q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            for param in self.model.parameters():
                if param.grad is not None:  # some parameters in dnd do not have grad, since they are retrieved, TODO: verify this hypothesis
                    param.grad.data.clamp_(-1, 1)  # |grad| <= 1
            self.optimizer.step()
        else:
            logger.debug('dnd is not ready yet. Do not replay now.')

    def encode(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        h = self.model.encode(state)
        return h
        
    def greedy_infer(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        # inference = self.try_greedy_infer(state)
        max_values, actions, indexes, scores, keys = self.model.predict(state)  # TODO: make the ordering of action, value consistent
        extra = {'scores': scores, 'indexes': indexes, 'keys': keys}
        # if inference is not None:
        #     actions, max_values, extra = inference
        return actions, max_values, extra  # return scale values of action and value. extra term is a dict containing additional values for reference

    def epsilon_greedy_infer(self, state):
        random_actions = torch.tensor([[random.randrange(self.output_dim)]], dtype=torch.long)
        actions, values, extra = self.greedy_infer(state)
        random_number = random.random()

        if random_number < self.epsilon:
            actions = random_actions

        return actions, values, extra

    def epsilon_decay(self):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps_done / self.decay_factor)

    def remember(self, state, q_target, action):
        self.replay_buffer.append(state, q_target, action)
    
    def save_checkpoint(self, output_dir):
        """
        save checkpoint for restore training
        """
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'dnd_state_dict': self.model.dnd.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(output_dir, 'checkpoint_{}'.format(self.steps_done)))

    def save_network(self, output_dir):
        """
        save checkpoint for restore training
        """
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'dnd_state_dict': self.model.dnd.state_dict(),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str)
    parser.add_argument('--render', action='store_true', help='render gym')

    args = parser.parse_args()
    experiment_name = args.name
    is_render = args.render

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
        output_dim=env.action_space.n,
        embedding_dim=32,
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

    ###################
    # initialize a gamma decay vector for calculate n-step reward
    gamma_vector = np.array([HYPARAMS['gamma']**i for i in range(HYPARAMS['n_step_reward'])])
    ###################
    

    for episode in range(HYPARAMS['episodes']):
        state = env.reset()
        step = 0
        rewards = deque(maxlen=HYPARAMS['n_step_reward'])
        while True:
            if is_render:
                env.render()
            # if agent.model.dnd.is_ready():  # TMEP: add a new method for agent to get the status of dnd
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            # TEMP: fucking ugly
            indexes = None
            scores = None

            if not agent.model.dnd.is_ready():
                keys = agent.encode(state)
                action_tensor = torch.tensor([[env.action_space.sample()]], dtype=torch.long)
                value_tensor = torch.tensor([[0.0]], dtype=torch.float)
            else:
                action_tensor, value_tensor, extra = agent.epsilon_greedy_infer(state_tensor)
                keys = extra['keys']
                indexes = extra['indexes'][0]
                scores = extra['scores'][0]

            if global_steps > HYPARAMS['warmup_steps']:
                # if training starts, use the action from the agent and decay the epsilon
                agent.epsilon_decay()

            action = action_tensor.item()
            logger.debug('episode: {} global_steps: {} value: {} action: {} state: {} epsilon: {}'.format(episode, global_steps, value_tensor.item(), action, state, agent.epsilon))
            # else:
                # otherwise, just use a random action
            # action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            
            ###################
            # cache rewards for getting n-step q 
            rewards.append(reward)
            if len(rewards) == HYPARAMS['n_step_reward']:
                if done:
                    q_target = np.dot(np.array(rewards), gamma_vector)
                else:
                    max_q = agent.get_max_q()  # use the maximum q to bootstrap the term max Q(s_t+N, a')
                    assert isinstance(max_q, float)
                    q_target = np.dot(np.array(rewards), gamma_vector) + HYPARAMS['gamma']**HYPARAMS['n_step_reward'] * max_q 
                # TODO: should use a state n step before
                agent.remember(state, q_target, action)  # save to replay buffer
                # reminder: q_target will plays the role of the target network in DQN

                agent.commit_single_to_dnd(
                    action=action,
                    key=keys,
                    value=torch.tensor(q_target, dtype=torch.float),  # reminder: store the Q_TARGET for value estimate
                    index=indexes[0] if indexes is not None else indexes,
                    score=scores[0] if scores is not None else scores,
                )
            ###################


            global_steps += 1
            step += 1

            # writer.log_training_v2(global_steps, {
            #     'train/value': value_tensor.item(),
            # })
            if done:
                break
            state = next_state

            # TODO: add logic for experience replay
            logger.debug('episode: {} global_steps: {} length_of_replay_buffer: {} dnd.is_ready: {}'.format(episode, global_steps, len(agent.replay_buffer), agent.model.dnd.is_ready()))
            if global_steps > HYPARAMS['warmup_steps'] and len(agent.replay_buffer) >= HYPARAMS['batch_size']:
                print('global_steps: {} | experience replay!'.format(global_steps))
                agent.replay(batch_size=HYPARAMS['batch_size'])
            
        # At the end of each episode
        logger.debug('cache_for_dnd: {}'.format(len(agent.cache_for_dnd['actions'])))
        if len(agent.cache_for_dnd['actions']) > 0:
            agent.push_to_dnd()  # push the updates to dnd
            agent.write_dnd()  # write to search_engine
            print(f'end of episode {episode}')

if __name__ == '__main__':
    main()