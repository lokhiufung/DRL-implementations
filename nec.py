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
    """
    differentiable neural dictionary; should  be differentiable
    """
    def __init__(self, encode_dim, capacity, p=50, similarity_threshold=238.0, alpha=0.5):
        self.encode_dim = encode_dim
        self.capacity = capacity
        self.p = p  # num of knn
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha

        self.similarity_avg = 0.0 
        self.keys = []
        self.values = []

    def __len__(self):
        return len(self.keys)

    def append(self, encoded_state, n_steps_q):
        if not self.exist_state(encoded_state): 
            # append when state exist
            logger.debug('add new state')
            self.keys.append(encoded_state)
            self.values.append(n_steps_q)
        else:
            # update when state not exist 
            logger.debug('state already existed')
            top_k_weights, top_k_index = self.query_k(encoded_state, k=1)
            self.values[top_k_index] = self.values[top_k_index] + self.alpha * (n_steps_q - self.values[top_k_index]) 
        # remove oldest memory if capacity is rearched
        if len(self) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)

        # else:
        #     _, index = self.query(encoded_state, 1)  # get the index of exist_state
        #     self.values[index] = n_steps_q

    def query_k(self, encoded_state, k):
        """
        args:
            encoded_state: tensor, size=(encoded_dim,)
            k: int, number of neighbors
        return:
            top_k_weights: weights of top k neighbors
            top_k_index: indexs of top k neighbors
        """
        if len(self) >= k:
            # print(len(self), k)
            # print(encoded_state)
            weights = self.get_weights(encoded_state)
            # print(weights)
            # print(self.p)
            top_k_weights, top_k_index = torch.topk(weights, k)
            return top_k_weights, top_k_index
        else:
            return None

    def exist_state(self, encoded_state):
        # for key in self.keys:
        #     if self.kernel_function(encoded_state, key) > threshold:
        #         return True
        #     else:
        #         return False
        if len(self.keys) > 0:
            keys = torch.cat(self.keys, dim=0)  # stack: stack on new axis; cat: cat on existing axis
            similarities = self.kernel_function(encoded_state, keys)
            max_state = similarities.max(-1)
            similarity = max_state[0].item()
            index = max_state[1].item()
            logger.debug('similarity: {} encoded_state: {} closest_state: {}'.format(similarity, encoded_state, self.keys[index]))
            if similarity > self.similarity_threshold:
                return True
        return False

    def get_max_n_steps_q(self):
        return max(self.values) if len(self) > 0 else 0.0 

    def get_expected_n_steps_q(self, encoded_state):
        queried = self.query_k(encoded_state, k=self.p)
        if queried:
            top_k_weights, top_k_index = queried 
            queried_values = torch.tensor(self.values, dtype=torch.float32)[top_k_index]
            n_steps_q = torch.sum(queried_values * top_k_weights).item()
        else:
            n_steps_q = 0.0
        return n_steps_q

    def get_weights(self, encoded_state):
        """
        args:
            encoded_state: tensor, size=(encoded_dim,)
        return:
            weights: weights of each value, sum to 1; size=(keys_size,)
        """
        keys = torch.cat(self.keys, dim=0)  # stack: stack on new axis; cat: cat on existing axis
        similarities = self.kernel_function(encoded_state, keys)
        weights = similarities / torch.sum(similarities)
        return weights
    
    def save(self, output_dir, index):
        np.save(os.path.join(output_dir, 'dnd_{}_keys.npy'), self.keys)
        np.save(os.path.join(output_dir, 'dnd_{}_values.npy'), self.values)

    @staticmethod
    def kernel_function(encoded_state, keys):
        """
        encoded_state: tensor, size = (encoded_dim,)
        keys: tensor, size = (keys_size, encoded_dim)
        """
        difference = encoded_state - keys
        distance = difference.norm(dim=-1).pow(2)
        return 1 / (distance + 1e-3)


class NECAgent(object):
    def __init__(self, input_dim, encode_dim, hidden_dim, output_dim, capacity, buffer_size, epsilon_start, epsilon_end, decay_factor, lr, p, similarity_threshold, alpha):
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.capacity = capacity
        self.buffer_size = buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.steps_done = 0
        self.decay_factor = decay_factor
        self.lr = lr
        self.p = p
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        
        self.encoder = Encoder(self.input_dim, self.encode_dim, self.hidden_dim)
        # one dnd one one action; query by index of a list
        self.dnd_list = [DND(self.encode_dim, self.capacity, self.p, self.similarity_threshold, self.alpha) for _ in range(self.output_dim)]

        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)
        self.optimizer = RMSprop(self.encoder.parameters(), lr=self.lr)

    def greedy_infer(self, state):
        n_steps_q = torch.zeros(self.output_dim, dtype=torch.float32)
        with torch.no_grad():
            encoded_state = self.encoder(state)
        for i, dnd in enumerate(self.dnd_list):
            n_steps_q[i] = dnd.get_expected_n_steps_q(encoded_state)
        max_output = n_steps_q.max(0) 
        value = max_output[0].view(1, 1)
        action = max_output[1].view(1, 1)
        logger.debug('n_steps_q: {}'.format(n_steps_q.numpy()))
        return action, value, encoded_state

    def epsilon_greedy_infer(self, state):
        action, value, encoded_state = self.greedy_infer(state)
        random_number = random.random()
        if random_number < self.epsilon:
            action = torch.tensor([[random.randrange(self.output_dim)]], dtype=torch.long)
        return action, value, encoded_state

    def replay(self, batch_size):
        batch = self.replay_buffer.get_batch(batch_size)
        states = torch.tensor([batch[i][0] for i in range(batch_size)], dtype=torch.float32)
        actions = torch.tensor([batch[i][1] for i in range(batch_size)], dtype=torch.long)
        expected_next_values = torch.tensor([batch[i][2] for i in range(batch_size)], dtype=torch.float32)
        
        n_steps_q = np.zeros(self.output_dim)
        encoded_state = self.encoder(states)
        for i, dnd in enumerate(self.dnd_list):
            n_steps_q[i] = dnd.get_expected_n_steps_q(encoded_state)
        values = torch.tensor(n_steps_q[actions], dtype=torch.float32)  ## part of computation graph
        loss = F.mse_loss(values, expected_next_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
        optimizer.step()

    def get_target_n_steps_q(self):
        target_n_steps_q = torch.zeros(self.output_dim, dtype=torch.float32)
        for i in range(self.output_dim):
            target_n_steps_q[i] = self.dnd_list[i].get_max_n_steps_q()
        return target_n_steps_q.max()  # max() of a 0-dim tensor 

    # def update_dnd_values(self, )
    def remember_to_replay_buffer(self, state, action, n_steps_q):
        self.replay_buffer.append(state, action, n_steps_q)

    def remember_to_dnd(self, encoded_state, action, n_steps_q):
        self.dnd_list[action].append(encoded_state, n_steps_q)

    def epsilon_decay(self):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps_done / self.decay_factor)

    def save_dnd(self, output_dir):
        for index, dnd in enumerate(self.dnd_list):
            dnd.save(output_dir, index)


if __name__ == '__main__':
    main()