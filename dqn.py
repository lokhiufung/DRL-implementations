import os
import random
import math
import argparse
from itertools import count
from collections import deque

# import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
import gym

from tensorboard_logger import TensorboardLogger
from utils import get_logger, compare_weights, load_json


logger = get_logger('dqn.py', fh_lv='debug', ch_lv='info')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', required=True, type=str, help='name of experiment')
    args = parser.parse_args()

    experiment_name = args.name
    hyparams = load_json('./hyparams/dqn_hyparams.json')[experiment_name]['hyparams']
    # hyparameters
    lr = hyparams['lr']
    buffer_size = hyparams['buffer_size']
    gamma = hyparams['gamma']
    epsilon_start = hyparams['epsilon_start']
    epsilon_end = hyparams['epsilon_end']
    decay_factor = hyparams['decay_factor']
    batch_size = hyparams['batch_size']
    replay_freq = hyparams['replay_freq']
    target_update_freq = hyparams['target_update_freq'] 
    episodes = hyparams['episodes']
    warmup_steps = hyparams['warmup_steps']
    # max_steps = 1e10
    logger.debug('experiment_name: {} hyparams: {}'.format(experiment_name, hyparams))

    # write to tensorboard 
    tensorboard_logdir = './experiment'
    if not os.path.exists(tensorboard_logdir):
        os.mkdir(tensorboard_logdir)
    writer = TensorboardLogger(logdir=tensorboard_logdir)

    env = gym.make('CartPole-v0')
    env.reset()
    # logger.debug('observation_space.shape: {}'.format(env.observation_space.shape))
    agent = DQNAgent(buffer_size, writer=writer, input_dim=env.observation_space.shape[0], output_dim=env.action_space.n, gamma=gamma, epsilon_start=epsilon_start, epsilon_end=epsilon_end, decay_factor=decay_factor)

    state, _, _, _ = env.step(env.action_space.sample()) # take a random action to start with
    writer.add_graph(agent.policy_network, torch.tensor([state], dtype=torch.float32))  # add model graph to tensorboard
    # state, reward, done, info = env.step(env.action_space.sample()) # take a random action to start with
    # for i in range(50):
    #     agent.remember(state, reward, env.action_space.sample(), state, False)
    # for i in range(50):
    #     agent.remember(state, reward, env.action_space.sample(), state, True)
    # loss = agent.replay(batch_size=5)
    global_steps = 0
    for episode in range(episodes):
        score = 0.0
        total_loss = 0.0
        env.reset()
        logger.debug('env.reset() episode {} starts!'.format(episode))
        for step in count():
            # env.render()
            action_tensor, value_tensor = agent.epsilon_greedy_infer(torch.tensor([state], dtype=torch.float32))
            _, target_value_tensor = agent.greedy_infer(torch.tensor([state], dtype=torch.float32))  # temp: for debug
            next_state, reward, done, info = env.step(action_tensor.item()) # take a random action
            # action = env.action_space.sample()
            # next_state, reward, done, info = env.step(action) # take a random action
            # logger.debug('episode: {} state: {} reward: {} action: {} next_state: {} done: {}'.format(episode, state, reward, action, next_state, done))
            agent.remember(state, reward, action_tensor.item(), next_state, done)
            # 2. test QNetwork
            # logger.debug('state_tensor: {} action_tensor: {} value_tensor: {}'.format(state_tensor, action_tensor, value_tensor))
            # logger.debug('state_tensor: {} action: {} value: {}'.format(state_tensor, action_tensor.item(), value_tensor.item()))
            # print('state: {} reward: {} action_tensor.item(): {} next_state: {} done: {}'.format(state, reward, action_tensor.item(), next_state, done))
            score += reward
            # experience replay
            if global_steps > max(batch_size, warmup_steps) and global_steps % replay_freq == 0:
                loss = agent.replay(batch_size)
                total_loss += loss
                logger.debug('episode: {} done: {} global_steps: {} loss: {}'.format(episode, done, global_steps, loss))
                writer.log_training(global_steps, loss, agent.lr, value_tensor.item(), target_value_tensor.item(), agent.epsilon)
            # update target_network 
            if global_steps > max(batch_size, warmup_steps) and global_steps % target_update_freq == 0:
                # 1. test replay_bufer 
                # logger.debug('step: {} number of samples in bufer: {} sample: {}'.format(step, len(agent.replay_buffer), agent.replay_buffer.get_batch(2)))
                agent.update_target_network()
            if global_steps > max(batch_size, warmup_steps) and global_steps % 1000:
                writer.log_linear_weights(global_steps, 'encoder.0.weight', agent.policy_network.get_weights()['encoder.0.weight'])
            agent.epsilon_decay()
            state = next_state  # update state manually
            global_steps += 1
            if done:
                logger.info('episode done! episode: {} score: {}'.format(episode, score))
                writer.log_episode(episode, total_loss / (step + 1), score)
                break
            # logger.debug('state_tensor: {} action_tensor: {} value_tensor: {}'.format(state_tensor, action_tensor, value_tensor))
            # logger.debug('output: {} state_tensor: {} state: {}'.format(output, state_tensor, state))
            # agent.remember(state, reward, action, next_state, done)
    env.close()


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.output_dim = output_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.output_layer = torch.nn.Linear(128, self.output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.output_layer(x)  # linear outputs correspond to q-value of each action
        return x

    def get_weights(self):
        weights = dict() 
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)  # once maxlen is rearched, the left sample will be poped out

    def append(self, state, reward, action, next_state, done):
        self.buffer.append((state, reward, action, next_state, done))

    def get_batch(self, batch_size):
        # random.sample() for sampling without replacement
        return random.sample(self.buffer, k=batch_size)

    def __len__(self):
        return len(self.buffer)

        
class DQNAgent(object):
    def __init__(self, buffer_size, input_dim, output_dim, writer, mode='train', gamma=0.9, epsilon_start=0.9, epsilon_end=0.05, decay_factor=200.0, lr=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.buffer_size = buffer_size
        self.lr = lr
        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)
        self.policy_network = QNetwork(self.input_dim, self.output_dim)
        self.target_network = QNetwork(self.input_dim, self.output_dim)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=self.lr)
        self.gamma = gamma
        # decay states
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.steps_done = 0
        self.decay_factor = decay_factor
        # tensorboard writer
        self.writer = writer
        self.target_network.eval()
        # synchronize the weights in both policy and target network
        self.update_target_network()

    def replay(self, batch_size):
        batch = self.replay_buffer.get_batch(batch_size)
        #######################
        # process batch to pytorch tensor
        #######################
        states = torch.tensor([batch[i][0] for i in range(batch_size)], dtype=torch.float32)
        rewards = torch.tensor([batch[i][1] for i in range(batch_size)], dtype=torch.float32)
        actions = torch.tensor([batch[i][2] for i in range(batch_size)], dtype=torch.long).unsqueeze(1)  # actions and states must share the same dimensions
        next_states = torch.tensor([batch[i][3] for i in range(batch_size)], dtype=torch.float32)
        dones = torch.tensor([batch[i][4] for i in range(batch_size)], dtype=torch.bool)
        # print('state: {} rewards: {} actions: {} next_states: {} dones: {}'.format(states, rewards, actions, next_states, dones))
        # print('output policy_netowrk: {}'.format(self.policy_network(states)))
        values = self.policy_network(states).gather(1, actions)  # Q_a value with a = argmax~a(Q)
        # print('values: {}'.format(values))
        next_values = torch.zeros(batch_size, dtype=torch.float32)
        # print('dones: {}'.format(dones))
        # print('self.target_network(next_states).max(1)[0]: {}'.format(self.target_network(next_states).max(1)[0]))
        next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        # print('next_values: {}'.format(next_values))
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        # print('expected_next_values: {}'.format(expected_next_values.unsqueeze(1)))
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping, (-1, 1)
        # grad_norm = 0.0
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)  # gradient cliping |grad| < = 1, clamp_ in-place original tensor, .data to get underlying tensor of a variable
            # grad_norm += param.grad.data.norm().item()**2 
        # self.writer.log_training(self.steps_done, loss, agent.lr, value_tensor.item(), target_value_tensor.item(), agent.epsilon)  # temp: self.steps_done == global_steps
        self.optimizer.step()
        # self.steps_done += 1  # train step + 1
        # self.epsilon_decay()

        return loss.item()#, grad_norm**0.5

    def greedy_infer(self, state):
        with torch.no_grad():
            # pytorch doc: Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim
            max_output = self.policy_network(state).max(1)
            # print('max_output: {}'.format(self.policy_network(state)))
            value = max_output[0].view(1, 1)
            action = max_output[1].view(1, 1)
        return action, value

    def epsilon_greedy_infer(self, state):
        action, value = self.greedy_infer(state)
        random_number = random.random()
        # print('random_number: {} epsilon: {}'.format(random_number, self.epsilon))
        if random_number > self.epsilon:
            action = torch.tensor([[random.randrange(self.output_dim)]], dtype=torch.long)
        return action, value

    def remember(self, state, reward, action, next_state, done):
        self.replay_buffer.append(state, reward, action, next_state, done)

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
    def epsilon_decay(self):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps_done / self.decay_factor)
        # print('steps_done: {}'.format(self.steps_done))
        # print('epsilon: {}'.format(self.epsilon))

    def save_checkpoint(self, output_dir):
        """
        save checkpoint for restore training
        """
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(output_dir, 'checkpoint_{}'.format(self.steps_done)))

    def save_network(self, output_dir):
        """
        save checkpoint for restore training
        """
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict()
        })


if __name__ == '__main__':
    main()