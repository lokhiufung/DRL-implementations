import math
import random
from collections import OrderedDict

from hydra.utils import instantiate 
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
# from drl.core.agent import Agent
from drl.network import Network
from drl.blocks.memory.replay_buffer import ReplayBuffer



class DQNAgent(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.policy_network = nn.Sequential(OrderedDict([
            ('encoder', )
        ]))
        self.target_network = Network.from_config(**agent_parameters['network'])
        self.replay_bufer = ReplayBuffer(**agent_parameters['replay_buffer'])

        optimizer_class = getattr(optim, optim_param['name'])
        self.optimizer = optimizer_class(self.network.parameters(). **optim_param)
        
        self.epsilon_start = agent_parameters['epsilon_start']
        self.epsilon_end = agent_parameters['epsilon_end']
        self.decay_factor = agent_parameters['decay_factor']
        self.epsilon = agent_parameters['epsilon_start']
        self.exploration_steps = agent_parameters['exploration_steps']
        self.gamma = agent_parameters['gamma']
        self.target_update_freq = agent_parameters['target_update_freq']
        self.policy_network_freq = agent_parameters['policy_update_freq']

        self.step_counter = 0
        if mode == 'train':
            self.batch_size = agent_parameters['batch_size']
            self.train_step_counter = 0

    def setup_optimizers(self)
    def _learn(self):
        batch = self.replay_buffer.get_batch(batch_size)
        #######################
        # process batch to pytorch tensor
        #######################
        states, rewards, actions, next_states, dones = batch
        actions = actions.unsqueeze(1)
        # states = torch.tensor([batch[i][0] for i in range(batch_size)], dtype=torch.float32)
        # rewards = torch.tensor([batch[i][1] for i in range(batch_size)], dtype=torch.float32)
        # actions = torch.tensor([batch[i][2] for i in range(batch_size)], dtype=torch.long).unsqueeze(1)  # actions and states must share the same dimensions
        # next_states = torch.tensor([batch[i][3] for i in range(batch_size)], dtype=torch.float32)
        # dones = torch.tensor([batch[i][4] for i in range(batch_size)], dtype=torch.bool)
        values = self.policy_network(states).gather(1, actions)  # Q_a value with a = argmax~a(Q)
        next_values = torch.zeros(batch_size, dtype=torch.float32)
        next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        
        self.optimizer.zero_grad()
        loss.backward()
        self.policy_network.clip_gradient()
        self.optimizer.step()
        return loss.item() #, grad_norm**0.5

    def greedy_infer(self, sensory_input):
        with torch.no_grad():
            # pytorch doc: Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim
            max_output = self.policy_network(sensory_input).max(1)
            # print('max_output: {}'.format(self.policy_network(state)))
            value = max_output[0].view(1, 1)
            action = max_output[1].view(1, 1)
        return action, value

    def _act(self, sensory_input):
        if random.random() < self.epsilon:
            action, _ = self.greedy_infer(sensory_input)
            return action
        else:
            return random.choice(range(self.num_action))

    def act(self, sensory_input):
        # how many interactions has been taken
        self.step_counter += 1
        # epsilon exponential decay
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.step_counter / self.decay_factor)
        # update target_network if neccessary
        if (self.step_counter - self.exploration_steps) % self.target_update_freq and self.step_counter > self.exploration_steps:
            self.target_network.clone_weights(self.policy_network)
        return self._act(sensory_input)

    def learn(self):
        self.train_step_counter += 1
        return self._learn()

    