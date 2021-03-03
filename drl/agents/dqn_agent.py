import math
import random
from collections import OrderedDict

from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from drl.core.agent import Agent
from drl.core.agent import Agent
from drl.network import Network
from drl.blocks.memory.replay_buffer import ReplayBuffer


class DQNAgent(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        # steup networks
        self.policy_network = nn.Sequential(OrderedDict([
            ('encoder', instantiate(cfg.network.encoder)),
            ('output_head', instantiate(cfg.network.output_head))
        ]))
        self.target_network = nn.Sequential(OrderedDict([
            ('encoder', instantiate(cfg.network.encoder)),
            ('output_head', instantiate(cfg.network.output_head))
        ]))

        self.replay_bufer = ReplayBuffer(**cfg.replay_buffer)
        
        self.agent_steps = 0
        self.exploration_scheduler = EpsilonGreeyExplorationScheduler(
            self.agent_steps,
            **cfg.exploration_scheduler
        )
        self.target_update_freq = cfg.target_update_freq
        self.policy_network_freq = cfg.policy_network_freq

    def setup_train_dataset(self, train_cfg):
        self._train_dataset = ReplayBufferDataset(
            self.replay_buffer,
            **train_cfg.dataset
        )
        self._train_dataloader = DataLoader(
            dataset=self._train_dataset,
            **train_cfg.dataloader,
        ) 

    def setup_optimizers(self, optim_cfg: DictConfig):
        self._optimizer = instantiate(optim_cfg)

    def forward(self, state):
        values = self.policy_network(states)
        return values
    
    def training_step(self, batch, batch_id):
        states, rewards, actions, next_states, dones = batch
        
        actions = actions.unsqueeze(1)
        values = self(states).gather(1, actions) # Q_a value with a = argmax~a(Q)
        
        next_values = torch.zeros(batch_size, dtype=torch.float32, device=states.device)
        next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        return {
            'loss': loss.item(),
            'aver_q': values.mean(dim=0).item(),
            'aver_expected_next_q': expected_next_values.mean(dim=1).item(),
        }

    def validation_step(self, batch, batch_id):
        states, rewards, actions, next_states, dones = batch
        
        actions = actions.unsqueeze(1)
        values = self(states).gather(1, actions) # Q_a value with a = argmax~a(Q)
        
        next_values = torch.zeros(batch_size, dtype=torch.float32, device=states.device)
        next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        return {
            'loss': loss.item(),
            'aver_q': values.mean(dim=0).item(),
            'aver_expected_next_q': expected_next_values.mean(dim=1).item(),
        }

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

    