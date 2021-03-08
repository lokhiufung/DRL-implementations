import math
import random
from collections import OrderedDict

from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drl.core.agent import Agent
from drl.core.decorators import epsilon_greedy_play_step
from drl.blocks.memory.replay_buffer import ReplayBuffer


class DQNAgent(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
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
        
        self.target_update_freq = cfg.target_update_freq
        self.policy_network_freq = cfg.policy_network_freq
        
        super().__init__(cfg)

    def setup_train_dataset(self, train_cfg):
        self._train_dataset = ReplayBufferDataset(
            self.replay_buffer,
            **train_cfg.dataset
        )
        self._train_dataloader = DataLoader(
            dataset=self._train_dataset,
            collate_fn=self._train_dataset.collate_fn,
        ) 

    def setup_optimizers(self, optim_cfg: DictConfig):
        self._optimizer = instantiate(optim_cfg)

    def forward(self, state):
        values = self.policy_network(states)
        return values
    
    def training_step(self, batch, batch_id):

        self.play_step()

        states, rewards, actions, next_states, dones = batch
        
        actions = actions.unsqueeze(1)
        values = self(states).gather(1, actions) # Q_a value with a = argmax~a(Q)
        
        next_values = torch.zeros(batch_size, dtype=torch.float32, device=states.device)
        next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        return OrderedDict([
            ('loss', loss),
            ('aver_q', values.mean(dim=0)),
            ('aver_expected_next_q', expected_next_values.mean(dim=1)),
        ])

    # def validation_step(self, batch, batch_id):
    #     states, rewards, actions, next_states, dones = batch
        
    #     actions = actions.unsqueeze(1)
    #     values = self(states).gather(1, actions) # Q_a value with a = argmax~a(Q)
        
    #     next_values = torch.zeros(batch_size, dtype=torch.float32, device=states.device)
    #     next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
    #     expected_next_values = rewards + self.gamma * next_values  # bellman's equation
    #     loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
    #     return OrderedDict([
    #         ('loss', loss.item()),
    #         ('aver_q', values.mean(dim=0).item()),
    #         ('aver_expected_next_q', expected_next_values.mean(dim=1).item()),
    #     ])
    
    @torch.no_grad()
    def play_step(self):
        
        current_state = self._env.current_state

        value, action = self.act(current_state)
        # exploration play
        if self._exploration_scheduler.eps < random.random():
            action = self._env.sample_action()

        next_state, reward, done, _ = self._env.step()

        self.replay_buffer.append((current_state, reward, next_state, action, done))

        if done:
            self._env.reset_for_next_episode()

    def act(self, state: np.array) -> int:
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        values = self(state)
        value, action = values.max(1)
        return value, action


    