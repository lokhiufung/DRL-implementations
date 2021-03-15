import math
import random
from collections import OrderedDict

import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drl.core.agent import Agent
from drl.core.decorators import take_agent_step
from drl.blocks.memory.replay_buffer import ReplayBuffer
from drl.datasets.replay_buffer_dataset import LowDimReplayBufferDataset


class DQNAgent(Agent):
    def __init__(self, cfg: DictConfig):
        # steup networks
        super().__init__(cfg)

        self.replay_buffer = ReplayBuffer(**cfg.replay_buffer)
        
        self.policy_network = nn.Sequential(OrderedDict([
            ('encoder', instantiate(cfg.network.encoder)),
            ('output_head', instantiate(cfg.network.output_head))
        ]))
        self.target_network = nn.Sequential(OrderedDict([
            ('encoder', instantiate(cfg.network.encoder)),
            ('output_head', instantiate(cfg.network.output_head))
        ]))
        self.target_network.eval()

        self.setup_train_dataloader(cfg.train_data)
        self.setup_optimizers(cfg.optimizer)

        self.gamma = self._cfg.gamma

    def setup_train_dataloader(self, train_cfg):
        self._train_dataset = LowDimReplayBufferDataset(
            self.replay_buffer,
            **train_cfg.dataset
        )
        self._train_dataloader = DataLoader(
            dataset=self._train_dataset,
            collate_fn=self._train_dataset.collate_fn,
        ) 

    def setup_optimizers(self, optim_cfg: DictConfig):
        self._optimizer = instantiate(optim_cfg, params=self.parameters())

    def forward(self, state):
        values = self.policy_network(state)
        return values
    
    def training_step(self, batch, batch_id):

        self.play_step()
        
        states, actions, rewards, next_states, dones = batch
        batch_size = states.size(0)
        actions = actions.unsqueeze(1)
        values = self(states).gather(1, actions) # Q_a value with a = argmax~a(Q)
        
        next_values = torch.zeros(batch_size, dtype=torch.float32, device=states.device)
        next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        
        # slowly update target_network
        if self.agent_steps % 4:
            self.update_target_network()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('aver_q', values.mean(dim=0), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return OrderedDict([
            ('loss', loss),
            ('aver_q', values.mean(dim=0)),
        ])

    @take_agent_step
    @torch.no_grad()
    def play_step(self):
        
        current_state = self._env.current_state

        value, action = self.act(current_state)
        # exploration play
        # if self._exploration_scheduler.eps < random.random():
        if self._exploration_scheduler.get_eps_on_step(self.agent_steps) < random.random():
            action = self._env.sample_action()

        next_state, reward, done, _ = self._env.step(action)

        self.replay_buffer.append(current_state, reward, next_state, action, done)

        if done:
            self._env.reset_for_next_episode()

    def act(self, state: np.array) -> int:
        if len(state.shape) < 2:
            state = np.expand_dims(state, axis=0)
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        values = self(state)
        value, action = values.max(dim=1)
        return value.detach().numpy(), action.detach().numpy()[0]  # probably buggy if index of np.array is used here to retrieve the action index

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())