from abc import ABC, abstractmethod
import random
import math

import numpy as np
import torch


class BaseAgent(ABC):

    HYPARAMS_INFO = {}

    writer = None
    optimizer = None
    learn_per_step = 10
    n_warmup_steps = 1000
    mode = 'train'
    n_grad_steps = 0

    @abstractmethod
    def act(self):
        """"""

    # @abstractmethod
    # def encode(self):    
    #     """"""

    @property
    def hyparams_info(self):
        return self.HYPARAMS_INFO

    def save_checkpoint(self, filepath):
        raise NotImplementedError
    
    def save_network(self, output_dir):
        raise NotImplementedError

    def call_before_training(self):
        return

    def call_after_warmup(self, global_steps, steps, writer=None):
        return
    
    def call_after_n_step_reward(self, global_steps, history, writer=None):
        return
    
    def call_end_of_step(self, global_steps, steps, writer=None):
        return

    def call_end_of_episode(self, episode, writer=None):
        return

    def backprop(self, global_steps, writer=None):
        
        self.optimizer.zero_grad()
        loss = self.learn()
        loss.backward()
        # gradient clipping
        for param in self.model.parameters():
            if param.grad is not None:  # some parameters in dnd do not have grad, since they are retrieved, TODO: verify this hypothesis
                param.grad.data.clamp_(-1, 1)  # |grad| <= 1
        self.optimizer.step()
        
        loss = loss.item()
        self.n_grad_steps += 1  # counter for gradient steps
        if writer is not None:
            writer.log_scalar(
                iteration=global_steps,
                train_data={
                    'step/loss': loss
                }
            )
    
    @abstractmethod
    def is_ready_to_learn(self):
        """"""


class ValueBasedAgent(BaseAgent):

    replay_buffer = None

    def __init__(
        self,
        input_dim,
        output_dim,
        lr=1e-3,
        gamma=0.99, 
        epsilon_start=0.9,
        epsilon_end=0.1,
        decay_factor=200, 
        buffer_size=10*10**5,
        batch_size=32,
        n_step_reward=1,
        learn_per_step=4,
        n_warmup_steps=1000,
        update_target_per_step=400,
        mode='train',
        writer=None
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.buffer_size = buffer_size
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step_reward = n_step_reward
        self.gamma_vector = np.array([self.gamma**i for i in range(self.n_step_reward)])

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_factor = decay_factor

        self.learn_per_step = learn_per_step
        self.update_target_per_step = update_target_per_step
        self.n_warmup_steps = n_warmup_steps
        
        self.mode = mode

        self.steps_done = 0

        self.writer = writer

    def act(self, state):
        if self.mode == 'train':
            action, value = self.epsilon_greedy_infer(state)
        else:
            action, value = self.greedy_infer(state)
        return action, value
    
    def evaluate(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError('Agents that use experience repplay must override this method.')

    def remember(self):
        raise NotImplementedError('Agents that use experience repplay must override this method.')

    def greedy_infer(self, state):
        raise NotImplementedError

    def get_random_action(self):
        random_actions = torch.tensor([[random.randrange(self.output_dim)]], dtype=torch.long)
        return random_actions

    def epsilon_greedy_infer(self, state):
        random_actions = self.get_random_action()
        actions, values = self.greedy_infer(state)
        random_number = random.random()

        if random_number < self.epsilon:
            actions = random_actions
        return actions, values

    def epsilon_decay(self):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps_done / self.decay_factor)

    # def call_after_warmup(self, global_steps, steps, writer=None):
    #     if writer is not None:
    #         writer.log_scalar(
    #             iteration=global_steps,
    #             train_data={
    #                 'step/epsilon': self.epsilon
    #             }
    #         )
    #     self.epsilon_decay()

    def call_end_of_step(self, global_steps, steps, writer=None):
        super().call_end_of_step(global_steps, steps, writer)

        if writer is not None:
            writer.log_scalar(
                iteration=global_steps,
                train_data={
                    'step/epsilon': self.epsilon
                }
            )
        self.epsilon_decay()

    def is_ready_to_learn(self):
        if len(self.replay_buffer) > self.batch_size:
            return True
        else:
            return False