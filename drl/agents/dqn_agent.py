import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

from drl.utils import get_logger
from drl.blocks.memory.replay_buffer import ReplayBuffer
from drl.core.agent import ValueBasedAgent


logger = get_logger(__name__, fh_lv='debug', ch_lv='info')


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.output_layer(x)  # linear outputs correspond to q-value of each action
        return x

    def get_weights(self):
        weights = dict() 
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights


class DQNAgent(ValueBasedAgent):

    HYPARAMS_INFO = {
        'lr': (float, 'learning rate for neural network optimization'),
        'gamma': (float, 'discount factor of cummulative rewards'),
        'epsilon_start': (float, 'exploration rate on start'),
        'epsilon_end': (float, 'exploration rate on end'),
        'decay_factor': (float, 'decay factor for exploration'), 
        'buffer_size': (int, 'maximum capacity of the replay buffer'),
        'batch_size': (int, 'batch size for neural network optimization'),
        'n_step_reward': (int, 'number of steps of TD reward'),
        'learn_per_step': (int, 'frequency of learning (per step)'),
        'update_target_per_step': (int, 'frequency of updating the target model (per step)'),
    }

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
        update_target_per_step=10,
        n_warmup_steps=1000,
        mode='train',
        writer=None
    ):
        super().__init__(input_dim, output_dim, lr, gamma, epsilon_start, epsilon_end, decay_factor, buffer_size, batch_size, n_step_reward, learn_per_step, n_warmup_steps, mode, writer)

        self.update_target_per_step = update_target_per_step

        self.replay_buffer = ReplayBuffer(capacity=self.buffer_size)
        self.model = QNetwork(self.input_dim, self.output_dim)
        self.target_model = QNetwork(self.input_dim, self.output_dim)

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

        self.n_target_updates = 0  # counter for target model updates

    def learn(self):
        batch = self.replay_buffer.get_batch(self.batch_size)
        #######################
        # process batch to pytorch tensor
        #######################
        states = torch.tensor([batch[i].state for i in range(self.batch_size)], dtype=torch.float32)
        rewards = torch.tensor([batch[i].reward for i in range(self.batch_size)], dtype=torch.float32)
        actions = torch.tensor([batch[i].action for i in range(self.batch_size)], dtype=torch.long).unsqueeze(1)  # actions and states must share the same dimensions
        next_states = torch.tensor([batch[i].next_state for i in range(self.batch_size)], dtype=torch.float32)
        dones = torch.tensor([batch[i].done for i in range(self.batch_size)], dtype=torch.bool)
        values = self.model(states).gather(1, actions)  # Q_a value with a = argmax~a(Q)
        next_values = torch.zeros(self.batch_size, dtype=torch.float32)
        next_values[~dones] = self.target_model(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        
        return loss

    def evaluate(self, state):
        """
        evaluate a state with target_network
        """
        with torch.no_grad():
            target_value = self.target_model(state).max(1)[0].view(1, 1)
        return target_value

    def greedy_infer(self, state):
        with torch.no_grad():
            # pytorch doc: Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim
            max_output = self.model(state).max(1)
            # print('max_output: {}'.format(self.policy_network(state)))
            value = max_output[0].view(1, 1)
            action = max_output[1].view(1, 1)
        return action, value

    def remember(self, state, reward, action, next_state, done):
        self.replay_buffer.append(state, reward, next_state, action, done)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.n_target_updates += 1

    def call_after_n_step_reward(self, global_steps, history):
        transition = history[0]
        self.remember(
            state=transition.state,
            reward=transition.reward,
            action=transition.action,
            next_state=transition.next_state,
            done=transition.done,
        )

    def call_end_of_step(self, global_steps, steps, writer=None):
        super().call_end_of_step(global_steps, steps, writer)

        if global_steps % self.update_target_per_step == 0 and self.n_grad_steps > 0:  # start updating the target model only after 1 grad step
            self.update_target_network()
        
        logger.debug('self.n_target_updates: {} | self.n_grad_steps: {}'.format(self.n_target_updates, self.n_grad_steps))
        if writer is not None:
            if self.n_grad_steps > 0:
                writer.log_scalar(
                    iteration=global_steps,
                    train_data={
                        'step/target update to grad step': self.n_target_updates / self.n_grad_steps 
                    }
                )
            elif self.n_target_updates > 0:
                logger.error('There should be no target update before any gradient step.')

    def call_end_of_episode(self, episode, writer=None):
        super().call_end_of_episode(episode, writer)
        if writer is not None:
            writer.log_scalar(
                iteration=episode,
                train_data={
                    'episode/percentage usage of buffer': len(self.replay_buffer) / self.replay_buffer.capacity
                }
            )
        if episode % 10 == 0 and episode != 0:
            self.save_network()