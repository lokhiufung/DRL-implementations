import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from drl.core.agent import ValueBasedAgent
from drl.blocks.memory.dnd import DifferentiableNeuralDictionary
from drl.blocks.memory.replay_buffer import NECReplayBuffer
from drl.utils import get_logger


logger = get_logger(__name__, fh_lv='debug', ch_lv='debug')


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


class NECAgent(ValueBasedAgent):
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
        embedding_dim=32,
        p=50,
        similarity_threshold=0.5,
        alpha=1.0,
        mode='train',
        writer=None
    ):
        super().__init__(input_dim, output_dim, lr, gamma, epsilon_start, epsilon_end, decay_factor, buffer_size, batch_size, n_step_reward, learn_per_step, n_warmup_steps, mode, writer)

        self.embedding_dim = embedding_dim
        self.p = p
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha

        self.replay_buffer = NECReplayBuffer(capacity=buffer_size)
        self.model = NECNetwork(
            input_dim=self.input_dim,
            n_actions=self.output_dim,
            embedding_dim=self.embedding_dim,
            p=self.p
        )

        self.cache_for_dnd = None
        self._reset_cache_for_dnd()

    def act(self, state):
        if not self.model.dnd.is_ready():
            action_tensor = self.get_random_action()
            value_tensor = torch.tensor([[0.0]], dtype=torch.float)
        else:
            action_tensor, value_tensor = self.epsilon_greedy_infer(state)
        
        return action_tensor, value_tensor

    def evaluate(self, state):
        if self.model.dnd.is_ready():
            max_q = self.model.get_max_q()
            # max_q = max_q.item()
            return max_q
        else:
            return torch.tensor([[0.0]], dtype=torch.float)

    def encode(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        key = self.model.encode(state)
        return key
    
    def greedy_infer(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        max_values, actions, indexes, scores, keys = self.model.predict(state)  # TODO: make the ordering of action, value consistent
        # extra = {'scores': scores, 'indexes': indexes, 'keys': keys}
        return actions, max_values# return scale values of action and value. extra term is a dict containing additional values for reference

    def learn(self, batch_size):

        # TODO: check the replay logic
        batch = self.replay_buffer.get_batch(batch_size)

        # reminder: make sure to cast to float before creating the computation graph
        states = torch.tensor([batch[i].state for i in range(batch_size)]).float()
        actions = torch.tensor([batch[i].action for i in range(batch_size)]).long()
        q_targets = torch.tensor([batch[i].q_target for i in range(batch_size)]).float()

        qs, _, _, _, _ = self.model(states, return_all_values=True)
        
        actions = actions.unsqueeze(-1)
        qs = torch.gather(qs, 1, actions).squeeze()
        # TODO: dims of qs and q_targets do not match 
        loss = F.mse_loss(qs, q_targets)
        return loss

    def is_ready_to_learn(self):
        if self.model.dnd.is_ready() and len(self.replay_buffer) > self.batch_size:
            return True
        else:
            return False
            
    def commit_single_to_dnd(self, action, key, value):
        """commit a single record to agent's dnd buffer
        """
        self.cache_for_dnd['actions'].append(action)
        self.cache_for_dnd['keys'].append(key)
        self.cache_for_dnd['values'].append(value)
        
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
                values=self.cache_for_dnd['values']
            )
        # if success, reset the cache
        self._reset_cache_for_dnd()

    def write_dnd(self):
        self.model.dnd.write()

    def _reset_cache_for_dnd(self):
        self.cache_for_dnd = {
            'actions': [],
            'keys': [],
            'values': [],
        }

    def remember(self, state, q_target, action):
        self.replay_buffer.append(state, q_target, action)

    def get_max_q(self):
        max_q = self.model.dnd.get_max_value()
        return max_q

    def save_checkpoint(self, output_dir):
        """
        save checkpoint for restore training
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(output_dir, 'checkpoint_{}'.format(self.steps_done)))

    # def save_network(self, output_dir):
    #     """
    #     save checkpoint for restore training
    #     """
    #     torch.save({
    #         'model_state_dict': self.model.state_dict(),
    #         'target_model_state_dict': self.target_model.state_dict()
    #     })

    def call_after_n_step_reward(self, global_steps, history):
        
        state, action, rewards = history.get_state_action_rewards()
        n_step_reward = len(history)

        if not history.check_done():  # reached done in the last transition
            max_q = self.get_max_q()  # use the maximum q to bootstrap the term max Q(s_t+N, a')
        else:
            max_q = 0.0
        q_target = np.dot(np.array(rewards), self.gamma_vector) + self.gamma**n_step_reward * max_q 

        self.remember(state, q_target, action)  # save to replay buffer
        # reminder: q_target will plays the role of the target network in DQN
        key = self.encode(state).unsqueeze(0)  # reminder: add an extra dim to key tensor
        self.commit_single_to_dnd(
            action=action,
            key=key,
            value=torch.tensor(q_target, dtype=torch.float),  # reminder: store the Q_TARGET for value estimate
        )

    def call_end_of_episode(self, episode, writer):
        super().call_end_of_episode(episode, writer)

        if len(self.cache_for_dnd['actions']) > 0:
            self.push_to_dnd()  # push the updates to dnd
            self.write_dnd()  # write to search_engine
