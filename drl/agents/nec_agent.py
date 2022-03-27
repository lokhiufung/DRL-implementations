import logging

import torch
import torch.nn.functional as F
import numpy as np

from drl.core.agent import ValueBasedAgent


logger = logging.getLogger(__name__)


class NECAgent(ValueBasedAgent):

    def act(self, state):
        if not self.model.dnd.is_ready():
            keys = self.encode(state)
            action_tensor = self.get_random_action()
            value_tensor = torch.tensor([[0.0]], dtype=torch.float)
        else:
            action_tensor, value_tensor, extra = self.epsilon_greedy_infer(state)
            keys = extra['keys']
            indexes = extra['indexes'][0]
            scores = extra['scores'][0]
        
        return action_tensor, value_tensor

    def encode(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        key = self.model.encode(state)
        return key
    
    def greedy_infer(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        # inference = self.try_greedy_infer(state)
        max_values, actions, indexes, scores, keys = self.model.predict(state)  # TODO: make the ordering of action, value consistent
        extra = {'scores': scores, 'indexes': indexes, 'keys': keys}
        # if inference is not None:
        #     actions, max_values, extra = inference
        return actions, max_values, extra  # return scale values of action and value. extra term is a dict containing additional values for reference

    def learn(self, batch_size):
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

            # loss.backward()
            # # gradient clipping
            # for param in self.model.parameters():
            #     if param.grad is not None:  # some parameters in dnd do not have grad, since they are retrieved, TODO: verify this hypothesis
            #         param.grad.data.clamp_(-1, 1)  # |grad| <= 1
            # self.optimizer.step()
            return loss
        else:
            logger.debug('dnd is not ready yet. Do not replay now.')

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

    def _reset_cache_for_dnd(self):
        self.cache_for_dnd = {
            'actions': [],
            'keys': [],
            'indexes': [],
            'scores': [],
            'values': [],
        }

    def get_max_q(self):
        max_q = self.dnd.get_max_value()
        return max_q

    def call_after_n_step_reward(self, global_steps, history):
        
        state, action, rewards = history.get_state_action_rewards()
        n_step_reward = len(history)

        if history.check_done():
            q_target = np.dot(np.array(rewards), self.gamma_vector)
        else:
            max_q = self.get_max_q()  # use the maximum q to bootstrap the term max Q(s_t+N, a')
            assert isinstance(max_q, float)
            q_target = np.dot(np.array(rewards), self.gamma_vector) + self.gamma**n_step_reward * max_q 

        self.remember(state, q_target, action)  # save to replay buffer
        # reminder: q_target will plays the role of the target network in DQN

        self.commit_single_to_dnd(
            action=action,
            key=keys,
            value=torch.tensor(q_target, dtype=torch.float),  # reminder: store the Q_TARGET for value estimate
            index=indexes[0] if indexes is not None else indexes,
            score=scores[0] if scores is not None else scores,
        )

    def call_end_of_episode(self):
        if len(self.cache_for_dnd['actions']) > 0:
            self.push_to_dnd()  # push the updates to dnd
            self.write_dnd()  # write to search_engine
