import torch

from drl.agents.base_agent import BaseAgent
from drl.parts.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(self, action_type, batch_size, buffer_size, approximator):
        super(DQNAgent, self).__init__(action_type, mode, batch_size, *args, **kwargs)
        

        self.replay_buffer = ReplayBuffer(buffer_size)

    def _fit(self, data):
        self._fit_replay_buffer()
        if global_steps > self.warmup_steps:
            data = self.replay_buffer.get_batch(batch_size)
            self._fit_approximator(data)


    def _fit_approximator(self, data):
        pass

    def _fit_replay_buffer(self, data):
        self.replay_buffer.append(data)

    def _fit_approximator(self, data):
        states, rewards, actions, next_states, dones = data.state, data.reward, data.action, data.next_state, data.done
        values = self.policy_network(states).gather(1, actions)  # Q_a value with a = argmax~a(Q)
        next_values = torch.zeros(batch_size, dtype=torch.float32)
        next_values[~dones] = self.target_network(next_states).max(1)[0][~dones].detach()  # detach this node from compution graph for preventing gradient flowing to target network
        expected_next_values = rewards + self.gamma * next_values  # bellman's equation
        loss = F.smooth_l1_loss(values, expected_next_values.unsqueeze(1))  # expand dims to match the output of policy_network
        
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)  # gradient cliping |grad| < = 1, clamp_ in-place original tensor, .data to get underlying tensor of a variable
        self.optimizer.step()

        return loss.item()
