# import random

# import numpy as np
# import torch
# from torch.optim import RMSprop

# from agents.encoders import FullyConnectedEncoder
# from agents.parts import DND
# from agents.parts import ReplayBuffer


# class NECAgent(object):
#     def __init__(
#         self, input_dim, encode_dim, hidden_dim, output_dim, capacity,
#         buffer_size, epsilon_start, epsilon_end, decay_factor,
#         lr, p, similarity_threshold, alpha
#         ):
#         self.input_dim = input_dim
#         self.encode_dim = encode_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.capacity = capacity
#         self.buffer_size = buffer_size
#         self.epsilon_start = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon = epsilon_start
#         self.steps_done = 0
#         self.decay_factor = decay_factor
#         self.lr = lr
#         self.p = p
#         self.similarity_threshold = similarity_threshold
#         self.alpha = alpha
        
#         self.encoder = FullyConnectedEncoder(self.input_dim, self.encode_dim, self.hidden_dim)
#         # one dnd one one action; query by index of a list
#         self.dnd_list = [DND(self.encode_dim, self.capacity, self.p, self.similarity_threshold, self.alpha) for _ in range(self.output_dim)]

#         self.replay_buffer = ReplayBuffer(max_size=self.buffer_size)
#         self.optimizer = RMSprop(self.encoder.parameters(), lr=self.lr)

#     @staticmethod
#     def _ensure_tensor(input_):
#         """
#         all inputs that comes into the object must pass through this pipe first 
        
#         return Tensor
#         """
#         if isinstance(input_, np.ndarray):
#             input_ = torch.from_numpy(input_).float().unsqueeze(0)
#         return input_

#     def encode_state(self, state, return_tensor=False):
#         """
#         encode state
#         """
#         state = self._ensure_tensor(state)
#         with torch.no_grad():
#             encoded_state = self.encoder(state)
#         if not return_tensor:
#             encoded_state = encoded_state.cpu().numpy()
#         return encoded_state

#     def greedy_infer(self, state, return_tensor=False):
#         # n_steps_q = torch.zeros(self.output_dim, dtype=torch.float32)
#         state = self._ensure_tensor(state)
#         with torch.no_grad():
#             encoded_state = self.encoder(state)
#         # n_steps_q = []
#         # for dnd in enumerate(self.dnd_list):
#         #     n_step_q.append(
#         #         dnd.get_expected_n_steps_q(encoded_state)
#         #         )
#         n_steps_q = torch.cat(
#             [dnd.get_expected_n_steps_q(encoded_state) for dnd in self.dnd_list]
#         )
#         # n_steps_q = torch.zeros(n_steps_q, dtype=torch.float32)
#         max_output = n_steps_q.max(0) 
#         value = max_output[0].view(1, 1)
#         action = max_output[1].view(1, 1)
#         # logger.debug('n_steps_q: {}'.format(n_steps_q.numpy()))
#         if not return_tensor:
#             action, value, encoded_state = action.cpu().numpy(), value.cpu().numpy(), encoded_state.cpu().numpy()
#         return action, value, encoded_state

#     def epsilon_greedy_infer(self, state, return_tensor=False):
#         action, value, encoded_state = self.greedy_infer(state, return_tensor=return_tensor)
#         random_number = random.random()
#         if random_number < self.epsilon:
#             action = torch.tensor([[random.randrange(self.output_dim)]], dtype=torch.long)
#         if not return_tensor:
#             action = action.cpu().numpy()
#         return action, value, encoded_state

#     def replay(self, batch_size):
#         batch = self.replay_buffer.get_batch(batch_size)
#         states = torch.tensor([batch[i][0] for i in range(batch_size)], dtype=torch.float32)
#         actions = torch.tensor([batch[i][1] for i in range(batch_size)], dtype=torch.long)
#         expected_next_values = torch.tensor([batch[i][2] for i in range(batch_size)], dtype=torch.float32)

#         # computation graph        
#         encoded_state = self.encoder(states)
#         n_steps_q = torch.cat(
#             [dnd.get_expected_n_steps_q(encoded_state) for dnd in self.dnd_list]
#         )
#         values = n_steps_q[actions]  ## part of computation graph
#         loss = F.mse_loss(values, expected_next_values)
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1)
#         optimizer.step()

#     def get_target_n_steps_q(self):
#         target_n_steps_q = torch.zeros(self.output_dim, dtype=torch.float32)
#         for i in range(self.output_dim):
#             target_n_steps_q[i] = self.dnd_list[i].get_max_n_steps_q()
#         return target_n_steps_q.max()  # max() of a 0-dim tensor 

#     # def update_dnd_values(self, )
#     def remember_to_replay_buffer(self, state, action, n_steps_q):
#         self.replay_buffer.append(state, action, n_steps_q)

#     def remember_to_dnd(self, encoded_state, action, n_steps_q):
#         self.dnd_list[action].append(encoded_state, n_steps_q)

#     def epsilon_decay(self):
#         self.steps_done += 1
#         self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps_done / self.decay_factor)

#     def save_encoder(self, output_dir):
#         pass

#     def save_dnd(self, output_dir):
#         for id_, dnd in enumerate(self.dnd_list):
#             dnd.save(output_dir, id_)

#     def load_encoder(self, checkpoint_dir):
#         pass

#     def load_dnd(self, checkpoint_dir):
#         for id_, dnd in enumerate(self.dnd_list):
#             dnd.load(checkpoint_dir, id_)