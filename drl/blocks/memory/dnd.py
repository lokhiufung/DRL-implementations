import os

import numpy as np
import torch
import torch.nn as nn
from annoy import AnnoyIndex


class DifferentiableNeuralDict(nn.Module):
    def __init__(
        self,
        dim,
        kernel_function='inverse_distance',
        capacity: int=10000,
        n_trees: int=10
    ):
        super().__init__()

        self.capacity = capacity
        self.dim = dim
        self.n_trees = n_trees
        self.kernel_function = kernel_function
        self.key_buffer = []
        self.value_buffer = []

        self.search_engine = None  # will be initialized after getting samples in buffer

    def forward(self, key):
        value = self.lookup(key)
        return value

    def lookup(self, key, n_neighbors=50, delta=1e-3):
        if self.search_engine is not None:
            indexes, distances = self.search_engine.get_nns_by_vector(
                key,
                n=n_neighbors,
                include_distance=True,
            )
            retrieved_keys = [self.key_buffer[index] for index in indexes]
            retrieved_values = [self.value_buffer[index] for index in indexes]

            weights = 1 / (torch.pow(key - retrieved_keys, exponent=2) + delta)
            weights_total = torch.sum(weights, dim=-1)

            retrieved_values = torch.from_numpy(retrieved_values)
            output_value = torch.sum(weights * retrieved_values)
            return output_value
        else:
            raise AttributeError('Plesae make sure `self.key_buffer` and `self.value_buffer` are not empty.')  

    def write(self):
        pass

    def write_to_buffer(self):
        pass
    
    def _build_search_engine(self):
        assert len(self.key_buffer) > 1
        self.search_engine = AnnoyIndex(self.dim, 'angular')
        for i, key in enumerate(key_buffer):
            self.search_engine.add_item(i, key)
        
        self.search_engine.build(self.n_trees)
    
    
    

class DND(object):
    """
    differentiable neural dictionary; should  be differentiable
    """
    def __init__(self, encode_dim, capacity, p=50, similarity_threshold=238.0, alpha=0.5):
        self.encode_dim = encode_dim
        self.capacity = capacity
        self.p = p  # num of knn
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha

        self.similarity_avg = 0.0 
        self.keys = []
        self.values = []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.keys[index], self.values[index]

    def append(self, encoded_state, n_steps_q):
        # if not self.exist_state(encoded_state): 
        self.keys.append(encoded_state)
        self.values.append(n_steps_q)
        # else:
        #     # update when state not exist 
        #     # logger.debug('state already existed')
        #     top_k_weights, top_k_index = self.query_k(encoded_state, k=1)
        #     self.values[top_k_index] = self.values[top_k_index] + self.alpha * (n_steps_q - self.values[top_k_index]) 
        # remove oldest memory if capacity is rearched
        # if len(self) >= self.capacity:
        #     self.keys.pop(0)
        #     self.values.pop(0)

        # else:
        #     _, index = self.query(encoded_state, 1)  # get the index of exist_state
        #     self.values[index] = n_steps_q

    def query_k(self, encoded_state, k):
        """
        Need to be differentiable
        args:
            encoded_state: tensor, size=(encoded_dim,), required_grad = True
            k: int, number of neighbors
        return:
            top_k_weights: weights of top k neighbors
            top_k_index: indexs of top k neighbors
        """
        weights = self.get_weights(encoded_state)
        top_k_weights, top_k_index = torch.topk(weights, k)
        return top_k_weights, top_k_index

    def exist_state(self, encoded_state):
        """
        encoded_state: tensor

        return bool 
        """
        if len(self.keys) > 0:
            keys = torch.cat(self.keys, dim=0)  # stack: stack on new axis; cat: cat on existing axis
            similarities = self.kernel_function(encoded_state, keys)
            max_state = similarities.max(-1)
            similarity = max_state[0].item()
            index = max_state[1].item()
            # logger.debug('similarity: {} encoded_state: {} closest_state: {}'.format(similarity, encoded_state, self.keys[index]))
            if similarity > self.similarity_threshold:
                return True
        return False

    def get_max_n_steps_q(self):
        assert len(self.values) > 0
        return max(self.values)

    def get_expected_n_steps_q(self, encoded_state):
        """
        produce a differentiable query
        encoded_state: tensor, require_grad = True
        """q
        assert len(self.keys) >= self.p
        queried = self.query_k(encoded_state, k=self.p)
        top_k_weights, top_k_index = queried 
        queried_values = torch.tensor(self.values, dtype=torch.float32)[top_k_index]
        n_steps_q = torch.sum(queried_values * top_k_weights).item()
        return n_steps_q

    def get_weights(self, encoded_state):
        """
        Need to be differentiable
        args:
            encoded_state: tensor, size=(encoded_dim,)
        return:
            weights: weights of each value, sum to 1; size=(keys_size,)
        """
        keys = torch.cat(self.keys, dim=0)  # stack: stack on new axis; cat: cat on existing axis
        similarities = self.kernel_function(encoded_state, keys)
        weights = similarities / torch.sum(similarities)
        return weights
    
    def save(self, output_dir, id_):
        torch.save(
            self.keys,
            os.path.join(output_dir, f'dnd_{id_}_keys.pt')
            )
        torch.save(
            self.values,
            os.path.join(output_dir, f'dnd_{id_}_values.pt')
            )

    def load(self, checkpoint_dir, id_):
        self.keys = torch.load(os.path.join(checkpoint_dir, f'dnd_{id_}_keys.pt'))
        self.values = torch.load(os.path.join(checkpoint_dir, f'dnd_{id_}_values.pt'))

    @staticmethod
    def kernel_function(encoded_state, keys):
        """
        encoded_state: tensor, size = (encoded_dim,)
        keys: tensor, size = (keys_size, encoded_dim)
        """
        difference = encoded_state - keys
        distance = difference.norm(dim=-1).pow(2)
        return 1 / (distance + 1e-3)
