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
        n_neighbors: int=50,
        n_trees: int=10
    ):
        super().__init__()

        self.capacity = capacity
        self.dim = dim
        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.kernel_function = kernel_function
        # `self.keys`, `self.values` are for tensor storing
        self.keys = []
        self.values = []
        # `self.key_buffer`, `self.value_buffer` are for building search engine
        self.key_buffer = []
        self.value_buffer = []

        self.search_engine = None  # will be initialized after getting samples in buffer

    def forward(self, key):
        value = self.lookup(key)
        return value

    def lookup(self, key: torch.Tensor, n_neighbors: int=50, delta: float=1e-3):
        if self.search_engine is not None:
            batch_size = key.size(0)  # get batch_size from `key` Tensor

            search_keys = key.cpu().numpy()
            retrieved_keys = []
            retrieved_values = []
            for i in range(batch_size):
                indexes, distances = self.search_engine.get_nns_by_vector(
                    search_keys[i],
                    n=self.n_neighbors,
                    include_distances=True,
                )
                retrieved_keys.append(torch.cat([self.keys[index] for index in indexes], dim=0).view(1, self.n_neighbors, self.dim))
                retrieved_values.append(torch.cat([self.values[index] for index in indexes], dim=0).view(1, self.n_neighbors))  # (self.n_neighbors, 1)

            retrieved_keys = torch.cat(retrieved_keys, dim=0)
            retrieved_values = torch.cat(retrieved_values, dim=0)
            print('key: ', key.size())
            print('retrieved_keys: ', retrieved_keys.size())
            print('retrieved_values: ', retrieved_values.size())
            key = key.view(batch_size, 1, self.dim)

            if self.kernel_function == 'inverse_distance':
                weights = 1 / (torch.sum((key - retrieved_keys)**2, dim=-1) + delta)  # (self.batch_size, n_neighbors, self.dim)
                weights_total = torch.sum(weights, dim=-1, keepdim=True)  # (self.batch_size, self.n_neighbors)
                print('key - retrieved_keys: ', (key - retrieved_keys).size())
                print('weights: ', weights.size())
                print('weights_total: ', weights_total.size())
                output_value = torch.matmul(weights, retrieved_values) / weights_total  # (self.batch_size, 1)
            else:
                raise NotImplementedError('Only `inverse_distance` kernel function is supported.') 
            return output_value
        else:
            raise AttributeError('Plesae make sure `self.key_buffer` and `self.value_buffer` are not empty.')  

    def write(self):
        # update keys and values as the parameters
        self.keys = nn.ParameterList([nn.Parameter(torch.from_numpy(key).view(1, -1), requires_grad=True) for key in self.key_buffer])
        self.values = nn.ParameterList([nn.Parameter(torch.from_numpy(value).view(1, -1), requires_grad=True) for value in self.value_buffer])

        # build new search engine with latest `self.key_buffer` and `self.value_buffer`
        self._build_search_engine()

    def write_to_buffer(self, key: torch.Tensor, value: torch.Tensor):
        key = key.detach().cpu().numpy()
        value = value.detach().cpu().numpy()

        self.key_buffer.append(key)
        self.value_buffer.append(value)
    
    def _build_search_engine(self):
        assert len(self.key_buffer) > 1
        self.search_engine = AnnoyIndex(self.dim, 'angular')
        for i, key in enumerate(self.key_buffer):
            self.search_engine.add_item(i, key)
        
        self.search_engine.build(self.n_trees)
    
    
if __name__ == '__main__':
    dnd = DifferentiableNeuralDict(
        dim=64
    )

    for _ in range(60):
        dnd.write_to_buffer(key=torch.normal(mean=0.0, std=1.0, size=(64,)), value=torch.tensor(0.3))
    
    dnd.write()

    output_value = dnd.lookup(
        key=torch.normal(mean=0.0, std=1.0, size=(10, 64)),
    )
    print(output_value)
