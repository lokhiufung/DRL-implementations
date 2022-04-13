import os
import time
import math
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from annoy import AnnoyIndex


__all__ = ['DifferentiableNeuralDictionary']


class DifferentiableNeuralDictionary(nn.Module):
    def __init__(
        self,
        n_actions,
        dim,
        kernel_function='inverse_distance',
        capacity: int=10000,
        n_neighbors: int=50,
        n_trees: int=10,
        delta: float=1e-3,
        score_threshold: float=1e-3,  # reminder: from 0 to 2, the smaller, the closer
        alpha: float=1.0,
    ):
        super().__init__()
        self.n_actions = n_actions

        self.capacity = capacity
        self.dim = dim
        self.n_trees = n_trees
        self.delta = delta
        self.n_neighbors = n_neighbors
        self.kernel_function = kernel_function
        self.score_threshold = score_threshold
        self.alpha = alpha

        self.dnds = []
        for _ in range(self.n_actions):
            self.dnds.append(_DifferentiableNeuralDictionary(
                dim=self.dim,
                kernel_function=self.kernel_function,
                capacity=self.capacity,
                n_neighbors=self.n_neighbors,
                n_trees=self.n_trees,
                delta=self.delta,
                score_threshold=self.score_threshold,
                alpha=self.alpha,
            ))
        self.dnds = nn.ModuleList(self.dnds)

    def reset(self):
        for action in range(self.n_actions):
            self.dnds[action].reset()

    def get_len(self, action):
        return len(self.dnds[action])

    def lookup(
        self,
        key: Union[torch.Tensor, np.ndarray],
        return_tensor=True,
        return_all_values=False
    ) -> Union[Tuple[torch.Tensor, torch.LongTensor], Tuple[np.ndarray, np.ndarray]]:
        
        values = []
        indexes = []  # retrieved indexes
        scores = []
        for i in range(self.n_actions):
            value, index, score = self.dnds[i].lookup(key, return_tensor)
            values.append(value)
            indexes.append(index)
            scores.append(score)

        # values = [self.dnds[i].lookup(key, return_tensor) for i in range(self.n_actions)]
        if return_tensor:
            values = torch.cat(values, dim=-1)  # concat to a single tensor (batch_size, n_actions) TODO: do not work with numpy array
            max_values, actions = torch.max(values, dim=-1, keepdim=True)
        else:
            raise NotImplementedError('lookup() for numpy has not been implemented yet.')
        if return_all_values:
            return (values, actions, indexes, scores) if return_tensor else (values.cpu().numpy(), actions.cpu().numpy(), indexes, scores)
        else:
            return (max_values, actions, indexes, scores) if return_tensor else (max_values.cpu().numpy(), actions.cpu().numpy(), indexes, scores)

    def forward(self, key: torch.Tensor, return_all_values=False) -> Tuple[torch.Tensor, torch.LongTensor, List[torch.LongTensor]]:
        values = []
        indexes = []  # retrieved indexes e.g [torch.LongTensor([1, 2, 3 ....50]), ...]
        scores = []
        for i in range(self.n_actions):
            value, index, score = self.dnds[i](key)

            values.append(value)
            indexes.append(index)
            scores.append(score)

        # values_indexes = [self.dnds[i].lookup(key, return_tensor=True) for i in range(self.n_actions)]
        values = torch.cat([value for value in values], dim=-1)  # concat to a single tensor (batch_size, n_actions)
        # print('values: ', values.size())
        max_values, actions = torch.max(values, dim=-1, keepdim=True)
        # reminder: indexes and scores only contain index and score with the BEST action
        indexes = [indexes[action] for action in actions]
        scores = [scores[action] for action in actions]
        if return_all_values:
            return values, actions, indexes, scores
        else:
            return max_values, actions, indexes, scores

    def write(self):
        for i in range(self.n_actions):
            self.dnds[i].write()

    def write_to_buffer(self, action, key, value):
        self.dnds[action].write_to_buffer(key, value)

    def update_to_buffer(self, actions, keys: List[torch.Tensor], values: List[float]):
        action_group = {}
        for i, action in enumerate(actions):
            if action not in action_group:
                action_group[action] = {}
                action_group[action]['values'] = []
                action_group[action]['keys'] = []
                
            action_group[action]['values'].append(values[i])
            action_group[action]['keys'].append(keys[i])

        for action, group in action_group.items():
            self.dnds[action].update_to_buffer(
                keys=group['keys'],
                values=group['values'],
            )
    
    def get_max_value(self):
        """return the largest value over all dnds for all actions
        """
        if self.is_ready():
            max_values = [dnd.get_max_value() for dnd in self.dnds]
            max_value = max(max_values)
        else:
            max_value = torch.tensor([[0.0]], dtype=torch.float)
        return max_value

    def is_ready(self):
        statuses = [dnd.is_ready() for dnd in self.dnds]
        if all(statuses):
            return True
        else:
            return False

class _DifferentiableNeuralDictionary(nn.Module):
    def __init__(
        self,
        dim,
        kernel_function='inverse_distance',
        capacity: int=10000,
        n_neighbors: int=50,
        n_trees: int=10,
        delta: float=1e-3,
        score_threshold: float=0.01,
        alpha: float=1.0,
    ):
        super().__init__()

        self.capacity = capacity
        self.dim = dim
        self.n_trees = n_trees
        self.delta = delta
        self.n_neighbors = n_neighbors
        self.kernel_function = kernel_function
        self.score_threshold = score_threshold
        self.alpha = alpha
        # `self.keys`, `self.values` are for tensor storing
        self.keys = []
        self.values = []
        # `self.key_buffer`, `self.value_buffer` are for building search engine
        self.key_buffer = []
        self.value_buffer = []
        self.last_visit_time = {}  # last visit time indexed by the index

        self.search_engine = None  # will be initialized after getting samples in buffer

    def reset(self):
        # TODO: not good
        self.keys = nn.ParameterList([nn.Parameter(torch.from_numpy(key).view(1, -1), requires_grad=True) for key in []])
        self.values = nn.ParameterList([nn.Parameter(torch.from_numpy(value).view(1, -1), requires_grad=True) for value in []])

        self.key_buffer = []
        self.value_buffer = []
        self.last_visit_time = {}  # last visit time indexed by the index

        self.search_engine = None  # will be initialized after getting samples in buffer

    def __len__(self):
        return len(self.keys)

    def is_ready(self):
        # check if there is the search_engine 
        return True if self.search_engine is not None else False

    def lookup(self, key: Union[torch.Tensor, np.ndarray], return_tensor=True) -> Tuple[Union[torch.Tensor, np.ndarray], List[List[int]]]:
        if isinstance(key, np.ndarray):
            key = torch.from_numpy(key).view(1, -1)
        with torch.no_grad():
            value, index, score = self.forward(key)
            if not return_tensor:
                value = value.cpu().numpy()
                index = [idx.cpu().numpy() for idx in index]
                score = [sc.cpu().numpy for sc in score]
        return value, index, score

    def forward(self, key: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        if self.search_engine is not None:
            batch_size = key.size(0)  # get batch_size from `key` Tensor

            search_keys = key.detach().cpu().numpy()  # need to be detached before transforming to numpy
            retrieved_keys = []
            retrieved_values = []
            retrieved_indexes = []
            retrieved_scores = []
            for i in range(batch_size):
                indexes, distances = self.search_engine.get_nns_by_vector(
                    search_keys[i],
                    n=self.n_neighbors,
                    include_distances=True,
                )
                retrieved_keys.append(torch.cat([self.keys[index] for index in indexes], dim=0).view(1, min(self.n_neighbors, len(indexes)), self.dim))
                retrieved_values.append(torch.cat([self.values[index] for index in indexes], dim=0).view(1, min(self.n_neighbors, len(indexes))))  # (self.n_neighbors, 1)
                retrieved_indexes.append(torch.tensor(indexes, dtype=torch.long))
                retrieved_scores.append(torch.tensor(distances, dtype=torch.float32))

                # update indexes last visit time
                visit_time = time.perf_counter()
                for index in indexes:
                    self.last_visit_time[index] = visit_time

            retrieved_keys = torch.cat(retrieved_keys, dim=0)
            retrieved_values = torch.cat(retrieved_values, dim=0)
            # print('key: ', key.size())
            # print('retrieved_keys: ', retrieved_keys.size())
            # print('retrieved_values: ', retrieved_values.size())
            key = key.view(batch_size, 1, self.dim)

            if self.kernel_function == 'inverse_distance':
                weights = 1 / (torch.sum((key - retrieved_keys)**2, dim=-1) + self.delta)  # (self.batch_size, n_neighbors, self.dim)
                weights_total = torch.sum(weights, dim=-1, keepdim=True)  # (self.batch_size, self.n_neighbors)
                # print('key - retrieved_keys: ', (key - retrieved_keys).size())
                # print('weights: ', weights.size())
                # print('weights_total: ', weights_total.size())
                output_value = torch.sum(weights * retrieved_values, dim=-1, keepdim=True)
                # print(output_value.size())
                output_value = output_value / weights_total  # (self.batch_size, 1)
            else:
                raise NotImplementedError('Only `inverse_distance` kernel function is supported.') 
            return output_value, retrieved_indexes, retrieved_scores
        else:
            raise AttributeError('Plesae make sure `self.key_buffer` and `self.value_buffer` are not empty.')  

    def write(self):
        # update keys and values as the parameters
        # nn.ParameterList
        self.keys = nn.ParameterList([nn.Parameter(torch.from_numpy(key).view(1, -1), requires_grad=True) for key in self.key_buffer])
        self.values = nn.ParameterList([nn.Parameter(torch.from_numpy(value).view(1, -1), requires_grad=True) for value in self.value_buffer])

        # build new search engine with latest `self.key_buffer` and `self.value_buffer`
        self._build_search_engine()

    def write_to_buffer(self, key: torch.Tensor, value: torch.Tensor):
        key = key.detach().cpu().numpy()
        value = value.detach().cpu().numpy()

        # reduce the null dim
        key = np.squeeze(key)
        value = np.squeeze(value)

        # when reached the max capacity, remove the oldest index before writing to the buffer
        if len(self.key_buffer) >= self.capacity:
            oldest_index = None
            smallest_visit_time = np.inf  # initialize visit time 
            for index, last_visit_time in self.last_visit_time.items():
                if last_visit_time < smallest_visit_time:  # reminder: the older the smaller last visit_time
                    oldest_index = index
            
            # remove the oldest index
            self.key_buffer.pop(oldest_index)
            self.value_buffer.pop(oldest_index)
        
        self.key_buffer.append(key)
        self.value_buffer.append(value)

    def update_to_buffer(self, keys: List[torch.TensorType], values: List[torch.TensorType]):
        for key, value in zip(keys, values):

            value_prev, closest_idx, score = self.lookup(key)
            
            # score_ = score  # TODO
            score = min(score[0]).item()  # reminder: score is a tensor. The lower, the closer
            closest_idx = closest_idx[0][0] 
            if score < self.score_threshold:  # TODO: change all score to distance or convert the distance to score
                # print('update_to_buffer() score: {} score_: {}'.format(score, score_))
                # if the index is already in dnd, update it using q update
                # closest_idx = sorted([(idx, sc) for idx, sc in zip(index, score)], key=lambda x: x[1], reverse=True)[0]  # get the index with largest score
                # with torch.no_grad():  # reminder: in-place operation is not allowed for Variable that requires grad
                # reminder: value is a tensor
                self.value_buffer[closest_idx] += self.alpha * (value.item() - self.value_buffer[closest_idx])  # TODO: test with this, values -> value_buffer, table update only at write()
                
            else:
                # else just write to the buffer
                self.write_to_buffer(key, value)

    def get_max_value(self):
        """return the larget value from the self.values
        """
        max_value = max(self.values)
        return max_value

    def _build_search_engine(self):
        assert len(self.key_buffer) > 1, ''  # TODO: may add a minimum capacity for writing buffer to search engine
        self.search_engine = AnnoyIndex(self.dim, 'angular')
        for i, key in enumerate(self.key_buffer):
            self.search_engine.add_item(i, key)
        
        self.search_engine.build(self.n_trees)
    
    
if __name__ == '__main__':
    import time


    dnd = DifferentiableNeuralDictionary(
        dim=64
    )

    for _ in range(10000):
        dnd.write_to_buffer(key=torch.normal(mean=0.0, std=1.0, size=(64,)), value=torch.tensor(0.3))
    
    dnd.write()

    start = time.perf_counter()
    output_value = dnd.lookup(
        key=torch.normal(mean=0.0, std=1.0, size=(1, 64)),
    )
    end = time.perf_counter()
    print(output_value.size())
    print('total time: {}'.format(end - start))