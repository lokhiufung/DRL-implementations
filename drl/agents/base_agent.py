import torch

from drl import approximators

class BaseAgent:
    def __init__(self, input_dim, output_dim, mode, batch_size, approximator, *args, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        if isinstance(approximator, str):
            self.approximator = getattr(approximators, approximator)(
                input_dim, output_dim, **args, **kwargs
            )
        else:
            self.approximator = approximator

    def greedy_infer(self, state):
        """greedy infer

        Args:
            state (nd.array/tensor): state tensor 
        """
        raise NotImplementedError

    def epsilon_greedy_infer(self, state):
        raise NotImplementedError

    def save_checkpoint(self, output_dir):
        """
        save checkpoint for restore training
        """
        ckpt = dict()
        for name, network in self.networks.items():
            ckpt[name] = network.state_dict()
        ckpt['optimizer'] = self.optimizer.state_dict()
        torch.save(ckpt)

    def save_network(self, output_dir):
        """
        save checkpoint for restore training
        """
        ckpt = dict()
        for name, network in self.networks.items():
            ckpt[name] = network.state_dict()
        torch.save(ckpt)

