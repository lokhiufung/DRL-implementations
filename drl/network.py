# import torch.nn as nn 
from drl.core.modules import TrainableModule

class Network(nn.Module):
    # def __init__(self, network_params):
    def _create_network_graph(self, network_params):
        for module_name, module_params in network_params.items():
        
    def clone(self, network):
        pass

    def clip_gradient(self):
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)  # gradient cliping |grad| < = 1, clamp_ in-place original tensor, .data to get underlying tensor of a variable
        