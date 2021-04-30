
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from drl.blocks.encoder.dense_encoder import DenseEncoder
from drl.blocks.memory.dnd import DifferentiableNeuralDictionary
from drl.blocks.heads import NoOpValueHead

class NECModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config):
        super().__init__(obs_space, action_space, num_outputs, model_config)

        nn.Module.__init__()

        self.encoder = DenseEncoder()
        self.dnd = DifferentiableNeuralDictionary(
            n_actions=action_space.action,
        )
        self.value_head = NoOpValueHead()

    def forward(self, input_dict, state, seq_len):
        pass

