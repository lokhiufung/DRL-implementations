from drl.blocks.memory.dnd import DifferentiableNeuralDictionary
from drl.blocks.memory.replay_buffer import ReplayBuffer
from drl.datasets.replay_buffer_dataset import LowDimReplayBufferDataset


class NECAgent(object):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dnd = {}
        for action in range(self._env.n_actions):
            self.dnd[action] = DifferentiableNeuralDictionary(
                dim=cfg.dim
            )

