from drl.agents.dqn_agent import DQNAgent
from drl.core.decorators import take_agent_step
from drl.blocks.memory.replay_buffer import ReplayBuffer
from drl.datasets.replay_buffer_dataset import LowDimReplayBufferDataset


class NECAgent(DQNAgent):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dnd = DifferentiableNeuralDict()

