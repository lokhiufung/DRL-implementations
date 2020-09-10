# from torch.nn import functional as F

from drl.approximators.dnn.baseline import BaselineNetwork
from drl.optimizers import 


class BaselineApproximator:
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        self.network = BaselineNetwork(input_dim, output_dim, hidden_dim)
        self.optimizer = 

    def fit(self, x, y):
        return self.net