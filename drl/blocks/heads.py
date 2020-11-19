from drl.core.modules import TrainableModule

from torch import nn


class ValueHead(TrainableModule):
    def __init__(self, output_dim, hidden_dim, activation=None):
        super(ValueHead, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim =hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
        )
    
    def forward(self, embedding):
        return self.model(embedding)


class SoftmaxPolicyHead(TrainableModule):
    def __init__(self, output_dim, hidden_dim):
        super(ValueHead, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim =hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax()
        )
    
    def forward(self, embedding):
        return self.model(embedding)


