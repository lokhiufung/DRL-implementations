import torch.nn as nn

# from drl.core.modules import TrainableModule


class DenseEncoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        for i in self.n_layers - 1:
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            elif i == self.n_layers - 1:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU)
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

