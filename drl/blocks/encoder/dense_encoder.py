import torch.nn as nn
import torch.nn.functional as F

# from drl.core.modules import TrainableModule

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim, *args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim=128):
        super(DenseEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        layers = []
        for i in range(self.n_layers):
            if i == 0:
                layers.append(LinearLayer(self.input_dim, self.hidden_dim))
            else:
                layers.append(LinearLayer(self.hidden_dim, self.hidden_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    import torch

    x = torch.normal(0, 1, size=(4, 4))
    encoder = DenseEncoder(2, 4)
    print(encoder)
    y = encoder(x)
    print(y)