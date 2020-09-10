from torch import nn


class BaselineNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.output_layer(x)  # linear outputs correspond to q-value of each action
        return x

