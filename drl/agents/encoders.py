from torch import nn


class FullyConnectedEncoder(nn.Module):
    def __init__(self, input_dim, encode_dim, hidden_dim=64):
        super(FullyConnectedEncoder, self).__init__()
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        self.hidden_dim = 64

        # encoder 
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.encode_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x
