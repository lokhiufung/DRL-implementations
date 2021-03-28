import torch.nn as nn 



class ConvEncoder(nn.Module):
    """
    Convlution encoder follows: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(
            self.input_channels,
            16,
            kernel_size=8,
            stride=4,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=4,
            stride=2,
        )
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(
            out_features=256
        )
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.fc(x)
        x = self.relu3(x)
        return x

