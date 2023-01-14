import torch
from torch import nn
import env_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConnextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 2

        self.width = env_config.config['width']

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ).to(device)

        self.policy_network = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(84, self.width),
        ).to(device)


        self.value_network = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(42, 1),
            nn.Tanh()
        ).to(device)



    def forward(self, x):
        x = self.cnn(x)

        action_distribution = self.policy_network(x)
        value = self.value_network(x)

        return action_distribution, value
