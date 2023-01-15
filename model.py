import torch
from torch import nn
import env_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConnextNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = 2

        self.width = env_config.config['width']

        self.out_channel = 64

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.out_channel, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(),
        ).to(device)

        self.res_block1 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block2 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block3 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block4 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block5 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block5 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block6 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block7 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block8 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block9 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)
        self.res_block10 = ResBlock(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1)

        self.policy_network = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channel, out_channels=2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(112, self.width),
        ).to(device)


        self.value_network = nn.Sequential(
            nn.Conv2d(in_channels=self.out_channel, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(56, 1),
            nn.Tanh()
        ).to(device)



    def forward(self, x):
        x = self.cnn(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.res_block9(x)
        x = self.res_block10(x)

        action_distribution = self.policy_network(x)
        value = self.value_network(x)

        return action_distribution, value


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ).to(device)

        self.relu = nn.ReLU().to(device)
    
    def forward(self, x):
        identity = x
        cnn_result = self.cnn(x)

        output = identity + cnn_result
        output = self.relu(output)
        return output

