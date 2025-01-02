import torch
import torch.nn as nn
import torch.nn.functional as F
from node import NUM_INPUT_CHANNELS

class AGZNetwork(nn.Module):
    def __init__(self, num_res_blocks: int, num_channels: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(NUM_INPUT_CHANNELS, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, num_channels) for _ in range(num_res_blocks)]
        )
        self.policy_head_fc = PolicyNet(num_channels)

        self.value_head_fc = ValueNet(num_channels)

    def forward(self, x, mask):
        """
        This depends considerably on the design of your network.
        This is called when you call network(x) where x can be whatever you want.
        """
        pi = self.policy_head_fc(self.res_blocks(self.conv_block(x)),mask)
        val = self.value_head_fc(self.res_blocks(self.conv_block(x)))
        return val, pi
        #raise NotImplementedError()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state'
    """
    def __init__(self, in_channels, out_channels=82):
        super(PolicyNet, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(out_channels - 1, out_channels)

    def forward(self, x, mask):
        """
        x : feature maps from the ResidualBlocks
        probas : a 9x9 + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.out_channels - 1)
        x = self.fc(x)
        x = mask * x
        probas = self.logsoftmax(x).exp()
        return probas


class ValueNet(nn.Module):
    """
    This network is used to predict which player is more likely to win given the input 'state'
    The output is a continuous variable, between -1 and 1. 
    """
    def __init__(self, in_channels, out_channels=82):
        super(ValueNet, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(out_channels - 1, 256)
        self.fc2 = nn.Linear(256, 1)
        

    def forward(self, x):
        """
        x : feature maps extracted from the state by the ResidualBlocks
        winning : probability of the current agent winning the game
                  considering the actual state of the board
        """
 
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.out_channels - 1)
        x = F.relu(self.fc1(x))
        #x = mask * x
        winning = torch.tanh(self.fc2(x))
        return winning