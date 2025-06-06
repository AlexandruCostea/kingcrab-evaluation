import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layers import ClippedReLU


class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.activation = ClippedReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.activation(x)
        return x


class DepthwiseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DWConvBlock(15, 16),
            DWConvBlock(16, 24),
            DWConvBlock(24, 32),
            DWConvBlock(32, 48),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, 32),
            ClippedReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.net(x)
        return self.head(x)