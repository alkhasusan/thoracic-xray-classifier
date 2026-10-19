"""
attention.py

scSE: concurrent Spatial and Channel Squeeze & Excitation
(Roy, Navab, Wachinger, 2018).

A drop-in attention block that takes a feature map [B, C, H, W] and returns
a re-weighted feature map of the same shape. It combines two branches:

  cSE (channel squeeze & excitation): "WHICH channels matter?"
      global average pool -> 1x1 conv (C -> C/r) -> ReLU -> 1x1 conv (C/r -> C)
      -> sigmoid -> multiply with the input channel-wise.

  sSE (spatial squeeze & excitation): "WHERE in the image matters?"
      1x1 conv (C -> 1) -> sigmoid -> multiply with the input pixel-wise.

The two re-weighted maps are combined by element-wise addition (as in the paper).

This file only defines the module. It is NOT attached to any backbone yet:
attach it to the winning backbone only, after the backbone comparison is done.

Usage:
    attn = SCSE(channels=1024, reduction=16)
    y = attn(x)          # y.shape == x.shape
"""

import torch.nn as nn


class ChannelSE(nn.Module):
    """cSE branch: channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class SpatialSE(nn.Module):
    """sSE branch: spatial attention."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.conv(x)


class SCSE(nn.Module):
    """scSE block: channel SE + spatial SE, combined by addition."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cse = ChannelSE(channels, reduction)
        self.sse = SpatialSE(channels)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


if __name__ == "__main__":
    import torch

    x = torch.randn(2, 256, 14, 14)
    y = SCSE(256)(x)
    print("scSE output shape (expect [2, 256, 14, 14]):", tuple(y.shape))