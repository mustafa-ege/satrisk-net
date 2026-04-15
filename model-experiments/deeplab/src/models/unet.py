from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2 block used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling step: maxpool followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsampling step: bilinear resize + concat with skip + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution translating features into logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Classic U-Net encoder-decoder with configurable feature widths."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        features: Sequence[int] = (64, 128, 256, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0], dropout)

        downs = []
        ups = []
        for idx in range(1, len(features)):
            downs.append(Down(features[idx - 1], features[idx], dropout))

        for idx in range(len(features) - 1, 0, -1):
            ups.append(Up(features[idx] + features[idx - 1], features[idx - 1], dropout))

        self.downs = nn.ModuleList(downs)
        self.ups = nn.ModuleList(ups)
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        # First block does not downsample, so store its output
        x = self.inc(x)
        skip_connections.append(x)

        for down in self.downs:
            # Each downsample reduces resolution and appends to skip list
            x = down(x)
            skip_connections.append(x)

        x = skip_connections.pop()

        for up in self.ups:
            skip = skip_connections.pop()
            x = up(x, skip)

        logits = self.outc(x)
        return logits

