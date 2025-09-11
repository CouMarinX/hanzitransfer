"""A tiny conditional UNet for raster fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, down: bool = True) -> None:
        super().__init__()
        stride = 2 if down else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class CondUNet(nn.Module):
    """Minimal UNet that conditions on extra channels."""

    def __init__(self, in_channels: int, base_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.middle = ConvBlock(base_channels * 4, base_channels * 4, down=False)
        self.dec2 = UpBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = UpBlock(base_channels * 2 + base_channels, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.dropout(e1))
        e3 = self.enc3(self.dropout(e2))
        m = self.middle(self.dropout(e3))
        d2 = self.dec2(m, e2)
        d1 = self.dec1(d2, e1)
        if noise > 0:
            d1 = d1 + noise * torch.randn_like(d1)
        out = torch.sigmoid(self.out(d1))
        return out
