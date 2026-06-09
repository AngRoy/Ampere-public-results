from __future__ import annotations

from ampere.neural.utils import require_torch

torch = require_torch()
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class CausalTCNReconstructor(nn.Module):
    """Small causal TCN that predicts all branch powers at each window timestep."""

    def __init__(
        self,
        *,
        input_channels: int,
        branch_count: int,
        hidden_channels: int = 32,
        layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.05,
        use_calibration: bool = False,
        residual_start_channel: int | None = None,
    ):
        super().__init__()
        self.branch_count = int(branch_count)
        self.residual_start_channel = residual_start_channel
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        blocks = []
        for layer in range(layers):
            blocks.append(
                TCNBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.output_projection = nn.Conv1d(hidden_channels, branch_count, kernel_size=1)
        self.use_calibration = bool(use_calibration)
        if self.use_calibration:
            self.branch_scale = nn.Parameter(torch.ones(branch_count, 1))
            self.branch_offset = nn.Parameter(torch.zeros(branch_count, 1))
        else:
            self.register_parameter("branch_scale", None)
            self.register_parameter("branch_offset", None)

    def forward(self, x):
        y = self.output_projection(self.tcn(torch.relu(self.input_projection(x))))
        if self.residual_start_channel is not None:
            start = int(self.residual_start_channel)
            stop = start + self.branch_count
            y = y + x[:, start:stop, :]
        if self.use_calibration:
            y = y * self.branch_scale.unsqueeze(0) + self.branch_offset.unsqueeze(0)
        return y

    def calibration_regularization(self):
        if not self.use_calibration:
            return torch.as_tensor(0.0, device=next(self.parameters()).device)
        return ((self.branch_scale - 1.0) ** 2).mean() + (self.branch_offset**2).mean()
