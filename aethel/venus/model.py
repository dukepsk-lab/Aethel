"""Venus — multi-timeframe CNN+TCN signal model.

Architecture (per symbol — one trained model per instrument):

    M1 sequence  -> Conv1d front-end -> TCN -> attention pool ┐
    M15 sequence -> Conv1d front-end -> TCN -> attention pool ├-> fusion MLP -> logit
    H1 sequence  -> Conv1d front-end -> TCN -> attention pool ┘

Deliberately no LSTM: TCN already covers long-range temporal dependencies
via dilated causal convolutions, and FX data at this granularity has too low
a signal-to-noise ratio to justify the extra parameters.

The raw sigmoid output is NOT used directly — it is calibrated (isotonic)
so that "confidence" is a real probability of the TP barrier being hit
before the SL barrier (see labeling.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):
    """Conv1d with left-padding only — no lookahead."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        self._pad = (kernel - 1) * dilation
        super().__init__(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self._pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.downsample(x)


class AttentionPool(nn.Module):
    """Learned weighted pooling over the time axis."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time) -> (batch, channels)
        x = x.transpose(1, 2)                       # (B, T, C)
        weights = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)


class TimeframeEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, levels: int = 4,
                 kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=1),
            nn.GELU(),
        )
        blocks = [
            TemporalBlock(hidden, hidden, kernel, dilation=2**i, dropout=dropout)
            for i in range(levels)
        ]
        self.tcn = nn.Sequential(*blocks)
        self.pool = AttentionPool(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) -> (batch, hidden)
        x = x.transpose(1, 2)
        return self.pool(self.tcn(self.front(x)))


class VenusNet(nn.Module):
    """Per-symbol model. Input: dict of (batch, time, features) tensors keyed
    by timeframe. Output: uncalibrated logit for P(TP before SL)."""

    TIMEFRAMES = ("M1", "M15", "H1")

    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {tf: TimeframeEncoder(n_features, hidden, dropout=dropout)
             for tf in self.TIMEFRAMES}
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * len(self.TIMEFRAMES), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = torch.cat([self.encoders[tf](batch[tf]) for tf in self.TIMEFRAMES], dim=-1)
        return self.head(fused).squeeze(-1)
