"""Challenger architectures for champion-challenger evaluation against the
TCN-based VenusNet. Same input contract (dict of per-timeframe sequences),
same fusion-head design — only the per-timeframe encoder differs.

- InceptionTimeNet: multi-scale conv kernels (after tsai / InceptionTime,
  Fawaz et al. 2020) — strong general-purpose TS classifier.
- ALSTMNet: LSTM with additive attention (after Qlib's ALSTM baseline).

Run the comparison with:  python -m aethel.venus.benchmark --data ... --symbol ...
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aethel.venus.model import AttentionPool

TIMEFRAMES = ("M5", "M15", "H1")


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, out_ch, 1) if in_ch > out_ch else nn.Identity()
        bn_ch = out_ch if in_ch > out_ch else in_ch
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_ch, out_ch, k, padding="same") for k in (9, 19, 39)
        ])
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, 1),
        )
        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        out = torch.cat([conv(b) for conv in self.convs] + [self.pool_conv(x)], dim=1)
        return self.act(self.bn(out))


class InceptionEncoder(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 64, depth: int = 3):
        super().__init__()
        blocks, ch = [], n_features
        for _ in range(depth):
            blocks.append(InceptionBlock(ch, 32))
            ch = 32 * 4
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Conv1d(ch, out_dim, 1)
        self.pool = AttentionPool(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) -> (batch, out_dim)
        return self.pool(self.proj(self.blocks(x.transpose(1, 2))))


class ALSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                      # (B, T, H)
        weights = torch.softmax(self.att(out), 1)  # (B, T, 1)
        return (weights * out).sum(dim=1)


class _MultiTimeframeNet(nn.Module):
    """Shared fusion-head wrapper so every challenger sees the exact same
    multi-timeframe input contract as VenusNet."""

    def __init__(self, encoder_cls, n_features: int, hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {tf: encoder_cls(n_features) for tf in TIMEFRAMES})
        self.head = nn.Sequential(
            nn.Linear(hidden * len(TIMEFRAMES), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = torch.cat([self.encoders[tf](batch[tf]) for tf in TIMEFRAMES], dim=-1)
        return self.head(fused).squeeze(-1)


class InceptionTimeNet(_MultiTimeframeNet):
    def __init__(self, n_features: int):
        super().__init__(InceptionEncoder, n_features)


class ALSTMNet(_MultiTimeframeNet):
    def __init__(self, n_features: int):
        super().__init__(ALSTMEncoder, n_features)
