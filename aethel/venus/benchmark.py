"""Champion-challenger benchmark: TCN (champion) vs InceptionTime vs ALSTM
on identical purged walk-forward folds.

    python -m aethel.venus.benchmark --symbol EURUSD --data data/EURUSD_m5.parquet

Promotion rule: a challenger replaces the champion only if it wins on OOF
AUC across a majority of folds AND on the trade-level avg-R — single-metric
wins on one fold are noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aethel.venus.features import FEATURE_COLUMNS
from aethel.venus.train import SEQ, build_dataset
from aethel.venus.validation import PurgedWalkForward


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def benchmark(m5: pd.DataFrame, epochs: int = 8, n_splits: int = 4,
              threshold: float = 0.65) -> dict:
    import torch

    from aethel.venus.challengers import ALSTMNet, InceptionTimeNet
    from aethel.venus.model import VenusNet

    contenders = {
        "TCN (champion)": VenusNet,
        "InceptionTime": InceptionTimeNet,
        "ALSTM": ALSTMNet,
    }

    x, y, t_end, _meta = build_dataset(m5)
    n = len(y)
    splitter = PurgedWalkForward(n_splits=n_splits, embargo_bars=100)
    folds = [(tr, te) for tr, te in splitter.split(n, t_end) if len(tr) >= 200]

    results: dict[str, dict] = {}
    for name, net_cls in contenders.items():
        aucs, avg_rs, trade_counts = [], [], []
        for tr, te in folds:
            torch.manual_seed(7)  # same init luck for every contender
            model = net_cls(n_features=len(FEATURE_COLUMNS))
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            for _ in range(epochs):
                perm = np.random.permutation(tr)
                for i in range(0, len(perm), 256):
                    idx = perm[i:i + 256]
                    opt.zero_grad()
                    loss = loss_fn(model({tf: x[tf][idx] for tf in SEQ}), y[idx])
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model({tf: x[tf][te] for tf in SEQ})).numpy()
            y_te = y[te].numpy()
            aucs.append(_auc(y_te, p))
            taken = p >= threshold
            if taken.any():
                trades_r = np.where(y_te[taken] == 1, 2.0, -1.0) - 0.10
                avg_rs.append(float(trades_r.mean()))
                trade_counts.append(int(taken.sum()))
        results[name] = {
            "auc_per_fold": [round(a, 4) for a in aucs],
            "mean_auc": round(float(np.nanmean(aucs)), 4),
            "avg_r": round(float(np.mean(avg_rs)), 3) if avg_rs else None,
            "trades": int(np.sum(trade_counts)) if trade_counts else 0,
            "params": sum(p.numel() for p in net_cls(len(FEATURE_COLUMNS)).parameters()),
        }
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--splits", type=int, default=4)
    args = p.parse_args()
    report = benchmark(pd.read_parquet(args.data), epochs=args.epochs,
                       n_splits=args.splits)
    print(json.dumps({"symbol": args.symbol, "contenders": report}, indent=2))
