"""Vectorized walk-forward backtest for Venus.

Usage:
    python -m aethel.venus.backtest --symbol EURUSD --data data/EURUSD_m5.parquet

For each purged walk-forward fold: train on the past, predict the test fold
out-of-sample, calibrate on prior OOF predictions, then simulate taking every
signal whose calibrated confidence clears the threshold. Outcomes come from
the triple-barrier labels (win = +tp_mult R, loss = -sl_mult R) minus a
spread cost, expressed in R-multiples.

This evaluates VENUS ONLY. The agent layer cannot be backtested historically
(API cost + the LLMs' training data contains your test period) — validate it
in shadow mode on a live demo feed.
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


def run_backtest(
    m5: pd.DataFrame,
    threshold: float = 0.65,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    spread_cost_r: float = 0.10,
    epochs: int = 10,
    n_splits: int = 5,
) -> dict:
    import torch

    from aethel.venus.calibration import Calibrator
    from aethel.venus.model import VenusNet

    x, y, t_end = build_dataset(m5)
    n = len(y)

    fold_results = []
    all_trades_r: list[float] = []
    prior_raw: list[np.ndarray] = []
    prior_y: list[np.ndarray] = []

    splitter = PurgedWalkForward(n_splits=n_splits, embargo_bars=100)
    for fold, (tr, te) in enumerate(splitter.split(n, t_end)):
        if len(tr) < 200:
            continue
        model = VenusNet(n_features=len(FEATURE_COLUMNS))
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
            raw = torch.sigmoid(model({tf: x[tf][te] for tf in SEQ})).numpy()
        y_te = y[te].numpy()

        # calibrate on OOF predictions from PREVIOUS folds only (no leakage)
        if prior_raw:
            cal = Calibrator()
            cal.fit(np.concatenate(prior_raw), np.concatenate(prior_y))
            conf = cal.transform(raw)
        else:
            conf = raw  # first fold: uncalibrated, reported but typical to discard
        prior_raw.append(raw)
        prior_y.append(y_te)

        taken = conf >= threshold
        trades_r = np.where(y_te[taken] == 1, tp_mult, -sl_mult) - spread_cost_r
        all_trades_r.extend(trades_r.tolist())
        fold_results.append({
            "fold": fold,
            "test_samples": int(len(te)),
            "signals_taken": int(taken.sum()),
            "win_rate": round(float(y_te[taken].mean()), 3) if taken.any() else None,
            "avg_r": round(float(trades_r.mean()), 3) if taken.any() else None,
            "base_rate": round(float(y_te.mean()), 3),
        })

    trades = np.array(all_trades_r)
    equity = trades.cumsum() if len(trades) else np.array([0.0])
    peak = np.maximum.accumulate(equity)
    summary = {
        "threshold": threshold,
        "total_trades": int(len(trades)),
        "win_rate": round(float((trades > 0).mean()), 3) if len(trades) else None,
        "total_r": round(float(trades.sum()), 2),
        "avg_r_per_trade": round(float(trades.mean()), 3) if len(trades) else None,
        "max_drawdown_r": round(float((equity - peak).min()), 2),
        "profit_factor": (
            round(float(trades[trades > 0].sum() / -trades[trades < 0].sum()), 2)
            if (trades < 0).any() and (trades > 0).any() else None
        ),
        "folds": fold_results,
    }
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    m5 = pd.read_parquet(args.data)
    report = run_backtest(m5, threshold=args.threshold, epochs=args.epochs)
    print(json.dumps({"symbol": args.symbol, **report}, indent=2))
