"""Offline training scaffold — purged walk-forward, per-symbol models,
isotonic calibration on out-of-fold predictions, champion artifact export.

Usage:
    python -m aethel.venus.train --symbol EURUSD --data data/EURUSD.parquet

The data file must contain M5 OHLCV with a UTC DatetimeIndex; M15/H1 are
resampled from it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aethel.venus.calibration import Calibrator
from aethel.venus.features import FEATURE_COLUMNS, compute_features
from aethel.venus.labeling import triple_barrier_labels
from aethel.venus.validation import PurgedWalkForward

SEQ = {"M5": 120, "M15": 96, "H1": 72}


def build_dataset(m5: pd.DataFrame):
    import torch

    frames = {
        "M5": m5,
        "M15": m5.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "tick_volume": "sum"}).dropna(),
        "H1": m5.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "tick_volume": "sum"}).dropna(),
    }
    feats = {tf: compute_features(df).ffill().fillna(0.0) for tf, df in frames.items()}

    # Primary direction: H1 EMA-50 trend, sampled per M15 bar; label whether
    # a trade in that direction hits TP (2 vol) before SL (1 vol).
    base = frames["M15"]
    h1_dist = feats["H1"]["dist_ema50"].reindex(base.index, method="ffill")
    direction = np.sign(h1_dist).fillna(0).astype(int)
    labels = triple_barrier_labels(base, direction, tp_mult=2.0, sl_mult=1.0, max_holding=48)

    samples, ys, t_ends, times, dirs = [], [], [], [], []
    for ts, row in labels.iterrows():
        windows = {}
        ok = True
        for tf, df in feats.items():
            window = df.loc[:ts].tail(SEQ[tf])
            if len(window) < SEQ[tf]:
                ok = False
                break
            windows[tf] = window.values
        if not ok:
            continue
        samples.append(windows)
        ys.append(row["label"])
        t_ends.append(int(row["t_end"]))
        times.append(ts)
        dirs.append(int(direction.loc[ts]))

    y = torch.tensor(ys, dtype=torch.float32)
    x = {tf: torch.tensor(np.stack([s[tf] for s in samples]), dtype=torch.float32)
         for tf in SEQ}
    meta = pd.DataFrame({"direction": dirs}, index=pd.DatetimeIndex(times))
    return x, y, np.array(t_ends), meta


def train(symbol: str, data_path: Path, out_dir: Path, epochs: int = 20,
          batch_size: int = 256, lr: float = 1e-3) -> None:
    import torch

    from aethel.venus.model import VenusNet

    m5 = pd.read_parquet(data_path)
    x, y, t_end, _meta = build_dataset(m5)
    n = len(y)
    print(f"{symbol}: {n} samples, positive rate {y.mean():.3f}")

    oof_raw, oof_y = [], []
    splitter = PurgedWalkForward(n_splits=5, embargo_bars=100)
    for fold, (tr, te) in enumerate(splitter.split(n, t_end)):
        model = VenusNet(n_features=len(FEATURE_COLUMNS))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            model.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                batch = {tf: x[tf][idx] for tf in SEQ}
                opt.zero_grad()
                loss = loss_fn(model(batch), y[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model({tf: x[tf][te] for tf in SEQ})
        oof_raw.append(torch.sigmoid(logits).numpy())
        oof_y.append(y[te].numpy())
        print(f"fold {fold}: test acc "
              f"{((torch.sigmoid(logits) > 0.5).float() == y[te]).float().mean():.3f}")

    calibrator = Calibrator()
    calibrator.fit(np.concatenate(oof_raw), np.concatenate(oof_y))

    # Final champion: retrain on all data, ship with the OOF-fit calibrator.
    model = VenusNet(n_features=len(FEATURE_COLUMNS))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model({tf: x[tf][idx] for tf in SEQ}), y[idx])
            loss.backward()
            opt.step()

    sym_dir = out_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), sym_dir / "model.pt")
    calibrator.save(sym_dir / "calibrator.pkl")
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    (sym_dir / "version.txt").write_text(version)
    print(f"saved {symbol} artifact version {version} -> {sym_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("models/artifacts"))
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    train(args.symbol, args.data, args.out, epochs=args.epochs)
