"""Helios training — same VenusNet architecture, selective labels.

Usage:
    python -m aethel.helios.train --symbol EURUSD --data data/EURUSD_m5.parquet
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aethel.venus.calibration import Calibrator
from aethel.venus.features import FEATURE_COLUMNS, compute_features
from aethel.venus.train import SEQ, pick_device
from aethel.venus.validation import PurgedWalkForward
from aethel.helios.labeling import selective_labels


def build_helios_dataset(m5: pd.DataFrame):
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

    labels = selective_labels(
        frames["M15"],
        h1_features=feats["H1"],
        tp_mult=2.0, sl_mult=1.0, max_holding=48,
    )

    samples, ys, t_ends, times, dirs = [], [], [], [], []
    base = frames["M15"]
    h1_dist = feats["H1"]["dist_ema50"].reindex(base.index, method="ffill")
    direction = np.sign(h1_dist).fillna(0).astype(int)

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
        dirs.append(int(direction.loc[ts]) if ts in direction.index else 0)

    print(f"Helios dataset: {len(ys)} samples (vs Venus ~100k) | positive rate {np.mean(ys):.3f}")
    y = torch.tensor(ys, dtype=torch.float32)
    x = {tf: torch.tensor(np.stack([s[tf] for s in samples]), dtype=torch.float32)
         for tf in SEQ}
    meta = pd.DataFrame({"direction": dirs}, index=pd.DatetimeIndex(times))
    return x, y, np.array(t_ends), meta


def train(symbol: str, data_path: Path, out_dir: Path, epochs: int = 20) -> None:
    import torch
    from aethel.venus.model import VenusNet

    device = pick_device()
    print(f"[Helios] training {symbol} on {device}")

    m5 = pd.read_parquet(data_path)
    x, y, t_end, _meta = build_helios_dataset(m5)
    y = y.to(device)
    n = len(y)
    if n < 200:
        print(f"[Helios] {symbol}: only {n} samples after filtering — skipping")
        return

    def to_device(idx):
        return {tf: x[tf][idx].to(device) for tf in SEQ}

    oof_raw, oof_y = [], []
    splitter = PurgedWalkForward(n_splits=5, embargo_bars=100)
    for fold, (tr, te) in enumerate(splitter.split(n, t_end)):
        if len(tr) < 100:
            continue
        model = VenusNet(n_features=len(FEATURE_COLUMNS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            model.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss = loss_fn(model(to_device(idx)), y[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(to_device(te))
        raw = torch.sigmoid(logits).cpu().numpy()
        oof_raw.append(raw)
        oof_y.append(y[te].cpu().numpy())
        acc = ((torch.sigmoid(logits) > 0.5).float() == y[te]).float().mean()
        print(f"  fold {fold}: acc={acc:.3f}")

    if not oof_raw:
        print(f"[Helios] {symbol}: no folds completed")
        return

    calibrator = Calibrator()
    calibrator.fit(np.concatenate(oof_raw), np.concatenate(oof_y))

    # Final model on all data
    model = VenusNet(n_features=len(FEATURE_COLUMNS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(model(to_device(idx)), y[idx])
            loss.backward()
            opt.step()

    sym_dir = out_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, sym_dir / "model.pt")
    calibrator.save(sym_dir / "calibrator.pkl")
    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    (sym_dir / "version.txt").write_text(version)
    print(f"[Helios] saved {symbol} -> {sym_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Helios selective model")
    p.add_argument("--symbol", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("models/artifacts_helios"))
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    train(args.symbol, Path(args.data), args.out, args.epochs)
