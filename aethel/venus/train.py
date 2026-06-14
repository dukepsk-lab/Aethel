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


def resample_frames(m5: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build M5/M15/H1 OHLCV frames from M5 input.

    H1 is labeled with ``label="right"`` so each bar carries the timestamp of its
    CLOSE, not its open. This is critical to avoid lookahead: ``df.loc[:ts]`` then
    returns only H1 bars that have fully closed at or before ``ts``. With the
    pandas default (``label="left"``) the in-progress H1 bar — whose high/low/close
    encode prices up to an hour in the future — would be selected at the M15 entry
    bar, leaking the future into both the H1 features and the H1-derived trade
    direction (``sign(dist_ema50)``).

    M15 keeps the default left label (the entry bar's own close IS the execution
    price, known at entry) and M5 stays native; with M15 timestamps on the bar
    open, ``loc[:ts]`` on left-labeled M5/M15 never reaches past the entry bar.
    """
    return {
        "M5": m5,
        "M15": m5.resample("15min").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "tick_volume": "sum"}).dropna(),
        "H1": m5.resample("1h", label="right", closed="left").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "tick_volume": "sum"}).dropna(),
    }


def remap_t_end_to_samples(t_end_frame: np.ndarray, frame_index: pd.DatetimeIndex,
                           sample_times: list) -> np.ndarray:
    """Convert barrier-touch positions from frame coordinates to sample-array
    coordinates so PurgedWalkForward purges correctly.

    ``triple_barrier_labels`` records ``t_end`` as an integer row position within
    the M15 frame. But the dataset keeps only a SUBSET of those rows as samples
    (Helios keeps ~30%), so a frame position is not comparable to a sample index.
    We map each barrier-touch *time* to the first sample at or after it.
    """
    sample_idx = pd.DatetimeIndex(sample_times)
    barrier_times = frame_index[t_end_frame]
    return np.searchsorted(sample_idx.values, barrier_times.values, side="left")


def pick_device():
    """Prefer CUDA, then Apple MPS, else CPU. Printed once so the operator
    can confirm the GPU is actually being used."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(m5: pd.DataFrame):
    import torch

    frames = resample_frames(m5)
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
    # remap barrier-touch positions from M15-frame coords to sample-array coords
    t_end = remap_t_end_to_samples(np.array(t_ends), base.index, times)
    return x, y, t_end, meta


def train(symbol: str, data_path: Path, out_dir: Path, epochs: int = 20,
          batch_size: int = 256, lr: float = 1e-3) -> None:
    import torch

    from aethel.venus.model import VenusNet

    device = pick_device()
    print(f"training on device: {device}")

    m5 = pd.read_parquet(data_path)
    x, y, t_end, _meta = build_dataset(m5)
    # keep features on CPU (can be large); move per-batch. labels are small.
    y = y.to(device)
    n = len(y)
    print(f"{symbol}: {n} samples, positive rate {y.mean():.3f}")

    def to_device(idx):
        return {tf: x[tf][idx].to(device) for tf in SEQ}

    oof_raw, oof_y = [], []
    splitter = PurgedWalkForward(n_splits=5, embargo_bars=100)
    for fold, (tr, te) in enumerate(splitter.split(n, t_end)):
        model = VenusNet(n_features=len(FEATURE_COLUMNS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            model.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(to_device(idx)), y[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(to_device(te))
        oof_raw.append(torch.sigmoid(logits).cpu().numpy())
        oof_y.append(y[te].cpu().numpy())
        print(f"fold {fold}: test acc "
              f"{((torch.sigmoid(logits) > 0.5).float() == y[te]).float().mean():.3f}")

    calibrator = Calibrator()
    calibrator.fit(np.concatenate(oof_raw), np.concatenate(oof_y))

    # Final champion: retrain on all data, ship with the OOF-fit calibrator.
    model = VenusNet(n_features=len(FEATURE_COLUMNS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(to_device(idx)), y[idx])
            loss.backward()
            opt.step()

    sym_dir = out_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    # save on CPU so the artifact loads anywhere regardless of training device
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, sym_dir / "model.pt")
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
