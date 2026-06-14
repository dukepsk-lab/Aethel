"""Walk-forward backtest for Venus, simulated through vectorbt.

Usage:
    python -m aethel.venus.backtest --symbol EURUSD --data data/EURUSD_m5.parquet

For each purged walk-forward fold: train on the past, predict the test fold
out-of-sample, calibrate on PRIOR folds' OOF predictions only (no leakage),
then hand the resulting long/short entries to vectorbt with volatility-scaled
SL/TP stops, fees and slippage. vectorbt gives us the full stats suite:
Sharpe, max drawdown, profit factor, trade-level win rate, equity curve.

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
from aethel.venus.labeling import ewm_volatility
from aethel.venus.train import SEQ, build_dataset
from aethel.venus.validation import PurgedWalkForward


def run_backtest(
    m5: pd.DataFrame,
    threshold: float = 0.65,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    fees: float = 0.00002,        # ~0.2 pip commission-equivalent per side
    slippage: float = 0.00005,    # ~0.5 pip
    epochs: int = 10,
    n_splits: int = 5,
    init_cash: float = 10_000,
) -> dict:
    print("[backtest] loading torch...", flush=True)
    import torch
    print("[backtest] loading vectorbt...", flush=True)
    import vectorbt as vbt

    from aethel.venus.calibration import Calibrator
    from aethel.venus.model import VenusNet

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[backtest] device: {device}", flush=True)

    print("[backtest] building dataset...", flush=True)
    x, y, t_end, meta = build_dataset(m5)
    y = y.to(device)
    x = {tf: t.to(device) for tf, t in x.items()}
    print(f"[backtest] dataset ready: {len(y)} samples", flush=True)
    n = len(y)

    m15 = m5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "tick_volume": "sum"}).dropna()
    close = m15["close"]
    vol = ewm_volatility(close, span=100)

    # OOS signal arrays assembled across folds
    conf = pd.Series(np.nan, index=meta.index)
    prior_raw: list[np.ndarray] = []
    prior_y: list[np.ndarray] = []
    fold_aucs = []

    splitter = PurgedWalkForward(n_splits=n_splits, embargo_bars=100)
    folds = [(tr, te) for tr, te in splitter.split(n, t_end) if len(tr) >= 200]
    print(f"[backtest] {len(folds)} folds | {n} samples | threshold={threshold}")

    for fold_idx, (tr, te) in enumerate(folds, 1):
        print(f"\n[fold {fold_idx}/{len(folds)}] train={len(tr)} test={len(te)}")
        model = VenusNet(n_features=len(FEATURE_COLUMNS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for ep in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss = loss_fn(model({tf: x[tf][idx] for tf in SEQ}), y[idx])
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  epoch {ep:>2}/{epochs}  loss={avg_loss:.4f}", flush=True)
        model.eval()
        with torch.no_grad():
            raw = torch.sigmoid(model({tf: x[tf][te] for tf in SEQ})).cpu().numpy()
        y_te = y[te].cpu().numpy()
        try:
            from sklearn.metrics import roc_auc_score

            auc = round(float(roc_auc_score(y_te, raw)), 4)
            fold_aucs.append(auc)
            print(f"  OOF AUC={auc}")
        except ValueError:
            fold_aucs.append(None)
            print("  OOF AUC=n/a (single class in test fold)")

        if prior_raw:  # calibrate on previous folds only — leak-free
            cal = Calibrator()
            cal.fit(np.concatenate(prior_raw), np.concatenate(prior_y))
            conf.iloc[te] = cal.transform(raw)
        prior_raw.append(raw)
        prior_y.append(y_te)

    # --- assemble vectorbt signals on the M15 grid ---
    conf_full = conf.reindex(close.index)
    direction = meta["direction"].reindex(close.index)
    take = conf_full >= threshold

    # diagnostics
    print(f"\n[signals] conf non-nan: {conf.notna().sum()} | conf_full non-nan: {conf_full.notna().sum()}", flush=True)
    print(f"[signals] conf range: {conf.dropna().min():.4f} – {conf.dropna().max():.4f}" if conf.notna().any() else "[signals] conf all NaN", flush=True)
    print(f"[signals] take=True: {take.sum()} | direction>0: {(direction>0).sum()} | direction<0: {(direction<0).sum()}", flush=True)

    entries = (take & (direction > 0)).fillna(False)
    short_entries = (take & (direction < 0)).fillna(False)
    print(f"[signals] entries: {entries.sum()} | short_entries: {short_entries.sum()}", flush=True)

    sl_frac = (sl_mult * vol / close).clip(lower=1e-5)
    tp_frac = (tp_mult * vol / close).clip(lower=1e-5)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        short_entries=short_entries,
        sl_stop=sl_frac.to_numpy(),
        tp_stop=tp_frac.to_numpy(),
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        size=1.0,
        size_type="percent",  # full notional per trade; risk scaling is live-side
        freq="15min",
    )

    trades = pf.trades
    stats = {
        "threshold": threshold,
        "oos_samples": int(conf.notna().sum()),
        "fold_aucs": fold_aucs,
        "total_trades": int(trades.count()),
        "win_rate": round(float(trades.win_rate()), 3) if trades.count() else None,
        "profit_factor": round(float(trades.profit_factor()), 2) if trades.count() else None,
        "total_return_pct": round(float(pf.total_return()) * 100, 2),
        "sharpe": round(float(pf.sharpe_ratio()), 2),
        "max_drawdown_pct": round(float(pf.max_drawdown()) * 100, 2),
        "avg_trade_return_pct": (
            round(float(trades.returns.mean()) * 100, 3) if trades.count() else None
        ),
    }
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--balance", type=float, default=10_000, help="Starting balance (default 10000)")
    args = p.parse_args()
    m5 = pd.read_parquet(args.data)
    report = run_backtest(m5, threshold=args.threshold, epochs=args.epochs, init_cash=args.balance)
    print(json.dumps({"symbol": args.symbol, **report}, indent=2))
