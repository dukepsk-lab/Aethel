"""Walk-forward backtest for Helios (selective-label model).

Usage:
    python -m aethel.helios.backtest --symbol EURUSD --data data/EURUSD_m5.parquet --model-dir models/artifacts_helios
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aethel.venus.features import FEATURE_COLUMNS
from aethel.venus.labeling import ewm_volatility
from aethel.venus.train import SEQ
from aethel.venus.validation import PurgedWalkForward
from aethel.venus.report import print_mt5_report, plot_equity_curve
from aethel.helios.train import build_helios_dataset


# Contract sizes: base units per standard lot
_CONTRACT_UNITS = {"XAUUSD": 100}   # troy oz; all forex = 100_000


def _position_value(symbol: str, lots: float, ref_price: float) -> float:
    """Approximate USD notional for vectorbt size_type='value'."""
    if symbol == "XAUUSD":
        return lots * 100 * ref_price   # gold: 100 oz/lot * price
    return lots * 100_000               # forex: 100k base units/lot (price ~1 so USD≈units)


def run_backtest(
    m5: pd.DataFrame,
    symbol: str = "UNKNOWN",
    threshold: float = 0.50,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    fees: float = 0.00002,
    slippage: float = 0.00005,
    epochs: int = 20,
    n_splits: int = 5,
    init_cash: float = 10_000,
    model_dir: Path | None = None,
    lot_per_equity: float = 200.0,
    min_lot: float = 0.01,
    max_lot: float = 10.0,
    data_days: int | None = None,
    plot_dir: Path | None = None,
) -> dict:
    if data_days is not None:
        cutoff = m5.index[-1] - pd.Timedelta(days=data_days)
        m5 = m5[m5.index >= cutoff]

    print("[helios-backtest] loading torch...", flush=True)
    import torch
    print("[helios-backtest] loading vectorbt...", flush=True)
    import vectorbt as vbt

    from aethel.venus.calibration import Calibrator
    from aethel.venus.model import VenusNet

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[helios-backtest] device: {device}", flush=True)

    pretrained_model = None
    pretrained_cal = None
    if model_dir is not None:
        sym_dir = Path(model_dir) / symbol
        print(f"[helios-backtest] loading pre-trained model from {sym_dir}...", flush=True)
        pretrained_model = VenusNet(n_features=len(FEATURE_COLUMNS))
        pretrained_model.load_state_dict(
            torch.load(sym_dir / "model.pt", map_location=device)
        )
        pretrained_model.to(device).eval()
        pretrained_cal = Calibrator.load(sym_dir / "calibrator.pkl")
        print("[helios-backtest] model + saved calibrator loaded (matches live)", flush=True)

    print("[helios-backtest] building Helios dataset...", flush=True)
    x, y, t_end, meta = build_helios_dataset(m5)
    y = y.to(device)
    x = {tf: t.to(device) for tf, t in x.items()}
    n = len(y)
    print(f"[helios-backtest] dataset: {n} selective samples", flush=True)

    m15 = m5.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "tick_volume": "sum"}).dropna()
    close = m15["close"]
    vol = ewm_volatility(close, span=100)

    conf = pd.Series(np.nan, index=meta.index)
    all_conf: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    prior_raw: list[np.ndarray] = []
    prior_y: list[np.ndarray] = []
    fold_aucs = []

    splitter = PurgedWalkForward(n_splits=n_splits, embargo_bars=100)
    folds = [(tr, te) for tr, te in splitter.split(n, t_end) if len(tr) >= 100]
    print(f"[helios-backtest] {len(folds)} folds | threshold={threshold}", flush=True)

    for fold_idx, (tr, te) in enumerate(folds, 1):
        print(f"\n[fold {fold_idx}/{len(folds)}] train={len(tr)} test={len(te)}", flush=True)

        if pretrained_model is not None:
            model = pretrained_model
        else:
            model = VenusNet(n_features=len(FEATURE_COLUMNS)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            for ep in range(1, epochs + 1):
                model.train()
                perm = np.random.permutation(tr)
                epoch_loss, n_batches = 0.0, 0
                for i in range(0, len(perm), 256):
                    idx = perm[i:i + 256]
                    opt.zero_grad()
                    loss = loss_fn(model({tf: x[tf][idx] for tf in SEQ}), y[idx])
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss.detach())
                    n_batches += 1
                print(f"  epoch {ep:>2}/{epochs}  loss={epoch_loss/max(n_batches,1):.4f}", flush=True)
            model.eval()

        with torch.no_grad():
            raw = torch.sigmoid(model({tf: x[tf][te] for tf in SEQ})).cpu().numpy()
        y_te = y[te].cpu().numpy()

        try:
            from sklearn.metrics import roc_auc_score
            auc = round(float(roc_auc_score(y_te, raw)), 4)
            fold_aucs.append(auc)
            print(f"  OOF AUC={auc}", flush=True)
        except ValueError:
            fold_aucs.append(None)

        if pretrained_cal is not None:
            # use the SAVED calibrator — this is exactly what live inference uses,
            # so backtest confidence (and the trades it gates) reflect live.
            cal_conf = pretrained_cal.transform(raw)
            conf.iloc[te] = cal_conf
        elif prior_raw:
            cal = Calibrator()
            cal.fit(np.concatenate(prior_raw), np.concatenate(prior_y))
            cal_conf = cal.transform(raw)
            conf.iloc[te] = cal_conf
        else:
            # first fold with no saved calibrator — fall back to raw sigmoid
            conf.iloc[te] = raw
            cal_conf = raw
        all_conf.append(cal_conf)
        all_labels.append(y_te)
        prior_raw.append(raw)
        prior_y.append(y_te)

    # signals on M15 grid — only at selective-label timestamps
    conf_full = conf.reindex(close.index)
    direction = meta["direction"].reindex(close.index)
    take = conf_full >= threshold

    print(f"\n[signals] conf range: {conf.dropna().min():.4f} – {conf.dropna().max():.4f}" if conf.notna().any() else "[signals] conf all NaN", flush=True)
    print(f"[signals] take={take.sum()} long={((take)&(direction>0)).sum()} short={(take&(direction<0)).sum()}", flush=True)

    if all_conf:
        c_arr = np.concatenate(all_conf).ravel()
        l_arr = np.concatenate(all_labels).ravel()
        bins = np.arange(0.0, 1.05, 0.1)
        print(f"\n{'─'*62}", flush=True)
        print(f"{'Confidence':>12}  {'Count':>7}  {'% of OOS':>9}  {'Hit Rate':>9}  Bar", flush=True)
        print(f"{'─'*62}", flush=True)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (c_arr >= lo) & (c_arr < hi)
            cnt = mask.sum()
            if cnt == 0:
                continue
            hit = l_arr[mask].mean()
            pct = cnt / len(c_arr) * 100
            bar = f"{'█' * int(hit * 20)}{'░' * (20 - int(hit * 20))}"
            marker = " ◀ threshold" if lo < threshold <= hi else ""
            print(f"  {lo:.1f}–{hi:.1f}      {cnt:>7,}  {pct:>8.1f}%  {hit:>8.1%}  {bar}{marker}", flush=True)
        print(f"{'─'*62}", flush=True)
        total_above = (c_arr >= threshold).sum()
        hit_above = l_arr[c_arr >= threshold].mean() if total_above else 0
        print(f"  Above {threshold:.2f}   {total_above:>7,}  {'':>9}  {hit_above:>8.1%}  ← signals used", flush=True)
        print(f"{'─'*62}\n", flush=True)

    entries = (take & (direction > 0)).fillna(False)
    short_entries = (take & (direction < 0)).fillna(False)

    sl_frac = (sl_mult * vol / close).clip(lower=1e-5)
    tp_frac = (tp_mult * vol / close).clip(lower=1e-5)

    # lot-based sizing
    lots = float(np.clip(
        np.floor(init_cash / lot_per_equity) * 0.01,
        min_lot, max_lot,
    ))
    ref_price = float(close.dropna().iloc[0])
    pos_value = _position_value(symbol, lots, ref_price)
    print(f"[sizing] lots={lots:.2f}  ref_price={ref_price:.5f}  position_value=${pos_value:,.0f}", flush=True)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        short_entries=short_entries,
        sl_stop=sl_frac.to_numpy(),
        tp_stop=tp_frac.to_numpy(),
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        size=pos_value,
        size_type="value",
        freq="15min",
    )

    trades = pf.trades
    model_tag = "helios-pretrained" if pretrained_model else f"helios-trained_{epochs}ep"
    stats = print_mt5_report(symbol, pf, trades, fold_aucs, threshold, init_cash, model_tag,
                             lots=lots, lot_per_equity=lot_per_equity)
    if plot_dir is not None:
        out_png = Path(plot_dir) / f"{symbol}_helios_equity.png"
        plot_equity_curve(symbol, pf, trades, out_png, init_cash, threshold, model_tag, lots=lots)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--balance", type=float, default=10_000)
    p.add_argument("--lot-per-equity", type=float, default=200.0)
    p.add_argument("--min-lot", type=float, default=0.01)
    p.add_argument("--max-lot", type=float, default=10.0)
    p.add_argument("--data-days", type=int, default=None,
                   help="Use only the last N calendar days of data (e.g. 365)")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--plot-dir", type=Path, default=Path("results"),
                   help="Directory to save equity curve PNG (default: results/)")
    p.add_argument("--no-plot", action="store_true", help="Skip equity curve generation")
    args = p.parse_args()
    m5 = pd.read_parquet(args.data)
    report = run_backtest(
        m5,
        symbol=args.symbol,
        threshold=args.threshold,
        epochs=args.epochs,
        init_cash=args.balance,
        model_dir=args.model_dir,
        lot_per_equity=args.lot_per_equity,
        min_lot=args.min_lot,
        max_lot=args.max_lot,
        data_days=args.data_days,
        plot_dir=None if args.no_plot else args.plot_dir,
    )
    print(json.dumps(report, indent=2))
