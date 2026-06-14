"""Diagnose the saved calibrator vs the deployed final model.

The calibrator (isotonic regression) is fit on out-of-fold predictions from the
per-fold models, but it is applied to a SEPARATE final model trained on all data.
If the final model's raw-score distribution sits outside the range the isotonic
map was fit on, calibrated confidence collapses toward the base rate — which is
why Venus produced 0 trades at threshold 0.55.

This tool prints, for a given symbol:
  * the isotonic mapping knots (raw -> calibrated)
  * the final model's raw-score distribution
  * where those raw scores land AFTER calibration
  * how many would clear common thresholds

Usage:
    python -m aethel.venus.diagnose_calibration --symbol EURUSD --data data/EURUSD_m5.parquet --model-dir models/artifacts
    python -m aethel.venus.diagnose_calibration --symbol EURUSD --data data/EURUSD_m5.parquet --model-dir models/artifacts_helios --helios
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aethel.venus.calibration import Calibrator
from aethel.venus.features import FEATURE_COLUMNS
from aethel.venus.train import SEQ, build_dataset


def _hist(vals: np.ndarray, lo: float = 0.0, hi: float = 1.0, bins: int = 10) -> None:
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(vals, bins=edges)
    total = max(len(vals), 1)
    peak = max(counts.max(), 1)
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        bar = "█" * int(c / peak * 40)
        print(f"  {e0:4.2f}–{e1:4.2f}  {c:>8,}  {c/total*100:5.1f}%  {bar}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--helios", action="store_true",
                   help="Use the selective-label Helios dataset builder")
    args = p.parse_args()

    import torch
    from aethel.venus.model import VenusNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sym_dir = args.model_dir / args.symbol

    model = VenusNet(n_features=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load(sym_dir / "model.pt", map_location=device))
    model.to(device).eval()
    cal = Calibrator.load(sym_dir / "calibrator.pkl")
    iso = cal._iso  # sklearn IsotonicRegression

    print(f"\n{'='*64}\nCalibration diagnostic — {args.symbol} ({sym_dir})\n{'='*64}")

    # --- isotonic mapping knots ---
    xk = np.asarray(iso.X_thresholds_)
    yk = np.asarray(iso.y_thresholds_)
    print("\nIsotonic map was FIT on OOF raw scores in range "
          f"[{xk.min():.4f}, {xk.max():.4f}]  ->  calibrated [{yk.min():.4f}, {yk.max():.4f}]")
    print("knots (raw -> calibrated), sampled:")
    idx = np.linspace(0, len(xk) - 1, min(12, len(xk))).astype(int)
    for i in idx:
        print(f"    raw {xk[i]:.4f}  ->  {yk[i]:.4f}")

    # --- final model raw scores on the actual dataset ---
    if args.helios:
        from aethel.helios.train import build_helios_dataset
        x, y, _t, _meta = build_helios_dataset(pd.read_parquet(args.data))
    else:
        x, y, _t, _meta = build_dataset(pd.read_parquet(args.data))
    x = {tf: t.to(device) for tf, t in x.items()}

    raws = []
    with torch.no_grad():
        for i in range(0, len(y), 4096):
            sl = slice(i, i + 4096)
            logits = model({tf: x[tf][sl] for tf in SEQ})
            raws.append(torch.sigmoid(logits).cpu().numpy())
    raw = np.concatenate(raws).ravel()
    calibrated = cal.transform(raw)

    print(f"\nFINAL model raw-score distribution  (n={len(raw):,}):")
    print(f"  range [{raw.min():.4f}, {raw.max():.4f}]  mean {raw.mean():.4f}")
    _hist(raw)

    below = (raw < xk.min()).mean() * 100
    above = (raw > xk.max()).mean() * 100
    print(f"\n  {below:.1f}% of final-model raw scores are BELOW the calibrator's "
          f"fitted min ({xk.min():.4f}) -> clipped to {yk.min():.4f}")
    print(f"  {above:.1f}% are ABOVE its fitted max ({xk.max():.4f}) -> clipped to {yk.max():.4f}")

    print("\nCALIBRATED confidence distribution (what the gate sees):")
    print(f"  range [{calibrated.min():.4f}, {calibrated.max():.4f}]  mean {calibrated.mean():.4f}")
    _hist(calibrated)

    print("\nFraction clearing thresholds:")
    for th in (0.5, 0.55, 0.6, 0.65, 0.7):
        print(f"  >= {th:.2f}:  {(calibrated >= th).mean()*100:6.2f}%  "
              f"({int((calibrated >= th).sum()):,} signals)")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
