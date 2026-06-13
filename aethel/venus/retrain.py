"""Scheduled auto-retrain and champion/challenger promotion for Venus.

Venus must not stay frozen: markets drift, so the champion model is
periodically re-evaluated against fresh data. A retrain runs the
champion/challenger benchmark on the latest walk-forward folds; the new
model is promoted to champion only when it clears an objective bar (OOF AUC
above chance + positive trade-level avg-R), never on a single noisy fold.

A model registry (`models/registry.json`) records every cycle's metrics so
"did Venus improve?" becomes a number, not a feeling. When MLflow is
installed the same metrics are logged there too (optional, fail-soft).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aethel.venus.benchmark import benchmark
from aethel.venus.train import train

logger = logging.getLogger(__name__)

ARTIFACTS = Path("models/artifacts")
REGISTRY_FILE = Path("models/registry.json")

# Promotion bar: OOF AUC must beat chance by a meaningful margin AND the
# strategy must be net-positive in R-multiples on the held-out folds.
MIN_AUC = 0.52
MIN_AVG_R = 0.0


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def _save_registry(reg: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2, default=str))


def get_registry() -> dict:
    return _load_registry()


def _promotion_criteria(benchmark_results: dict,
                        contender: str = "TCN (champion)") -> bool:
    """Promote only if the freshly-trained model clears the objective bar:
    OOF mean AUC >= MIN_AUC AND trade-level avg-R > MIN_AVG_R."""
    res = benchmark_results.get(contender, {})
    mean_auc = res.get("mean_auc")
    avg_r = res.get("avg_r")
    if mean_auc is None or avg_r is None:
        return False
    return mean_auc >= MIN_AUC and avg_r > MIN_AVG_R


def _try_mlflow_log(symbol: str, bench: dict, action: str) -> None:
    """Log this cycle's metrics to MLflow if it is installed. Fail-soft —
    MLflow is an optional local tool, never a hard dependency."""
    try:
        import mlflow
    except Exception:
        return
    try:
        mlflow.set_experiment("venus_retrain")
        with mlflow.start_run(run_name=f"{symbol}_{action}"):
            tcn = bench.get("TCN (champion)", {})
            mlflow.log_param("symbol", symbol)
            mlflow.log_param("action", action)
            mlflow.log_metric("mean_auc", float(tcn.get("mean_auc") or 0.0))
            mlflow.log_metric("avg_r", float(tcn.get("avg_r") or 0.0))
            mlflow.log_metric("trades", int(tcn.get("trades") or 0))
    except Exception as e:  # pragma: no cover - depends on local MLflow
        logger.warning("mlflow log failed: %s", e)


def run_retrain_cycle(symbol: str, data_path: Path,
                      out_dir: Path | None = None, force: bool = False) -> dict:
    """Full retrain cycle for one symbol.

    1. Benchmark champion vs challengers on the latest data (purged WF folds).
    2. If the promotion bar is met (or ``force``), train the full champion
       and overwrite the deployed artifacts via ``train``.
    3. Record metrics + action in the registry, and log to MLflow if present.

    ``data_path`` is the latest M5 parquet; ``out_dir`` defaults to
    ``models/artifacts/{symbol}``. Returns ``{symbol, action, benchmark}``
    where action is one of ``promoted`` / ``skipped``.
    """
    out_dir = out_dir or (ARTIFACTS / symbol)
    logger.info("[retrain] starting cycle for %s", symbol)
    m5_df = pd.read_parquet(data_path)
    bench = benchmark(m5_df, epochs=8, n_splits=4, threshold=0.65)
    tcn_auc = bench.get("TCN (champion)", {}).get("mean_auc")
    logger.info("[retrain] benchmark done for %s — TCN mean_auc=%s", symbol, tcn_auc)

    promoted = force or _promotion_criteria(bench)
    action = "skipped"
    if promoted:
        logger.info("[retrain] promotion bar met — training champion for %s", symbol)
        train(symbol, data_path, out_dir)
        action = "promoted"
        logger.info("[retrain] %s champion updated", symbol)
    else:
        logger.info("[retrain] %s did not clear promotion bar — keeping model", symbol)

    reg = _load_registry()
    reg[symbol] = {
        "last_retrain": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "benchmark": bench,
    }
    _save_registry(reg)
    _try_mlflow_log(symbol, bench, action)
    return {"symbol": symbol, "action": action, "benchmark": bench}


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(
        description="Run Venus retrain cycle locally, then optionally deploy "
                    "model artifacts to a remote VPS via rsync."
    )
    ap.add_argument("--symbol", required=True,
                    help="Symbol to retrain, e.g. EURUSD")
    ap.add_argument("--data", required=True, type=Path,
                    help="Path to M5 parquet file for this symbol")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override output dir (default: models/artifacts/<symbol>)")
    ap.add_argument("--force", action="store_true",
                    help="Promote even if promotion bar is not met (use for first run)")
    ap.add_argument("--epochs", type=int, default=20,
                    help="Training epochs for the full champion model (default: 20)")
    ap.add_argument("--deploy", metavar="USER@HOST",
                    help="rsync artifacts to VPS after training, e.g. ubuntu@1.2.3.4")
    ap.add_argument("--deploy-path", default="~/aethel/models/artifacts",
                    help="Remote destination path (default: ~/aethel/models/artifacts)")
    ap.add_argument("--deploy-key", metavar="PATH",
                    help="SSH identity file for rsync (optional)")
    args = ap.parse_args()

    result = run_retrain_cycle(args.symbol, args.data,
                               out_dir=args.out_dir, force=args.force)
    print(json.dumps(result, indent=2, default=str))

    if args.deploy and result["action"] == "promoted":
        import subprocess
        local_dir = str(ARTIFACTS / args.symbol) + "/"
        remote = f"{args.deploy}:{args.deploy_path}/{args.symbol}/"
        cmd = ["rsync", "-avz", "--mkpath"]
        if args.deploy_key:
            cmd += ["-e", f"ssh -i {args.deploy_key}"]
        cmd += [local_dir, remote]
        print(f"\n→ deploying to {remote}")
        subprocess.run(cmd, check=True)
        print("✓ deploy complete — restart the bot on VPS to load the new model")
    elif args.deploy and result["action"] == "skipped":
        print("\n↷ promotion bar not met — nothing deployed (use --force to override)")
