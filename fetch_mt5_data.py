"""fetch_mt5_data.py — รันบน Windows host ที่มี MT5 terminal เปิดอยู่

ดึง M5 OHLCV ย้อนหลัง N วัน สำหรับทุก symbol แล้ว save เป็น parquet
พร้อม train Venus ทันทีถ้าใส่ --train

Usage:
    python fetch_mt5_data.py
    python fetch_mt5_data.py --symbols EURUSD GBPUSD --days 365
    python fetch_mt5_data.py --days 730 --train --epochs 15
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---- guard: MetaTrader5 is Windows-only ----
try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("❌  MetaTrader5 package not found.\n"
             "   Run:  pip install MetaTrader5\n"
             "   And make sure MT5 terminal is installed and running.")

import pandas as pd

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
TF_MAP = {
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
}


def connect() -> None:
    if not mt5.initialize():
        code, msg = mt5.last_error()
        sys.exit(f"❌  MT5 initialize() failed [{code}]: {msg}\n"
                 "    Is MetaTrader 5 terminal open and logged in?")
    info = mt5.account_info()
    print(f"✅  Connected — account #{info.login}  "
          f"server: {info.server}  balance: {info.balance:.2f} {info.currency}")


def fetch_symbol(symbol: str, days: int, out_dir: Path) -> Path:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # MT5 needs UTC datetime objects (no tzinfo for copy_rates_range)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5,
                                 start.replace(tzinfo=None),
                                 end.replace(tzinfo=None))
    if rates is None or len(rates) == 0:
        code, msg = mt5.last_error()
        print(f"  ⚠️  {symbol}: no data returned [{code}] {msg}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = (df.rename(columns={"tick_volume": "tick_volume"})
            .set_index("time")
            [["open", "high", "low", "close", "tick_volume"]])
    df = df[~df.index.duplicated(keep="last")].sort_index()

    path = out_dir / f"{symbol}_m5.parquet"
    df.to_parquet(path)

    span_days = (df.index[-1] - df.index[0]).days
    print(f"  ✅  {symbol}: {len(df):,} bars  "
          f"({df.index[0].date()} → {df.index[-1].date()}, {span_days}d)  "
          f"→ {path}")
    return path


def show_summary(symbol: str, path: Path) -> None:
    df = pd.read_parquet(path)
    close = df["close"]
    print(f"     range: {close.min():.5f} – {close.max():.5f}  "
          f"avg spread proxy: {(df['high'] - df['low']).mean():.5f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch MT5 M5 data → parquet")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--days",    type=int,  default=365,
                    help="How many calendar days to pull (default 365)")
    ap.add_argument("--out",     type=Path, default=Path("data"),
                    help="Output directory (default: data/)")
    ap.add_argument("--train",   action="store_true",
                    help="Run Venus training immediately after fetching")
    ap.add_argument("--helios", action="store_true",
                    help="Also train Helios selective model after fetching")
    ap.add_argument("--epochs",  type=int,  default=15,
                    help="Training epochs per symbol (default 15, used with --train)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    connect()

    fetched: list[tuple[str, Path]] = []
    for symbol in args.symbols:
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"  ⚠️  {symbol} not found in broker — skipping")
            continue
        if not info.visible:
            mt5.symbol_select(symbol, True)  # add to Market Watch

        path = fetch_symbol(symbol, args.days, args.out)
        if path:
            show_summary(symbol, path)
            fetched.append((symbol, path))

    mt5.shutdown()
    print(f"\n{'─'*55}")
    print(f"Fetched {len(fetched)}/{len(args.symbols)} symbols → {args.out}/")

    if not args.train:
        print("\nNext step — backtest each symbol:")
        for sym, p in fetched:
            print(f"  python -m aethel.venus.backtest --symbol {sym} --data {p} --epochs 10")
        print("\nThen train:")
        for sym, p in fetched:
            print(f"  python -m aethel.venus.train    --symbol {sym} --data {p} --epochs 15")
        return

    # ---- optional immediate training ----
    print(f"\nStarting Venus training (epochs={args.epochs}) …\n")
    from aethel.venus.train import train

    for sym, path in fetched:
        print(f"{'─'*55}\n🏋  Training {sym} …")
        try:
            train(sym, path, Path("models/artifacts"), epochs=args.epochs)
        except Exception as e:
            print(f"  ❌  {sym} training failed: {e}")
    print("\n✅  Done. Artifacts saved to models/artifacts/")

    if args.helios:
        print(f"\nStarting Helios training (selective labels, epochs={args.epochs}) …\n")
        from aethel.helios.train import train as helios_train
        for sym, path in fetched:
            print(f"{'─'*55}\n🔭  Training Helios {sym} …")
            try:
                helios_train(sym, path, Path("models/artifacts_helios"), epochs=args.epochs)
            except Exception as e:
                print(f"  ❌  {sym} Helios training failed: {e}")
        print("\n✅  Helios done. Artifacts saved to models/artifacts_helios/")


if __name__ == "__main__":
    main()
