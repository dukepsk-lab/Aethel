"""MT5-style report + equity curve plotter for Aethel backtests.

Usage (from backtest modules):
    from aethel.venus.report import print_mt5_report, plot_equity_curve
    print_mt5_report(symbol, pf, trades, fold_aucs, threshold, init_cash, out_path)
"""
from __future__ import annotations

from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd


def print_mt5_report(
    symbol: str,
    pf,                    # vectorbt Portfolio
    trades,                # pf.trades
    fold_aucs: list,
    threshold: float,
    init_cash: float,
    model_tag: str = "",
) -> dict:
    """Print a Strategy Tester-style summary and return the stats dict."""
    n = int(trades.count())
    eq = pf.value()
    start_dt = eq.index[0].strftime("%Y.%m.%d")
    end_dt   = eq.index[-1].strftime("%Y.%m.%d")

    net_profit    = float(pf.total_profit()) if n else 0.0
    net_pct       = float(pf.total_return()) * 100
    gross_profit  = float(trades.pnl.values[trades.pnl.values > 0].sum()) if n else 0.0
    gross_loss    = float(trades.pnl.values[trades.pnl.values < 0].sum()) if n else 0.0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")
    win_rate      = float(trades.win_rate()) * 100 if n else 0.0
    max_dd        = float(pf.max_drawdown()) * 100
    sharpe        = float(pf.sharpe_ratio()) if n else 0.0
    avg_win       = float(trades.pnl.values[trades.pnl.values > 0].mean()) if n else 0.0
    avg_loss      = float(trades.pnl.values[trades.pnl.values < 0].mean()) if n else 0.0
    max_win       = float(trades.pnl.values.max()) if n else 0.0
    max_loss      = float(trades.pnl.values.min()) if n else 0.0
    avg_ret_pct   = float(trades.returns.mean()) * 100 if n else 0.0
    expected_pay  = net_profit / n if n else 0.0

    # consecutive wins / losses
    if n:
        pnl_arr = trades.pnl.values
        wins = (pnl_arr > 0).astype(int)
        max_consec_win = max_consec_loss = cur_w = cur_l = 0
        for w in wins:
            if w:
                cur_w += 1; cur_l = 0
            else:
                cur_l += 1; cur_w = 0
            max_consec_win  = max(max_consec_win, cur_w)
            max_consec_loss = max(max_consec_loss, cur_l)
    else:
        max_consec_win = max_consec_loss = 0

    # long / short breakdown
    dirs = trades.direction.values if hasattr(trades, "direction") else []
    n_long = int((np.asarray(dirs) == 1).sum()) if len(dirs) else 0
    n_short = n - n_long

    auc_str = (f"{np.mean([a for a in fold_aucs if a]):.4f}"
               f" (folds: {', '.join(f'{a:.4f}' for a in fold_aucs if a)})"
               if fold_aucs else "n/a")

    w = 52
    div  = "─" * w
    div2 = "═" * w

    print(f"\n{div2}")
    print(f"  STRATEGY TESTER REPORT  —  {symbol}  {model_tag}")
    print(f"{div2}")
    print(f"  Period       {start_dt} – {end_dt}")
    print(f"  OOF AUC      {auc_str}")
    print(f"  Threshold    {threshold:.2f}")
    print(div)
    print(f"  {'Initial deposit':<28} ${init_cash:>12,.2f}")
    print(f"  {'Net profit':<28} ${net_profit:>+12,.2f}   ({net_pct:+.2f}%)")
    print(f"  {'Gross profit':<28} ${gross_profit:>12,.2f}")
    print(f"  {'Gross loss':<28} ${gross_loss:>12,.2f}")
    print(f"  {'Profit factor':<28} {profit_factor:>13.2f}")
    print(f"  {'Expected payoff':<28} ${expected_pay:>+12.2f}")
    print(div)
    print(f"  {'Max drawdown':<28} {max_dd:>12.2f}%")
    print(f"  {'Sharpe ratio':<28} {sharpe:>13.2f}")
    print(div)
    print(f"  {'Total trades':<28} {n:>13,}")
    print(f"  {'  Long':<28} {n_long:>13,}")
    print(f"  {'  Short':<28} {n_short:>13,}")
    print(f"  {'Win rate':<28} {win_rate:>12.1f}%")
    print(f"  {'Max consec. wins':<28} {max_consec_win:>13}")
    print(f"  {'Max consec. losses':<28} {max_consec_loss:>13}")
    print(div)
    print(f"  {'Avg trade return':<28} {avg_ret_pct:>+12.3f}%")
    print(f"  {'Avg winning trade':<28} ${avg_win:>+12.2f}")
    print(f"  {'Avg losing trade':<28} ${avg_loss:>+12.2f}")
    print(f"  {'Largest win':<28} ${max_win:>+12.2f}")
    print(f"  {'Largest loss':<28} ${max_loss:>+12.2f}")
    print(f"{div2}\n")

    return {
        "symbol": symbol,
        "period": f"{start_dt}/{end_dt}",
        "threshold": threshold,
        "oos_auc_mean": round(float(np.mean([a for a in fold_aucs if a])), 4) if fold_aucs else None,
        "fold_aucs": fold_aucs,
        "total_trades": n,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "net_profit": round(net_profit, 2),
        "net_return_pct": round(net_pct, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "expected_payoff": round(expected_pay, 2),
        "avg_trade_return_pct": round(avg_ret_pct, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
        "max_consec_wins": max_consec_win,
        "max_consec_losses": max_consec_loss,
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
    }


def plot_equity_curve(
    symbol: str,
    pf,
    trades,
    out_path: Path,
    init_cash: float = 10_000,
    threshold: float = 0.0,
    model_tag: str = "",
) -> None:
    """Save an equity curve PNG with drawdown panel and trade markers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.lines import Line2D
    except ImportError:
        print("[report] matplotlib not installed — skipping plot")
        return

    eq    = pf.value()
    dates = eq.index.to_pydatetime()

    # drawdown series
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100

    # trade entry/exit times and P&L
    n = int(trades.count())
    if n:
        entry_times = pd.DatetimeIndex(trades.entry_idx.map(lambda i: eq.index[int(i)]))
        exit_times  = pd.DatetimeIndex(trades.exit_idx.map(lambda i: eq.index[min(int(i), len(eq)-1)]))
        pnls        = trades.pnl.values
        win_mask    = pnls > 0
    else:
        entry_times = exit_times = pd.DatetimeIndex([])
        pnls = np.array([])
        win_mask = np.array([], dtype=bool)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.spines[:].set_color("#30363d")

    # ── equity curve ──────────────────────────────────────────────────────
    ax1.plot(dates, eq.values, color="#58a6ff", linewidth=1.2, label="Equity")
    ax1.axhline(init_cash, color="#484f58", linewidth=0.8, linestyle="--")
    ax1.fill_between(dates, init_cash, eq.values,
                     where=(eq.values >= init_cash), alpha=0.12, color="#58a6ff")
    ax1.fill_between(dates, init_cash, eq.values,
                     where=(eq.values < init_cash),  alpha=0.12, color="#f85149")

    # trade markers
    if n:
        eq_at = lambda times: np.interp(
            [t.timestamp() for t in times],
            [d.timestamp() for d in dates],
            eq.values,
        )
        w_y = eq_at(entry_times[win_mask])
        l_y = eq_at(entry_times[~win_mask])
        if win_mask.sum():
            ax1.scatter([entry_times[win_mask]], w_y,
                        marker="^", color="#3fb950", s=30, zorder=5, alpha=0.8)
        if (~win_mask).sum():
            ax1.scatter([entry_times[~win_mask]], l_y,
                        marker="v", color="#f85149", s=30, zorder=5, alpha=0.8)

    net_pnl  = float(pf.total_profit()) if n else 0
    net_pct  = float(pf.total_return()) * 100
    wr       = float(trades.win_rate()) * 100 if n else 0
    pf_val   = abs(float(trades.pnl.values[trades.pnl.values > 0].sum()) /
                   float(trades.pnl.values[trades.pnl.values < 0].sum())) if n and (trades.pnl.values < 0).any() else 0

    title = (f"{symbol}  |  {model_tag}  |  threshold {threshold:.2f}  |  "
             f"{n} trades  |  WR {wr:.1f}%  |  PF {pf_val:.2f}  |  "
             f"Net {net_pnl:+,.0f} ({net_pct:+.1f}%)")
    ax1.set_title(title, color="#e6edf3", fontsize=10, pad=8)
    ax1.set_ylabel("Equity ($)", color="#8b949e", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.legend(
        handles=[
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#3fb950",
                   markersize=7, label="Win entry", linestyle="none"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#f85149",
                   markersize=7, label="Loss entry", linestyle="none"),
            Line2D([0], [0], color="#58a6ff", linewidth=1.5, label="Equity"),
        ],
        loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#8b949e",
    )

    # ── drawdown panel ────────────────────────────────────────────────────
    ax2.fill_between(dates, dd.values, 0, alpha=0.7, color="#f85149")
    ax2.plot(dates, dd.values, color="#f85149", linewidth=0.6)
    ax2.set_ylabel("Drawdown %", color="#8b949e", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    max_dd = float(dd.min())
    ax2.set_title(f"Max drawdown  {max_dd:.2f}%", color="#8b949e", fontsize=8, pad=4)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right", color="#8b949e", fontsize=8)

    fig.tight_layout(h_pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[report] equity curve saved → {out_path}")
