"use client";

import styles from "./TradeStats.module.css";

function computeStats(trades: any[]) {
  const closed = trades
    .slice()
    .sort((a, b) => new Date(a.closed_at).getTime() - new Date(b.closed_at).getTime());

  const total = closed.length;
  const wins = closed.filter((t) => t.profit > 0);
  const losses = closed.filter((t) => t.profit < 0);
  const winCount = wins.length;
  const lossCount = losses.length;

  const grossProfit = wins.reduce((s, t) => s + t.profit, 0);
  const grossLoss = Math.abs(losses.reduce((s, t) => s + t.profit, 0));
  const netProfit = closed.reduce((s, t) => s + t.profit, 0);

  const winRate = total ? winCount / total : 0;
  const lossRate = total ? lossCount / total : 0;
  const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0;

  const avgWin = winCount ? grossProfit / winCount : 0;
  const avgLoss = lossCount ? grossLoss / lossCount : 0;

  const profits = closed.map((t) => t.profit);
  const bestTrade = profits.length ? Math.max(...profits) : 0;
  const worstTrade = profits.length ? Math.min(...profits) : 0;

  const expectancy = winRate * avgWin - lossRate * avgLoss;

  const longs = closed.filter((t) => t.action === "BUY");
  const shorts = closed.filter((t) => t.action === "SELL");
  const longsWonPct = longs.length
    ? (longs.filter((t) => t.profit > 0).length / longs.length) * 100
    : 0;
  const shortsWonPct = shorts.length
    ? (shorts.filter((t) => t.profit > 0).length / shorts.length) * 100
    : 0;

  // max consecutive wins / losses
  let maxConsecWins = 0;
  let maxConsecLosses = 0;
  let curWins = 0;
  let curLosses = 0;
  for (const t of closed) {
    if (t.profit > 0) {
      curWins += 1;
      curLosses = 0;
      maxConsecWins = Math.max(maxConsecWins, curWins);
    } else if (t.profit < 0) {
      curLosses += 1;
      curWins = 0;
      maxConsecLosses = Math.max(maxConsecLosses, curLosses);
    } else {
      curWins = 0;
      curLosses = 0;
    }
  }

  // max drawdown on cumulative equity curve
  let cum = 0;
  let peak = 0;
  let maxDrawdown = 0;
  for (const t of closed) {
    cum += t.profit;
    if (cum > peak) peak = cum;
    const dd = peak - cum;
    if (dd > maxDrawdown) maxDrawdown = dd;
  }

  return {
    total,
    winCount,
    lossCount,
    winRate,
    profitFactor,
    netProfit,
    avgWin,
    avgLoss,
    bestTrade,
    worstTrade,
    expectancy,
    longsWonPct,
    shortsWonPct,
    maxConsecWins,
    maxConsecLosses,
    maxDrawdown,
  };
}

function fmtMoney(n: number) {
  const sign = n < 0 ? "-" : "";
  return `${sign}$${Math.abs(n).toFixed(2)}`;
}

export function TradeStats({ trades }: { trades: any[] | undefined }) {
  const closed = (trades ?? []).filter((t) => t && t.closed_at);

  if (!closed.length) {
    return (
      <section className={styles.card}>
        <div className={styles.header}>Performance</div>
        <div className={styles.empty}>
          No closed trades yet — stats appear once trades close
        </div>
      </section>
    );
  }

  const s = computeStats(closed);
  const winPct = s.winRate * 100;
  const lossPct = 100 - winPct;

  const signClass = (n: number) =>
    n > 0 ? styles.pos : n < 0 ? styles.neg : undefined;

  const pfDisplay = s.profitFactor === Infinity ? "∞" : s.profitFactor.toFixed(2);

  const tiles: { label: string; value: string; cls?: string }[] = [
    { label: "Total Trades", value: String(s.total) },
    { label: "Wins", value: String(s.winCount), cls: styles.pos },
    { label: "Losses", value: String(s.lossCount), cls: styles.neg },
    { label: "Win Rate", value: `${winPct.toFixed(1)}%` },
    { label: "Profit Factor", value: pfDisplay, cls: signClass(s.profitFactor - 1 > 0 ? 1 : s.profitFactor < 1 ? -1 : 0) },
    { label: "Net Profit", value: fmtMoney(s.netProfit), cls: signClass(s.netProfit) },
    { label: "Avg Win", value: fmtMoney(s.avgWin), cls: styles.pos },
    { label: "Avg Loss", value: fmtMoney(-s.avgLoss), cls: styles.neg },
    { label: "Best Trade", value: fmtMoney(s.bestTrade), cls: styles.pos },
    { label: "Worst Trade", value: fmtMoney(s.worstTrade), cls: styles.neg },
    { label: "Expectancy", value: fmtMoney(s.expectancy), cls: signClass(s.expectancy) },
    { label: "Longs Won", value: `${s.longsWonPct.toFixed(0)}%` },
    { label: "Shorts Won", value: `${s.shortsWonPct.toFixed(0)}%` },
    { label: "Max Cons. Wins", value: String(s.maxConsecWins), cls: styles.pos },
    { label: "Max Cons. Losses", value: String(s.maxConsecLosses), cls: styles.neg },
    { label: "Max Drawdown", value: fmtMoney(-s.maxDrawdown), cls: styles.neg },
  ];

  return (
    <section className={styles.card}>
      <div className={styles.header}>Performance</div>
      <div className={styles.body}>
        <div className={styles.ratioBar}>
          <div className={styles.ratioWin} style={{ width: `${winPct}%` }} />
          <div className={styles.ratioLoss} style={{ width: `${lossPct}%` }} />
        </div>
        <div className={styles.ratioLabels}>
          <span className={styles.pos}>{winPct.toFixed(1)}% win</span>
          <span className={styles.neg}>{lossPct.toFixed(1)}% loss</span>
        </div>
        <div className={styles.grid}>
          {tiles.map((t) => (
            <div key={t.label} className={styles.tile}>
              <span className={styles.tileLabel}>{t.label}</span>
              <span className={`${styles.tileValue} ${t.cls ?? ""}`}>{t.value}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
