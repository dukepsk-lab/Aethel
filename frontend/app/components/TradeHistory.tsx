"use client";

import styles from "./TradeHistory.module.css";
import clsx from "clsx";

export function TradeHistory({ trades }: { trades: any[] | undefined }) {
  const total = trades?.reduce((s: number, t: any) => s + t.profit, 0) ?? 0;
  const wins = trades?.filter((t: any) => t.profit > 0).length ?? 0;

  return (
    <div className={styles.feed}>
      <div className={styles.header}>
        <span>Trade History</span>
        {!!trades?.length && (
          <span className={styles.count}>
            {wins}/{trades.length} wins ·{" "}
            <span className={total >= 0 ? styles.profit : styles.loss}>
              {total >= 0 ? "+" : ""}{total.toFixed(2)}
            </span>
          </span>
        )}
      </div>

      <div className={styles.list}>
        {!trades?.length && (
          <div className={styles.empty}>No closed trades yet — shadow mode active</div>
        )}
        {!!trades?.length && (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Closed</th>
                <th>Symbol</th>
                <th></th>
                <th className={styles.num}>Lots</th>
                <th className={styles.num}>Entry</th>
                <th className={styles.num}>Exit</th>
                <th className={styles.num}>P&L</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any) => (
                <tr key={t.ticket}>
                  <td className={styles.time}>
                    {new Date(t.closed_at).toLocaleString([], {
                      month: "2-digit", day: "2-digit",
                      hour: "2-digit", minute: "2-digit",
                    })}
                  </td>
                  <td className={styles.sym}>{t.symbol}</td>
                  <td className={clsx(styles.dir, t.action === "BUY" ? styles.buy : styles.sell)}>
                    {t.action === "BUY" ? "▲" : "▼"}
                  </td>
                  <td className={styles.num}>{t.lots?.toFixed(2)}</td>
                  <td className={styles.num}>{t.entry?.toFixed(5)}</td>
                  <td className={styles.num}>{t.exit?.toFixed(5)}</td>
                  <td className={clsx(styles.num, styles.pnl, t.profit >= 0 ? styles.profit : styles.loss)}>
                    {t.profit >= 0 ? "+" : ""}{t.profit?.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
