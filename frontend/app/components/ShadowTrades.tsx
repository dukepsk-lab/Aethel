"use client";

import styles from "./TradeHistory.module.css";
import clsx from "clsx";

export function ShadowTrades({ trades }: { trades: any[] | undefined }) {
  const total = trades?.reduce((s: number, t: any) => s + (t.hypothetical_pnl ?? 0), 0) ?? 0;

  return (
    <div className={styles.feed}>
      <div className={styles.header}>
        <span>Shadow Trades (hypothetical)</span>
        {!!trades?.length && (
          <span className={styles.count}>
            {trades.length} orders ·{" "}
            <span className={total >= 0 ? styles.profit : styles.loss}>
              {total >= 0 ? "+" : ""}{total.toFixed(2)}
            </span>
          </span>
        )}
      </div>

      <div className={styles.list}>
        {!trades?.length && (
          <div className={styles.empty}>
            No shadow orders yet — waiting for a signal to pass all gates
          </div>
        )}
        {!!trades?.length && (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th></th>
                <th className={styles.num}>Lots</th>
                <th className={styles.num}>Entry</th>
                <th className={styles.num}>Now</th>
                <th className={styles.num}>Hypo P&L</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t: any) => (
                <tr key={t.decision_id}>
                  <td className={styles.time}>
                    {new Date(t.created_at).toLocaleString([], {
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
                  <td className={styles.num}>{t.current_price?.toFixed(5) ?? "—"}</td>
                  <td className={clsx(styles.num, styles.pnl,
                      (t.hypothetical_pnl ?? 0) >= 0 ? styles.profit : styles.loss)}>
                    {t.hypothetical_pnl != null
                      ? `${t.hypothetical_pnl >= 0 ? "+" : ""}${t.hypothetical_pnl.toFixed(2)}`
                      : "—"}
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
