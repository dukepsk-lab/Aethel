"use client";

import styles from "./OpenPositions.module.css";
import clsx from "clsx";

export function OpenPositions({ positions }: { positions: any[] | undefined }) {
  const floating = positions?.reduce((s: number, p: any) => s + p.profit, 0) ?? 0;

  return (
    <section className={styles.card}>
      <div className={styles.header}>
        <span>Open Positions</span>
        {!!positions?.length && (
          <span className={clsx(styles.floating, floating >= 0 ? styles.profit : styles.loss)}>
            {floating >= 0 ? "+" : ""}{floating.toFixed(2)} USD
          </span>
        )}
      </div>

      {!positions?.length && (
        <div className={styles.empty}>No open positions</div>
      )}
      {!!positions?.length && (
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Ticket</th>
              <th>Symbol</th>
              <th></th>
              <th className={styles.num}>Lots</th>
              <th className={styles.num}>Entry</th>
              <th className={styles.num}>SL</th>
              <th className={styles.num}>TP</th>
              <th className={styles.num}>P&L</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p: any) => (
              <tr key={p.ticket}>
                <td className={styles.ticket}>#{p.ticket}</td>
                <td className={styles.sym}>{p.symbol}</td>
                <td className={clsx(styles.dir, p.action === "BUY" ? styles.buy : styles.sell)}>
                  {p.action === "BUY" ? "▲ BUY" : "▼ SELL"}
                </td>
                <td className={styles.num}>{p.lots?.toFixed(2)}</td>
                <td className={styles.num}>{p.entry?.toFixed(5)}</td>
                <td className={clsx(styles.num, styles.sl)}>{p.sl?.toFixed(5)}</td>
                <td className={clsx(styles.num, styles.tp)}>{p.tp?.toFixed(5)}</td>
                <td className={clsx(styles.num, styles.pnl, p.profit >= 0 ? styles.profit : styles.loss)}>
                  {p.profit >= 0 ? "+" : ""}{p.profit?.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
