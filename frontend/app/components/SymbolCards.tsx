"use client";

import styles from "./SymbolCards.module.css";

const SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"];

const STATE_COLOR: Record<string, string> = {
  EXECUTED: "var(--green)",
  APPROVED: "var(--accent)",
  VETOED: "var(--red)",
  RISK_REJECTED: "var(--red)",
  SIGNAL: "var(--muted)",
  GATED: "var(--muted)",
  PROPOSED: "var(--yellow)",
  EXPIRED: "var(--muted)",
  MANAGED: "var(--green)",
  CLOSED: "var(--purple)",
  ANALYZED: "var(--purple)",
};

export function SymbolCards({ decisions }: { decisions: any[] | undefined }) {
  return (
    <div className={styles.grid}>
      {SYMBOLS.map((sym) => {
        const last = decisions?.find((d: any) => d.symbol === sym);
        const conf = last?.venus_signal?.confidence;
        const dir = last?.venus_signal?.direction;
        const state = last?.state;
        const ares = last?.ares_proposal;
        return (
          <div key={sym} className={styles.card}>
            <div className={styles.top}>
              <span className={styles.sym}>{sym}</span>
              {dir && (
                <span className={`${styles.dir} ${dir === "BUY" ? styles.buy : styles.sell}`}>
                  {dir === "BUY" ? "▲" : "▼"} {dir}
                </span>
              )}
            </div>

            <div className={styles.conf}>
              {conf != null ? (
                <>
                  <div className={styles.confBar}>
                    <div
                      className={styles.confFill}
                      style={{ width: `${(conf * 100).toFixed(0)}%`,
                               background: conf >= 0.65 ? "var(--green)" : "var(--muted)" }}
                    />
                  </div>
                  <span className={styles.confVal}>{(conf * 100).toFixed(0)}%</span>
                </>
              ) : <span className={styles.noSig}>no signal</span>}
            </div>

            {ares && (
              <div className={styles.levels}>
                <span className={styles.level}>E {ares.entry?.toFixed(5)}</span>
                <span className={styles.level} style={{ color: "var(--red)" }}>SL {ares.stop_loss?.toFixed(5)}</span>
                <span className={styles.level} style={{ color: "var(--green)" }}>TP {ares.take_profit?.toFixed(5)}</span>
                <span className={styles.rr}>R:R {ares.risk_reward ?? "—"}</span>
              </div>
            )}

            {state && (
              <div className={styles.state} style={{ color: STATE_COLOR[state] ?? "var(--muted)" }}>
                {state}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
