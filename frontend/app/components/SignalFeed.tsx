"use client";

import styles from "./SignalFeed.module.css";
import clsx from "clsx";

export function SignalFeed({ signals }: { signals: any[] | undefined }) {
  return (
    <div className={styles.feed}>
      <div className={styles.header}>
        <span>Venus · Helios Signals</span>
        <span className={styles.count}>{signals?.length ?? 0} evaluations</span>
      </div>

      <div className={styles.list}>
        {!signals?.length && (
          <div className={styles.empty}>No signals evaluated yet…</div>
        )}
        {signals?.map((s: any, i: number) => (
          <div key={`${s.time}-${s.symbol}-${i}`} className={styles.row}>
            <div className={styles.rowTop}>
              <span className={styles.time}>
                {new Date(s.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
              </span>
              <span className={styles.sym}>{s.symbol}</span>
              <span className={clsx(styles.dir, s.direction === "BUY" ? styles.buy : styles.sell)}>
                {s.direction === "BUY" ? "▲" : "▼"}
              </span>

              {/* Venus confidence */}
              <span className={styles.modelLabel}>V</span>
              <span className={styles.conf}>{(s.confidence * 100).toFixed(1)}%</span>

              {/* Helios confidence (if available) */}
              {s.helios_confidence != null && (
                <>
                  <span className={styles.modelLabel} style={{ color: "var(--purple, #bc8cff)" }}>H</span>
                  <span className={clsx(styles.conf, s.helios_agreed ? styles.heliosOn : styles.heliosOff)}>
                    {(s.helios_confidence * 100).toFixed(1)}%
                    {s.helios_agreed ? " ✓" : " ✗"}
                  </span>
                </>
              )}
              {s.helios_confidence == null && (
                <span className={styles.heliosMissing}>H —</span>
              )}

              <span className={clsx(styles.badge, s.passed_gate ? styles.pass : styles.block)}>
                {s.passed_gate ? "PASSED" : "BLOCKED"}
              </span>
            </div>
            {!s.passed_gate && <div className={styles.reason}>⊘ {s.reason}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}
