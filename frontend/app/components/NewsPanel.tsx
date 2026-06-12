"use client";

import styles from "./NewsPanel.module.css";
import clsx from "clsx";

const IMPACT_CLS: Record<string, string> = {
  high: "high", medium: "medium", low: "low",
};

export function NewsPanel({ news }: { news: any }) {
  const events = news?.events ?? [];
  const blackout = news?.blackout ?? {};
  const blocked = Object.entries(blackout).filter(([, v]) => v) as [string, string][];

  return (
    <section className={styles.card}>
      <div className={styles.header}>
        <span>Economic Calendar (12h)</span>
        <span className={styles.count}>{events.length} events</span>
      </div>

      {blocked.length > 0 && (
        <div className={styles.blackout}>
          🚫 NEWS BLACKOUT active:{" "}
          {blocked.map(([sym, title]) => `${sym} (${title})`).join(" · ")}
        </div>
      )}

      <div className={styles.list}>
        {!events.length && <div className={styles.empty}>No upcoming events in the next 12h</div>}
        {events.map((e: any, i: number) => (
          <div key={`${e.time}-${e.title}-${i}`} className={styles.row}>
            <span className={styles.time}>
              {new Date(e.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
            <span className={styles.ccy}>{e.currency}</span>
            <span className={clsx(styles.impact, styles[IMPACT_CLS[e.impact] ?? "low"])}>
              {e.impact === "high" ? "●●●" : e.impact === "medium" ? "●●" : "●"}
            </span>
            <span className={styles.title}>{e.title}</span>
            {(e.forecast || e.previous) && (
              <span className={styles.figures}>
                {e.forecast && `f ${e.forecast}`}{e.forecast && e.previous && " · "}
                {e.previous && `p ${e.previous}`}
              </span>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
