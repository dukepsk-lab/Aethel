"use client";

import styles from "./MetricsPanel.module.css";

export function MetricsPanel({ metrics }: { metrics: any }) {
  if (!metrics) return null;
  const c = metrics.counters ?? {};
  const lat = metrics.stage_latency ?? {};

  const stages = ["venus", "ares", "athena", "risk_gate", "hermes"];
  const costEntries = Object.entries(c)
    .filter(([k]) => k.startsWith("llm_cost"))
    .map(([k, v]) => {
      const prov = k.replace("llm_cost_usd_x1e6_", "");
      return { prov, usd: ((v as number) / 1_000_000).toFixed(4) };
    });

  return (
    <div className={styles.wrap}>
      {/* counters */}
      <div className={styles.card}>
        <div className={styles.header}>Signal Pipeline</div>
        <div className={styles.grid}>
          {[
            ["Venus signals", c.venus_signals],
            ["Gate blocked", c.gate_blocked],
            ["Ares failures", c.ares_failures],
            ["Athena approve", c.athena_approve],
            ["Athena veto", c.athena_veto],
            ["Risk rejected", Object.entries(c).filter(([k]) => k.startsWith("risk_reject")).reduce((s, [, v]) => s + (v as number), 0)],
            ["Orders shadow", c.orders_shadow],
            ["Orders live", c.orders_live],
            ["Kill switch blocks", c.kill_switch_blocks],
          ].map(([label, val]) => (
            <div key={label as string} className={styles.stat}>
              <span className={styles.label}>{label}</span>
              <span className={styles.val}>{val ?? 0}</span>
            </div>
          ))}
        </div>
      </div>

      {/* latency */}
      <div className={styles.card}>
        <div className={styles.header}>Stage Latency (ms)</div>
        <div className={styles.latGrid}>
          {stages.map((s) => {
            const d = lat[s];
            return (
              <div key={s} className={styles.latRow}>
                <span className={styles.stage}>{s}</span>
                <div className={styles.latBar}>
                  <div
                    className={styles.latFill}
                    style={{ width: d ? `${Math.min((d.avg_ms / 3000) * 100, 100)}%` : "0%" }}
                  />
                </div>
                <span className={styles.latVal}>{d ? `${d.avg_ms}` : "—"}</span>
                <span className={styles.latMax}>{d ? `max ${d.max_ms}` : ""}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* cost */}
      {costEntries.length > 0 && (
        <div className={styles.card}>
          <div className={styles.header}>LLM Cost (session)</div>
          <div className={styles.grid}>
            {costEntries.map(({ prov, usd }) => (
              <div key={prov} className={styles.stat}>
                <span className={styles.label}>{prov}</span>
                <span className={styles.val}>${usd}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
