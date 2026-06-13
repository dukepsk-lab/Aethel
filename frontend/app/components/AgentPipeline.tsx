"use client";
import styles from "./AgentPipeline.module.css";

const STAGES = [
  { id: "venus",    label: "Venus",    icon: "🔮", desc: "ML Signal" },
  { id: "ares",     label: "Ares",     icon: "⚔️",  desc: "Proposal" },
  { id: "apollo",   label: "Apollo",   icon: "🕊",  desc: "Sentiment" },
  { id: "athena",   label: "Athena",   icon: "🦉",  desc: "Review" },
  { id: "risk",     label: "Risk Gate",icon: "🛡",  desc: "Limits" },
  { id: "hermes",   label: "Hermes",   icon: "⚡",  desc: "Execute" },
];

export function AgentPipeline({ metrics, decisions }: { metrics: any; decisions: any[] | undefined }) {
  const c = metrics?.counters ?? {};
  const lat = metrics?.stage_latency ?? {};

  // derive per-stage activity from counters
  const stageData: Record<string, { count: number; avgMs: number | null; active: boolean }> = {
    venus:  { count: c.venus_signals ?? 0,   avgMs: lat.venus?.avg_ms ?? null,    active: (c.venus_signals ?? 0) > 0 },
    ares:   { count: (c.athena_approve ?? 0) + (c.athena_veto ?? 0) + (c.ares_failures ?? 0), avgMs: lat.ares?.avg_ms ?? null, active: (lat.ares?.avg_ms ?? 0) > 0 },
    apollo: { count: c.apollo_briefs ?? 0,   avgMs: lat.apollo?.avg_ms ?? null,   active: (c.apollo_briefs ?? 0) > 0 },
    athena: { count: (c.athena_approve ?? 0) + (c.athena_veto ?? 0), avgMs: lat.athena?.avg_ms ?? null, active: ((c.athena_approve ?? 0) + (c.athena_veto ?? 0)) > 0 },
    risk:   { count: (c.athena_approve ?? 0), avgMs: lat.risk_gate?.avg_ms ?? null, active: (lat.risk_gate?.avg_ms ?? 0) > 0 },
    hermes: { count: (c.orders_live ?? 0) + (c.orders_shadow ?? 0), avgMs: lat.hermes?.avg_ms ?? null, active: ((c.orders_live ?? 0) + (c.orders_shadow ?? 0)) > 0 },
  };

  return (
    <section className={styles.wrap}>
      <div className={styles.header}>Agent Pipeline</div>
      <div className={styles.pipeline}>
        {STAGES.map((stage, i) => {
          const d = stageData[stage.id];
          return (
            <div key={stage.id} className={styles.nodeWrap}>
              <div className={`${styles.node} ${d.active ? styles.active : ""}`}>
                <span className={styles.icon}>{stage.icon}</span>
                <span className={styles.name}>{stage.label}</span>
                <span className={styles.desc}>{stage.desc}</span>
                <span className={styles.count}>{d.count > 0 ? d.count : "—"}</span>
                {d.avgMs && <span className={styles.latency}>{d.avgMs}ms</span>}
              </div>
              {i < STAGES.length - 1 && (
                <div className={`${styles.arrow} ${d.active ? styles.arrowActive : ""}`}>→</div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
