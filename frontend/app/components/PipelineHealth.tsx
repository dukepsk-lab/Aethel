"use client";
import styles from "./PipelineHealth.module.css";

export function PipelineHealth({ metrics, trades }: { metrics: any; trades: any[] | undefined }) {
  const c = metrics?.counters ?? {};
  const total = (c.athena_approve ?? 0) + (c.athena_veto ?? 0);
  const vetoRate = total > 0 ? Math.round((c.athena_veto / total) * 100) : null;
  const winRate = trades && trades.length > 0
    ? Math.round((trades.filter((t: any) => t.pnl > 0).length / trades.length) * 100)
    : null;
  const llmCosts = Object.entries(c)
    .filter(([k]) => k.startsWith("llm_cost"))
    .reduce((s, [, v]) => s + (v as number), 0);
  const totalCostUsd = (llmCosts / 1_000_000).toFixed(3);

  const items = [
    { label: "Signals today", value: c.venus_signals ?? 0 },
    { label: "Approved", value: c.athena_approve ?? 0, color: "var(--green)" },
    { label: "Vetoed", value: c.athena_veto ?? 0, color: "var(--red)" },
    { label: "Veto rate", value: vetoRate != null ? `${vetoRate}%` : "—" },
    { label: "Win rate", value: winRate != null ? `${winRate}%` : "—", color: winRate != null ? (winRate >= 50 ? "var(--green)" : "var(--red)") : undefined },
    { label: "LLM cost", value: `$${totalCostUsd}` },
    { label: "Orders live", value: c.orders_live ?? 0, color: c.orders_live > 0 ? "var(--green)" : undefined },
    { label: "Errors", value: c.pipeline_errors ?? 0, color: c.pipeline_errors > 0 ? "var(--red)" : undefined },
  ];

  return (
    <div className={styles.bar}>
      {items.map(({ label, value, color }) => (
        <div key={label} className={styles.item}>
          <span className={styles.label}>{label}</span>
          <span className={styles.value} style={{ color }}>{value}</span>
        </div>
      ))}
    </div>
  );
}
