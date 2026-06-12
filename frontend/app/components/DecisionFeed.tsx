"use client";

import styles from "./DecisionFeed.module.css";
import clsx from "clsx";

const STATE_BADGE: Record<string, { label: string; cls: string }> = {
  EXECUTED:       { label: "EXEC",    cls: "green" },
  APPROVED:       { label: "APPRVD",  cls: "blue" },
  VETOED:         { label: "VETO",    cls: "red" },
  RISK_REJECTED:  { label: "RISK ✗",  cls: "red" },
  PROPOSED:       { label: "PROP",    cls: "yellow" },
  RISK_CHECKED:   { label: "RISK ✓",  cls: "blue" },
  SIGNAL:         { label: "SIG",     cls: "muted" },
  GATED:          { label: "GATED",   cls: "muted" },
  EXPIRED:        { label: "EXP",     cls: "muted" },
  MANAGED:        { label: "MGD",     cls: "green" },
  CLOSED:         { label: "CLSD",    cls: "purple" },
  ANALYZED:       { label: "ANLZD",   cls: "purple" },
};

export function DecisionFeed({
  decisions,
  selectedId,
  onSelect,
}: {
  decisions: any[] | undefined;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className={styles.feed}>
      <div className={styles.header}>
        <span>Decision Feed</span>
        <span className={styles.count}>{decisions?.length ?? 0} decisions</span>
      </div>

      <div className={styles.list}>
        {!decisions?.length && (
          <div className={styles.empty}>Waiting for signals…</div>
        )}
        {decisions?.map((d: any) => {
          const badge = STATE_BADGE[d.state] ?? { label: d.state, cls: "muted" };
          const conf = d.venus_signal?.confidence;
          const dir = d.venus_signal?.direction;
          const isSelected = d.decision_id === selectedId;
          return (
            <div
              key={d.decision_id}
              className={clsx(styles.row, isSelected && styles.selected)}
              onClick={() => onSelect(d.decision_id)}
            >
              <div className={styles.rowTop}>
                <span className={styles.time}>
                  {new Date(d.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                </span>
                <span className={styles.sym}>{d.symbol}</span>
                {dir && (
                  <span className={clsx(styles.dir, dir === "BUY" ? styles.buy : styles.sell)}>
                    {dir === "BUY" ? "▲" : "▼"}
                  </span>
                )}
                {conf != null && (
                  <span className={styles.conf}>{(conf * 100).toFixed(0)}%</span>
                )}
                <span className={clsx(styles.badge, styles[badge.cls])}>{badge.label}</span>
              </div>

              {d.athena_decision?.veto_reason && (
                <div className={styles.veto}>⊘ {d.athena_decision.veto_reason}</div>
              )}
              {d.risk_outcome?.rule && d.state === "RISK_REJECTED" && (
                <div className={styles.veto}>⊘ risk: {d.risk_outcome.rule}</div>
              )}
              {d.ares_proposal && d.state !== "VETOED" && d.state !== "RISK_REJECTED" && (
                <div className={styles.levels}>
                  E {d.ares_proposal.entry?.toFixed(5)} &nbsp;
                  <span style={{ color: "var(--red)" }}>SL {d.ares_proposal.stop_loss?.toFixed(5)}</span> &nbsp;
                  <span style={{ color: "var(--green)" }}>TP {d.ares_proposal.take_profit?.toFixed(5)}</span> &nbsp;
                  R:R {d.ares_proposal.risk_reward}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
