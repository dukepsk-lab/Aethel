"use client";

import styles from "./AgentDebate.module.css";

export function AgentDebate({ decision, onClose }: { decision: any; onClose: () => void }) {
  const v = decision.venus_signal;
  const a = decision.ares_proposal;
  const th = decision.athena_decision;
  const risk = decision.risk_outcome;
  const exec = decision.execution;

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span>
          Agent Debate — {decision.symbol}{" "}
          <span className={styles.id}>{decision.decision_id.slice(0, 8)}</span>
        </span>
        <button className={styles.close} onClick={onClose}>✕</button>
      </div>

      <div className={styles.scroll}>

        {/* VENUS */}
        <Section icon="⊕" title="Venus (ML Signal)" color="var(--accent)">
          <Row label="Direction" value={v?.direction} color={v?.direction === "BUY" ? "var(--green)" : "var(--red)"} />
          <Row label="Confidence" value={v?.confidence != null ? `${(v.confidence * 100).toFixed(1)}%` : "—"} />
          <Row label="Model" value={v?.model_version} />
          {v?.features && Object.entries(v.features).slice(0, 6).map(([k, val]) => (
            <Row key={k} label={k} value={(val as number).toFixed(4)} />
          ))}
        </Section>

        {/* ARES */}
        {a ? (
          <Section icon="🗡" title="Ares (DeepSeek) — Proposal" color="var(--yellow)">
            <Row label="Action" value={a.action} color={a.action === "BUY" ? "var(--green)" : "var(--red)"} />
            <Row label="Entry" value={a.entry?.toFixed(5)} />
            <Row label="Stop Loss" value={a.stop_loss?.toFixed(5)} color="var(--red)" />
            <Row label="Take Profit" value={a.take_profit?.toFixed(5)} color="var(--green)" />
            <Row label="Risk %" value={`${a.risk_pct}%`} />
            <Row label="Risk:Reward" value={a.risk_reward} />
            {a.rationale && <Prose text={a.rationale} />}
          </Section>
        ) : (
          <Section icon="🗡" title="Ares — No Proposal" color="var(--muted)">
            <div className={styles.na}>Signal did not reach Ares (gated or error)</div>
          </Section>
        )}

        {/* ATHENA */}
        {th ? (
          <Section
            icon="🦉"
            title={`Athena (Claude) — ${th.verdict}`}
            color={th.verdict === "APPROVE" ? "var(--green)" : "var(--red)"}
          >
            <Row
              label="Verdict"
              value={th.verdict}
              color={th.verdict === "APPROVE" ? "var(--green)" : "var(--red)"}
            />
            {th.veto_reason && <Row label="Veto reason" value={th.veto_reason} color="var(--red)" />}
            {th.athena_rationale && <Prose text={th.athena_rationale} />}
          </Section>
        ) : (
          <Section icon="🦉" title="Athena — Pending" color="var(--muted)">
            <div className={styles.na}>Awaiting review</div>
          </Section>
        )}

        {/* RISK GATE */}
        {risk && (
          <Section
            icon="🛡"
            title={`Risk Gate — ${decision.state === "RISK_REJECTED" ? "REJECTED" : "PASSED"}`}
            color={decision.state === "RISK_REJECTED" ? "var(--red)" : "var(--green)"}
          >
            {risk.rule && <Row label="Failed rule" value={risk.rule} color="var(--red)" />}
            {risk.detail && <Row label="Detail" value={risk.detail} />}
            {risk.lots && <Row label="Lots computed" value={risk.lots} />}
            {risk.symbol && <Row label="Symbol" value={risk.symbol} />}
          </Section>
        )}

        {/* HERMES */}
        {exec && (
          <Section
            icon="⚡"
            title={`Hermes — ${exec.shadow ? "SHADOW" : exec.success ? "EXECUTED" : "FAILED"}`}
            color={exec.shadow ? "var(--muted)" : exec.success ? "var(--green)" : "var(--red)"}
          >
            {exec.shadow && <Row label="Mode" value="Shadow — order logged, not sent" color="var(--muted)" />}
            {exec.ticket && <Row label="MT5 ticket" value={exec.ticket} />}
            {exec.retcode && <Row label="MT5 retcode" value={exec.retcode} />}
            {exec.detail && <Row label="Detail" value={exec.detail} />}
          </Section>
        )}

      </div>
    </div>
  );
}

function Section({ icon, title, color, children }: {
  icon: string; title: string; color: string; children: React.ReactNode;
}) {
  return (
    <div className={styles.section}>
      <div className={styles.sectionHead} style={{ borderLeftColor: color }}>
        <span>{icon}</span>
        <span style={{ color }}>{title}</span>
      </div>
      <div className={styles.sectionBody}>{children}</div>
    </div>
  );
}

function Row({ label, value, color }: { label: string; value: any; color?: string }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}</span>
      <span className={styles.rowVal} style={{ color }}>{String(value ?? "—")}</span>
    </div>
  );
}

function Prose({ text }: { text: string }) {
  return <p className={styles.prose}>{text}</p>;
}
