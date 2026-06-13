"use client";

import { useState } from "react";
import useSWR from "swr";
import { EquityChart } from "./components/EquityChart";
import { DecisionFeed } from "./components/DecisionFeed";
import { SignalFeed } from "./components/SignalFeed";
import { TradeHistory } from "./components/TradeHistory";
import { OpenPositions } from "./components/OpenPositions";
import { ShadowTrades } from "./components/ShadowTrades";
import { NewsPanel } from "./components/NewsPanel";
import { AgentDebate } from "./components/AgentDebate";
import { MetricsPanel } from "./components/MetricsPanel";
import { SymbolCards } from "./components/SymbolCards";
import { KillSwitch } from "./components/KillSwitch";
import { PipelineHealth } from "./components/PipelineHealth";
import { AgentPipeline } from "./components/AgentPipeline";
import { API, fetcher } from "./lib/api";
import styles from "./page.module.css";

type FeedTab = "decisions" | "signals" | "shadow" | "trades";

export default function Dashboard() {
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null);
  const [feedTab, setFeedTab] = useState<FeedTab>("decisions");

  const { data: risk, mutate: mutateRisk } = useSWR(
    `${API}/risk/status`, fetcher, { refreshInterval: 5000 }
  );
  const { data: metrics } = useSWR(
    `${API}/metrics`, fetcher, { refreshInterval: 5000 }
  );
  const { data: decisions } = useSWR(
    `${API}/decisions?limit=50`, fetcher, { refreshInterval: 8000 }
  );
  const { data: trades } = useSWR(
    `${API}/trades?limit=100`, fetcher, { refreshInterval: 15000 }
  );
  const { data: signals } = useSWR(
    `${API}/signals?limit=100`, fetcher, { refreshInterval: 8000 }
  );
  const { data: positions } = useSWR(
    `${API}/positions`, fetcher, { refreshInterval: 5000 }
  );
  const { data: shadowTrades } = useSWR(
    `${API}/shadow-trades?limit=50`, fetcher, { refreshInterval: 10000 }
  );
  const { data: news } = useSWR(
    `${API}/news`, fetcher, { refreshInterval: 60000 }
  );

  const selectedDecision = decisions?.find((d: any) => d.decision_id === selectedDecisionId);

  return (
    <div className={styles.shell}>
      {/* ── TOP BAR ── */}
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <span className={styles.logo}>⚡</span>
          <span className={styles.brandName}>Aethel</span>
          <span className={styles.brandSub}>Command Center</span>
        </div>

        <div className={styles.statusRow}>
          {[
            { label: "EQUITY", value: risk?.equity != null ? `$${risk.equity.toFixed(2)}` : "—" },
            {
              label: "DAILY P&L",
              value: risk?.daily_pnl_pct != null ? `${risk.daily_pnl_pct > 0 ? "+" : ""}${risk.daily_pnl_pct.toFixed(2)}%` : "—",
              color: risk?.daily_pnl_pct > 0 ? "var(--green)" : risk?.daily_pnl_pct < 0 ? "var(--red)" : undefined,
            },
            { label: "TRADES TODAY", value: risk?.trades_today ?? "—" },
            { label: "OPEN POS", value: risk?.open_positions ?? "—" },
            { label: "VETO RATE", value: metrics?.veto_rate != null ? `${(metrics.veto_rate * 100).toFixed(0)}%` : "—" },
          ].map(({ label, value, color }) => (
            <div key={label} className={styles.statusStat}>
              <span className={styles.statLabel}>{label}</span>
              <span className={styles.statValue} style={{ color }}>{value}</span>
            </div>
          ))}
        </div>

        <KillSwitch tripped={risk?.kill_switch_tripped} onTrip={mutateRisk} />
      </header>

      {/* ── SHADOW MODE BANNER ── */}
      {risk?.shadow_mode && !risk.kill_switch_tripped && (
        <div className={styles.shadowBanner}>
          <span>👁 SHADOW MODE — orders validated but not sent to broker</span>
        </div>
      )}
      {risk?.kill_switch_tripped && (
        <div className={styles.killBanner}>
          <span>🛑 KILL SWITCH ACTIVE — no new trades until next trading day reset</span>
        </div>
      )}

      {/* ── MAIN GRID ── */}
      <div className={styles.grid}>

        {/* LEFT COLUMN */}
        <div className={styles.leftCol}>
          <AgentPipeline metrics={metrics} decisions={decisions} />
          <SymbolCards decisions={decisions} />
          <OpenPositions positions={positions} />
          <section className={styles.card}>
            <div className={styles.cardHeader}>Equity Curve</div>
            <EquityChart trades={trades} currentEquity={risk?.equity} />
          </section>
          <NewsPanel news={news} />
          <MetricsPanel metrics={metrics} />
        </div>

        {/* RIGHT COLUMN */}
        <div className={styles.rightCol}>
          <PipelineHealth metrics={metrics} trades={trades} />
          <div className={styles.tabs}>
            {([
              ["decisions", `Decisions${decisions?.length ? ` (${decisions.length})` : ""}`],
              ["signals", `Signals${signals?.length ? ` (${signals.length})` : ""}`],
              ["shadow", `Shadow${shadowTrades?.length ? ` (${shadowTrades.length})` : ""}`],
              ["trades", `Trades${trades?.length ? ` (${trades.length})` : ""}`],
            ] as [FeedTab, string][]).map(([tab, label]) => (
              <button
                key={tab}
                className={feedTab === tab ? styles.tabActive : styles.tab}
                onClick={() => setFeedTab(tab)}
              >
                {label}
              </button>
            ))}
          </div>
          {feedTab === "decisions" && (
            <DecisionFeed
              decisions={decisions}
              selectedId={selectedDecisionId}
              onSelect={setSelectedDecisionId}
            />
          )}
          {feedTab === "signals" && <SignalFeed signals={signals} />}
          {feedTab === "shadow" && <ShadowTrades trades={shadowTrades} />}
          {feedTab === "trades" && <TradeHistory trades={trades} />}
          {feedTab === "decisions" && selectedDecision && (
            <AgentDebate decision={selectedDecision} onClose={() => setSelectedDecisionId(null)} />
          )}
        </div>
      </div>
    </div>
  );
}
