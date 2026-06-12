"use client";

import useSWR from "swr";

const API = process.env.NEXT_PUBLIC_AETHEL_API ?? "http://localhost:8000";
const fetcher = (url: string) => fetch(url).then((r) => r.json());

export default function Dashboard() {
  const { data: risk } = useSWR(`${API}/risk/status`, fetcher, { refreshInterval: 5000 });
  const { data: metrics } = useSWR(`${API}/metrics`, fetcher, { refreshInterval: 5000 });
  const { data: decisions } = useSWR(`${API}/decisions?limit=20`, fetcher, { refreshInterval: 10000 });

  const tripKillSwitch = async () => {
    if (!confirm("Trip the kill switch? Trading stops until the next trading day.")) return;
    await fetch(`${API}/risk/kill-switch`, { method: "POST" });
  };

  return (
    <main style={{ padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>🏛 NEXMIND — Aethel Command Center</h1>

      <section style={{ display: "flex", gap: 24, margin: "16px 0" }}>
        <Stat label="Equity" value={risk?.equity?.toFixed(2) ?? "—"} />
        <Stat label="Daily PnL %" value={risk?.daily_pnl_pct?.toFixed(2) ?? "—"} />
        <Stat label="Trades today" value={risk?.trades_today ?? "—"} />
        <Stat label="Open positions" value={risk?.open_positions ?? "—"} />
        <Stat label="Veto rate" value={metrics?.veto_rate ?? "—"} />
      </section>

      <button
        onClick={tripKillSwitch}
        disabled={risk?.kill_switch_tripped}
        style={{ background: risk?.kill_switch_tripped ? "#444" : "#c0392b", color: "#fff",
                 border: 0, padding: "12px 24px", fontSize: 16, cursor: "pointer", borderRadius: 6 }}
      >
        {risk?.kill_switch_tripped ? "🛑 KILL SWITCH TRIPPED" : "🛑 KILL SWITCH"}
      </button>

      <h2 style={{ marginTop: 32 }}>Recent decisions</h2>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ textAlign: "left", color: "#888" }}>
            <th>Time</th><th>Symbol</th><th>State</th><th>Venus</th><th>Athena</th>
          </tr>
        </thead>
        <tbody>
          {decisions?.map((d: any) => (
            <tr key={d.decision_id} style={{ borderTop: "1px solid #222" }}>
              <td>{new Date(d.created_at).toLocaleTimeString()}</td>
              <td>{d.symbol}</td>
              <td>{d.state}</td>
              <td>{d.venus_signal?.confidence?.toFixed(2)} {d.venus_signal?.direction}</td>
              <td>{d.athena_decision?.verdict}{d.athena_decision?.veto_reason ? ` — ${d.athena_decision.veto_reason}` : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ background: "#141923", padding: "12px 20px", borderRadius: 8 }}>
      <div style={{ color: "#888", fontSize: 12 }}>{label}</div>
      <div style={{ fontSize: 22 }}>{value}</div>
    </div>
  );
}
