# Aethel 🏛

Hybrid algorithmic trading system: a multi-timeframe ML model (**Venus**) plus a
multi-agent LLM team trading **EURUSD, GBPUSD, USDJPY, XAUUSD** on MetaTrader 5,
running 24/5.

## Agent Team

| Agent | God of | Engine | Role |
|---|---|---|---|
| **Venus** | — | CNN + TCN (PyTorch) | Calibrated P(TP before SL) signal, per symbol |
| **Ares** 🗡 | War | DeepSeek (cloud API) | Proposes trade setups (action, entry, SL, TP, risk %) |
| **Athena** 🦉 | Wisdom | Claude Sonnet 4.6 (cloud API) | Chief Risk Officer with absolute veto |
| **Hermes** ⚡ | Messengers | Pure Python | Limit-order execution — zero LLM latency |
| **Mnemosyne** 📖 | Memory | Gemini (cloud API) | Post-trade lessons into pgvector for RAG |

## Pipeline

```
MT5 data ─> Venus ─> Signal Gate ─> Ares ─> Athena ─> RISK GATE ─> Hermes ─> MT5
 (M1/M15/H1)  │     (threshold,    (propose) (veto      (hardcoded   (limit order,
              │      cooldown,                power)     Python —     broker-side
              │      call budget)                        AI cannot    SL/TP, TTL,
              │                                          override)    idempotent)
              └── closed trades ──> Mnemosyne ──> pgvector (async, never blocks)
```

## Design invariants

1. **Fail closed.** Any LLM timeout, invalid JSON, or pipeline error resolves to
   *no trade*. There is no default-approve path.
2. **LLMs never compute lot size.** Ares proposes `risk_pct`; the Risk Gate
   computes lots deterministically from contract specs (`risk_gate/sizing.py`).
3. **Broker-side SL/TP on every order.** A dead bot process never leaves an
   unprotected position.
4. **Decisions expire.** Every approval carries a TTL and a max price-drift
   tolerance; stale decisions are discarded, never re-negotiated.
5. **The Risk Gate is sovereign.** Kill switch (daily loss, persisted in
   PostgreSQL across restarts), lot cap, net-USD-exposure limit, spread filter,
   news blackout, trade counts — none of it can be relaxed by an agent.
6. **Shadow mode first.** `AETHEL_SHADOW_MODE=true` runs the full pipeline and
   logs validated orders without sending them. LLM agents cannot be backtested
   historically — shadow mode on a demo feed is the honest validation:
   Venus backtest → shadow mode → demo account → small live.

## Topology

The `MetaTrader5` Python package is **Windows-only**. Default deployment:

- **Linux brain** — FastAPI (NEXMIND backend), Venus inference, agents,
  PostgreSQL+pgvector. Talks to the gateway over HTTP (`AETHEL_MT5_MODE=gateway`).
- **Windows gateway** — `aethel/mt5/gateway_server.py` next to the MT5 terminal,
  bearer-token protected, private network only.

Single-Windows-VPS deployment works too: set `AETHEL_MT5_MODE=direct`.

## Quick start

```bash
cp .env.example .env          # fill in API keys
docker compose up -d db
pip install -e ".[ml,dev]"
pytest                        # safety-layer tests
uvicorn aethel.api.main:app   # starts API + trading loop (shadow mode)
```

Train Venus per symbol (offline, before going live):

```bash
python -m aethel.venus.train --symbol EURUSD --data data/EURUSD_m1.parquet
```

Training uses triple-barrier labeling, purged walk-forward validation with
embargo, and isotonic calibration — see `aethel/venus/`.

## API (NEXMIND backend)

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + shadow-mode flag |
| `GET /metrics` | per-stage latency, veto rate, gate blocks |
| `GET /decisions` | full agent-debate audit trail per decision |
| `GET /trades` | closed trades |
| `GET /risk/status` | equity, daily PnL, kill-switch state |
| `POST /risk/kill-switch` | **manual kill switch** — the only write path |

## Project layout

```
aethel/
  config.py          hard limits, symbol specs, settings
  core/              schemas (typed stage contracts), lifecycle state machine
  mt5/               broker abstraction: direct (Windows) + HTTP gateway
  venus/             model, features, triple-barrier labels, calibration, training
  signal_gate.py     confidence threshold + cooldown + agent call budget
  news/              economic calendar ingestion (Athena's eyes, blackout source)
  agents/            Ares / Athena / Mnemosyne + fail-closed LLM clients
  risk_gate/         the ironclad checkpoint + deterministic sizing
  hermes/            execution (idempotent, shadow-capable) + rule-based management
  db/                PostgreSQL models, persisted risk state, pgvector memory
  observability/     metrics + Telegram alerts
  orchestrator.py    the pipeline loop
  api/               NEXMIND FastAPI backend
frontend/            NEXMIND Command Center (Next.js)
```

## Disclaimer

Trading foreign exchange and CFDs carries substantial risk of loss. This is a
research framework; run it in shadow mode and on demo accounts. Nothing here is
financial advice.
