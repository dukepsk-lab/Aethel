# Aethel

A hybrid trading system for MetaTrader 5: an ML model produces the signal, a small group of LLM agents review it, and a hard-coded risk gate decides what is allowed to execute. The design question was how to use LLMs in a trading loop without letting them touch risk limits.

## How a trade flows

```
Venus (ML signal)  ->  Ares (proposes setup)  ->  Athena (risk review, veto)
        -> Risk Gate (plain Python, fails closed)  ->  Hermes (executes)  ->  MT5
```

- **Venus** (`aethel/venus/`): feature engineering with fractional differentiation, isotonic probability calibration on out-of-fold predictions, drift detection (calibration drift and per-feature PSI), challenger models, retraining.
- **Helios** (`aethel/helios/`): selective triple-barrier labeling (only bars that pass a volume, trend and volatility filter are labeled), training and backtest.
- **Agents** (`aethel/agents/`): Ares proposes a setup and a risk percentage, never a lot size. Athena reviews it against news, drawdown and exposure and can veto. Apollo summarizes calendar sentiment and is advisory and fail-soft. Mnemosyne writes post-trade lessons into pgvector. Themis audits the week and only suggests; nothing is applied automatically.
- **Risk Gate** (`aethel/risk_gate/`): hard-coded checks (decision TTL, kill switch, daily loss, trade limits, news blackout, sizing). Nothing an LLM outputs can relax a rule, and every check fails closed.
- **Hermes** (`aethel/hermes/`): execution with no LLM calls. A decision id can execute only once, broker-side SL/TP are part of the order, and shadow mode validates and logs orders without sending them.
- **MT5 gateway** (`aethel/mt5/`): a small FastAPI service that runs on the Windows host beside the terminal.
- **Observability** (`aethel/observability/`): audit log, metrics, alerts, signal log.
- **Dashboard** (`frontend/`): Next.js console for model health, signal feed and broker status.

## Status

Shadow mode is the intended way to evaluate this, since the LLM agents cannot be backtested on history. There are no live or shadow results in this repository. The ML layer (Venus, Helios) has its own backtest code.

## Layout

```
aethel/   venus, helios, agents, risk_gate, hermes, mt5, news, observability, db, core, api
frontend/ Next.js dashboard
tests/    10 test files (signal gate, risk gate, sizing, leakage, drift, lifecycle, ...)
scripts/  model deployment
```

## Running it

Python 3.11 or newer. The database layer expects Postgres with pgvector (`docker-compose.yml`). Agent calls go through litellm, so API keys are needed for the LLM parts; see `.env.example`.

```
pip install -e ".[dev]"
python -m pytest -q
python -m aethel.api.main
cd frontend && npm install && npm run dev
```

Tests: 66 passed on a fresh install. On the Windows host, start the MT5 gateway with `uvicorn aethel.mt5.gateway_server:app` and set `AETHEL_MT5_GATEWAY_TOKEN`.
