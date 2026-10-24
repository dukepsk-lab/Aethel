# Aethel: Multi-Symbol MT5 Execution Gateway & Real-Time Dashboard

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Async%20Gateway-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black.svg)](https://nextjs.org/)
[![Execution](https://img.shields.io/badge/Engine-Direct%20MT5%20C--API-orange.svg)](https://www.metatrader5.com/)

> **Asynchronous multi-asset execution gateway, signal verification firewall, and operations dashboard bridging quantitative models directly into MetaTrader 5.**

---

## 1. System Topology

```
                         AETHEL SYSTEM TOPOLOGY
 ┌─────────────────────┐      ┌─────────────────────────┐
 │ Next.js 14 Web UI   │ <==> │ FastAPI Async Engine    │
 │ (ModelStatus, Feed) │      │ (Signal Gates & Auth)   │
 └─────────────────────┘      └─────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ Aethel Orchestrator     │
                              │ (State & Telemetry Log) │
                              └─────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ Direct MT5 API Gateway  │
                              │ (Sub-millisecond Deals) │
                              └─────────────────────────┘
```

### Key Engineering Features:
- **Signal Gate Firewall (`test_signal_gate.py`):** Pre-trade validation intercepting duplicate orders, stale quotes ($> 200\text{ms}$), and invalid risk-reward ratios.
- **Direct MT5 Factory (`aethel/mt5/`):** High-throughput C-Python bridge providing thread-safe order routing.
- **Modern Observability UI (`frontend/`):** Next.js 14 console rendering real-time model health, signal feeds, and broker connection status.

---

## 2. Quick Start

```bash
# 1. Start Backend API Gateway
python -m aethel.api.main

# 2. Launch Operations Console
cd frontend && npm run dev
```
