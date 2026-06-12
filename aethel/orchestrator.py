"""The Aethel orchestrator — drives one full pipeline pass per symbol:

    MT5 data -> Venus -> Signal Gate -> Ares -> Athena -> Risk Gate -> Hermes

and separately: trade management ticks, closed-trade detection and the
async Mnemosyne post-trade loop.

Failure policy: ANY exception in the agent path resolves to no trade.
There is no retry-into-execution and no default approval.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from aethel.agents.ares import Ares
from aethel.agents.athena import Athena
from aethel.agents.llm import LLMClient, LLMUnavailable
from aethel.agents.mnemosyne import Mnemosyne
from aethel.config import SYMBOLS, get_settings
from aethel.core.lifecycle import TradeState
from aethel.core.schemas import RiskRejection, Timeframe, Verdict, utcnow
from aethel.db.memory import MemoryStore
from aethel.db.models import Decision
from aethel.db.risk_state import RiskStateStore
from aethel.db.session import get_sessionmaker
from aethel.hermes.executor import Hermes
from aethel.hermes.management import TradeManager
from aethel.mt5 import MT5Client
from aethel.news.calendar import JsonFeedCalendar, NewsService
from aethel.observability.alerts import send_alert
from aethel.observability.metrics import metrics
from aethel.risk_gate.gate import RiskGate
from aethel.signal_gate import SignalGate
from aethel.venus.inference import VenusInference

log = structlog.get_logger("orchestrator")

# enough history for feature warm-up (fracdiff/ATR percentile ~500 bars) + model window
CANDLE_COUNTS = {Timeframe.M5: 700, Timeframe.M15: 700, Timeframe.H1: 600}


class Orchestrator:
    def __init__(self, mt5: MT5Client, artifacts_dir: Path = Path("models/artifacts")):
        self.s = get_settings()
        self.mt5 = mt5
        llm = LLMClient()
        self.ares = Ares(llm)
        self.athena = Athena(llm)
        self.mnemosyne = Mnemosyne(llm)
        self.signal_gate = SignalGate()
        self.news = NewsService(JsonFeedCalendar())
        self.risk_gate = RiskGate(self.news)
        self.hermes = Hermes(mt5)
        self.trade_manager = TradeManager(mt5)
        self.venus: dict[str, VenusInference] = {}
        for symbol in SYMBOLS:
            try:
                self.venus[symbol] = VenusInference(artifacts_dir, symbol)
            except FileNotFoundError:
                log.warning("venus_artifact_missing", symbol=symbol)

    async def run_forever(self) -> None:
        log.info("aethel_started", shadow_mode=self.s.shadow_mode, symbols=SYMBOLS)
        while True:
            for symbol in self.venus:
                try:
                    await self.process_symbol(symbol)
                except Exception as e:
                    # fail closed: log, alert, move on — never trade through an error
                    log.error("pipeline_error", symbol=symbol, error=str(e))
                    metrics.incr("pipeline_errors")
            try:
                await self.trade_manager.manage_all()
            except Exception as e:
                log.error("management_error", error=str(e))
            await asyncio.sleep(300)  # one pass per M5 close

    async def process_symbol(self, symbol: str) -> None:
        # 1. Venus signal
        with metrics.time_stage("venus"):
            candles = {
                tf: await self.mt5.get_candles(symbol, tf, count)
                for tf, count in CANDLE_COUNTS.items()
            }
            signal = self.venus[symbol].predict(candles)
        metrics.incr("venus_signals")

        # 2. Signal Gate
        consult, reason = self.signal_gate.should_consult_agents(signal)
        if not consult:
            log.debug("gate_blocked", symbol=symbol, reason=reason)
            metrics.incr("gate_blocked")
            return
        self.signal_gate.record_consultation(symbol)

        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            risk_store = RiskStateStore(session)
            memory = MemoryStore(session)

            account = await self.mt5.get_account()
            positions = await self.mt5.get_positions()
            daily_pnl = await risk_store.daily_pnl_pct(account.equity)

            # kill-switch trip check happens every pass, before any LLM spend
            if daily_pnl <= -self.s.max_daily_loss_pct and not await risk_store.is_tripped():
                await risk_store.trip_kill_switch(f"daily loss {daily_pnl:.2f}%")
                await send_alert(f"🛑 KILL SWITCH tripped: daily loss {daily_pnl:.2f}%")
            if await risk_store.is_tripped():
                metrics.incr("kill_switch_blocks")
                return

            # 3. Ares proposes
            try:
                with metrics.time_stage("ares"):
                    tick = await self.mt5.get_tick(symbol)
                    notes = await memory.recall(
                        symbol, " ".join(f"{k}={v}" for k, v in signal.features.items()))
                    proposal = await self.ares.propose(
                        signal, candles[Timeframe.M15], tick, notes)
            except (LLMUnavailable, ValueError) as e:
                log.warning("ares_failed_no_trade", symbol=symbol, error=str(e))
                metrics.incr("ares_failures")
                return  # fail closed

            # 4. Athena reviews
            try:
                with metrics.time_stage("athena"):
                    events = await self.news.upcoming_for_symbol(symbol)
                    decision = await self.athena.review(
                        signal, proposal, account, positions, daily_pnl,
                        [e.model_dump(mode="json") for e in events])
            except LLMUnavailable as e:
                log.warning("athena_failed_no_trade", symbol=symbol, error=str(e))
                metrics.incr("athena_failures")
                return  # fail closed — Athena unreachable means NO trade

            record = Decision(
                decision_id=decision.decision_id, symbol=symbol,
                state=TradeState.PROPOSED, venus_signal=signal.model_dump(mode="json"),
                ares_proposal=proposal.model_dump(mode="json"),
                athena_decision=decision.model_dump(mode="json"),
                created_at=utcnow(), updated_at=utcnow(),
            )
            session.add(record)

            if decision.verdict == Verdict.VETO:
                record.state = TradeState.VETOED
                metrics.incr("athena_veto")
                log.info("athena_veto", symbol=symbol, reason=decision.veto_reason)
                await session.commit()
                return
            metrics.incr("athena_approve")
            record.state = TradeState.APPROVED

            # 5. Risk Gate — fresh tick, hard limits
            with metrics.time_stage("risk_gate"):
                tick = await self.mt5.get_tick(symbol)
                outcome = await self.risk_gate.validate(
                    decision, account, positions, tick, daily_pnl,
                    await risk_store.trades_today(), await risk_store.is_tripped())

            if isinstance(outcome, RiskRejection):
                record.state = TradeState.RISK_REJECTED
                record.risk_outcome = outcome.model_dump()
                metrics.incr(f"risk_reject_{outcome.rule}")
                log.info("risk_rejected", symbol=symbol, rule=outcome.rule,
                         detail=outcome.detail)
                await session.commit()
                return
            record.state = TradeState.RISK_CHECKED
            record.risk_outcome = outcome.model_dump()

            # 6. Hermes executes
            with metrics.time_stage("hermes"):
                result = await self.hermes.execute(outcome)
            record.execution = result.model_dump()
            if result.success:
                record.state = TradeState.EXECUTED
                record.ticket = result.ticket
                await risk_store.record_trade()
                metrics.incr("orders_shadow" if result.shadow else "orders_live")
            else:
                metrics.incr("orders_failed")
                await send_alert(f"⚠️ Order failed {symbol}: {result.detail}")
            await session.commit()
            log.info("pipeline_complete", symbol=symbol, decision_id=decision.decision_id,
                     executed=result.success, shadow=result.shadow)

    async def post_trade_analysis(self, decision_id: str, trade_context: dict) -> None:
        """Runs async after a trade closes — never blocks the signal loop."""
        try:
            lesson = await self.mnemosyne.analyze(trade_context)
        except LLMUnavailable as e:
            log.warning("mnemosyne_failed", decision_id=decision_id, error=str(e))
            return
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            memory = MemoryStore(session)
            await memory.store(
                decision_id=decision_id,
                symbol=trade_context["symbol"],
                won=trade_context["profit"] > 0,
                profit=trade_context["profit"],
                lesson=lesson.lesson,
                context_tags=lesson.context_tags,
            )
            record = await session.get(Decision, decision_id)
            if record:
                record.state = TradeState.ANALYZED
                record.updated_at = utcnow()
                await session.commit()
        metrics.incr("trades_analyzed")
