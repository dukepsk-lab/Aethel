"""The Aethel orchestrator — drives one full pipeline pass per symbol:

    MT5 data -> Venus -> Signal Gate -> Ares -> Athena -> Risk Gate -> Hermes

and separately: trade management ticks, closed-trade detection and the
async Mnemosyne post-trade loop.

Failure policy: ANY exception in the agent path resolves to no trade.
There is no retry-into-execution and no default approval.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import structlog

from aethel.agents.apollo import Apollo
from aethel.agents.ares import Ares
from aethel.agents.athena import Athena
from aethel.agents.llm import LLMClient, LLMUnavailable
from aethel.agents.mnemosyne import Mnemosyne
from aethel.agents.themis import Themis
from aethel.config import SYMBOLS, get_settings
from aethel.core.lifecycle import TradeState
from aethel.core.regime import classify_regime
from aethel.core.schemas import Action, AresProposal, AthenaDecision, RiskRejection, Timeframe, Verdict, utcnow
from aethel.db.memory import MemoryStore
from aethel.db.models import Decision
from aethel.db.risk_state import RiskStateStore
from aethel.db.session import get_sessionmaker
from aethel.hermes.executor import Hermes
from aethel.hermes.management import TradeManager
from aethel.mt5 import MT5Client
from aethel.news.calendar import JsonFeedCalendar, NewsService
from aethel.observability.alerts import send_alert
from aethel.observability.audit import gather_weekly_stats
from aethel.observability.export import export_audit_markdown
from aethel.observability.metrics import metrics
from aethel.observability.signal_log import signal_log
from aethel.risk_gate.gate import RiskGate
from aethel.signal_gate import SignalGate
from aethel.venus.inference import VenusInference

log = structlog.get_logger("orchestrator")


def _market_is_open() -> bool:
    from datetime import timezone
    now = datetime.now(timezone.utc)
    wd = now.weekday()  # 0=Mon, 5=Sat, 6=Sun
    if wd == 5:  # Saturday
        return False
    if wd == 6 and now.hour < 22:  # Sunday before 22:00 UTC
        return False
    return True

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
        self.apollo = Apollo(llm, self.news)
        self.themis = Themis(llm)
        self.risk_gate = RiskGate(self.news)
        self.hermes = Hermes(mt5)
        self.trade_manager = TradeManager(mt5)
        self._market_was_open: bool = True
        self.venus: dict[str, VenusInference] = {}
        for symbol in SYMBOLS:
            try:
                self.venus[symbol] = VenusInference(artifacts_dir, symbol)
            except FileNotFoundError:
                log.warning("venus_artifact_missing", symbol=symbol)
        self.helios: dict[str, VenusInference] = {}
        if self.s.helios_enabled:
            helios_dir = Path(self.s.helios_artifacts_dir)
            for symbol in SYMBOLS:
                try:
                    self.helios[symbol] = VenusInference(helios_dir, symbol)
                except FileNotFoundError:
                    log.info("helios_artifact_missing", symbol=symbol)

    async def run_forever(self) -> None:
        log.info("aethel_started", shadow_mode=self.s.shadow_mode, symbols=SYMBOLS)
        await send_alert(f"✅ Aethel started — {'SHADOW' if self.s.shadow_mode else 'LIVE'} mode | symbols: {', '.join(SYMBOLS)}")
        asyncio.create_task(self._weekly_audit_loop())
        try:
            while True:
                is_open = _market_is_open()
                if self._market_was_open and not is_open:
                    await send_alert("🔒 Market closed — Aethel paused until Sunday 22:00 UTC")
                elif not self._market_was_open and is_open:
                    await send_alert("🔓 Market open — Aethel resuming")
                self._market_was_open = is_open
                if not is_open:
                    log.info("market_closed_sleeping", next_check_min=30)
                    await asyncio.sleep(1800)  # check again in 30 min
                    continue
                for symbol in self.venus:
                    try:
                        await self.process_symbol(symbol)
                    except Exception as e:
                        log.error("pipeline_error", symbol=symbol, error=str(e))
                        metrics.incr("pipeline_errors")
                try:
                    await self.trade_manager.manage_all()
                except Exception as e:
                    log.error("management_error", error=str(e))
                await asyncio.sleep(300)  # one pass per M5 close
        except (asyncio.CancelledError, KeyboardInterrupt):
            log.info("aethel_stopping")
            try:
                await send_alert("🔴 Aethel stopped — server shutdown")
            except Exception:
                pass
            raise

    async def process_symbol(self, symbol: str) -> None:
        # 1. Venus signal
        with metrics.time_stage("venus"):
            candles = {
                tf: await self.mt5.get_candles(symbol, tf, count)
                for tf, count in CANDLE_COUNTS.items()
            }
            signal = self.venus[symbol].predict(candles)
        metrics.incr("venus_signals")

        # Helios inference (fail-soft — missing model just means no consensus data)
        helios_conf: float | None = None
        helios_agreed: bool | None = None
        if symbol in self.helios:
            try:
                h_signal = self.helios[symbol].predict(candles)
                helios_conf = h_signal.confidence
                helios_agreed = helios_conf >= self.s.helios_confidence_threshold
            except Exception as e:
                log.warning("helios_inference_failed", symbol=symbol, error=str(e))

        # Annotate venus signal with helios info
        signal = signal.model_copy(update={
            "helios_confidence": helios_conf,
            "helios_agreed": helios_agreed,
        })

        # 2. Signal Gate
        consult, reason = self.signal_gate.should_consult_agents(signal)
        signal_log.record(signal, passed=consult, reason=reason)
        if not consult:
            log.debug("gate_blocked", symbol=symbol, reason=reason)
            metrics.incr("gate_blocked")
            return
        self.signal_gate.record_consultation(symbol)

        consensus_str = ""
        if helios_agreed is True:
            consensus_str = " | ✅ Helios agrees"
        elif helios_agreed is False:
            consensus_str = " | ⚠️ Helios disagrees"
        elif helios_agreed is None and self.s.helios_enabled:
            consensus_str = " | ❓ Helios unavailable"

        await send_alert(
            f"📡 Signal | {symbol} | {signal.direction.value} | conf={signal.confidence:.2f}{consensus_str}"
        )

        tier = self.signal_gate.get_tier(signal)

        # advisory context — failure here must never block the pipeline
        try:
            regime = classify_regime(candles[Timeframe.H1]).model_dump()
        except Exception as e:
            log.warning("regime_failed", symbol=symbol, error=str(e))
            regime = None

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

            if tier == "high_conf":
                # HIGH_CONF fast path — skip Ares and Athena, ATR-based SL/TP
                log.info("high_conf_fast_path", symbol=symbol, confidence=signal.confidence)
                await send_alert(
                    f"⚡ HIGH CONF {signal.direction.value} {symbol} conf={signal.confidence:.2f} — fast path, boosted lots"
                )
                metrics.incr("high_conf_trades")
                try:
                    tick = await self.mt5.get_tick(symbol)
                    entry = tick.ask if signal.direction == Action.BUY else tick.bid
                    h1_candles = candles[Timeframe.H1]
                    highs = [c.high for c in h1_candles[-20:]]
                    lows = [c.low for c in h1_candles[-20:]]
                    atr = float(np.mean([h - l for h, l in zip(highs, lows)]))
                    sl_dist = 2.0 * atr
                    tp_dist = 4.0 * atr
                    if signal.direction == Action.BUY:
                        stop_loss = entry - sl_dist
                        take_profit = entry + tp_dist
                    else:
                        stop_loss = entry + sl_dist
                        take_profit = entry - tp_dist
                    risk_pct = min(
                        self.s.base_risk_pct * self.s.high_confidence_lot_multiplier,
                        self.s.max_risk_pct_high_conf,
                    )
                    proposal = AresProposal(
                        symbol=symbol,
                        action=signal.direction,
                        entry=round(entry, 5),
                        stop_loss=round(stop_loss, 5),
                        take_profit=round(take_profit, 5),
                        risk_pct=risk_pct,
                        rationale=f"high-confidence fast path (conf={signal.confidence:.2f})",
                    )
                except Exception as e:
                    log.warning("high_conf_fast_path_failed_fallback_normal",
                                symbol=symbol, error=str(e))
                    tier = "normal"

                if tier == "high_conf":
                    decision = AthenaDecision(
                        verdict=Verdict.APPROVE,
                        proposal=proposal,
                        athena_rationale="high-confidence fast path — Athena bypassed",
                        venus_confidence=signal.confidence,
                        expires_at=utcnow() + timedelta(seconds=self.s.decision_ttl_seconds),
                    )
                    record = Decision(
                        decision_id=decision.decision_id, symbol=symbol,
                        state=TradeState.APPROVED, venus_signal=signal.model_dump(mode="json"),
                        ares_proposal=proposal.model_dump(mode="json"),
                        athena_decision={"verdict": "FAST_PATH", "veto_reason": None},
                        created_at=utcnow(), updated_at=utcnow(),
                    )
                    session.add(record)

            if tier == "normal":
                # 3. Ares proposes
                try:
                    with metrics.time_stage("ares"):
                        tick = await self.mt5.get_tick(symbol)
                        notes = await memory.recall(
                            symbol, " ".join(f"{k}={v}" for k, v in signal.features.items()),
                            regime=regime["label"] if regime else None)
                        proposal = await self.ares.propose(
                            signal, candles[Timeframe.M15], tick, notes, regime)
                except (LLMUnavailable, ValueError) as e:
                    log.warning("ares_failed_no_trade", symbol=symbol, error=str(e))
                    metrics.incr("ares_failures")
                    return  # fail closed

                # 4. Apollo sentiment (fail-soft, hourly-cached), then Athena reviews
                with metrics.time_stage("apollo"):
                    brief = await self.apollo.brief(list(self.venus))
                try:
                    with metrics.time_stage("athena"):
                        events = await self.news.upcoming_for_symbol(symbol)
                        decision = await self.athena.review(
                            signal, proposal, account, positions, daily_pnl,
                            [e.model_dump(mode="json") for e in events],
                            regime, brief.model_dump() if brief else None)
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
                mode = "SHADOW" if result.shadow else "LIVE"
                await send_alert(
                    f"{'👁' if result.shadow else '✅'} Trade {mode} | {symbol} | {decision.proposal.action} | ticket={result.ticket}"
                )
            else:
                metrics.incr("orders_failed")
                await send_alert(f"⚠️ Order failed {symbol}: {result.detail}")
            await session.commit()
            log.info("pipeline_complete", symbol=symbol, decision_id=decision.decision_id,
                     executed=result.success, shadow=result.shadow)

    async def _weekly_audit_loop(self) -> None:
        """Themis runs once a week at the Sunday rollover. Fail-soft: an
        audit failure is logged and skipped — never touches trading."""
        while True:
            now = utcnow()
            # next Sunday at daily_reset_hour_utc
            days_ahead = (6 - now.weekday()) % 7
            target = (now + timedelta(days=days_ahead)).replace(
                hour=self.s.daily_reset_hour_utc, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=7)
            await asyncio.sleep((target - now).total_seconds())
            try:
                account = await self.mt5.get_account()
                async with get_sessionmaker()() as session:
                    stats = await gather_weekly_stats(session, account.equity)
                audit = await self.themis.audit(stats)
                await send_alert(self.themis.format_report(audit))
                try:
                    export_audit_markdown(audit.model_dump())
                except Exception as e:  # markdown export is best-effort
                    log.warning("audit_export_failed", error=str(e))
                metrics.incr("themis_audits")
                log.info("themis_audit_complete", grade=audit.grade)
            except Exception as e:
                log.error("themis_audit_failed", error=str(e))
                metrics.incr("themis_failures")

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
            regime_ctx = trade_context.get("regime")
            await memory.store(
                decision_id=decision_id,
                symbol=trade_context["symbol"],
                won=trade_context["profit"] > 0,
                profit=trade_context["profit"],
                lesson=lesson.lesson,
                context_tags=lesson.context_tags,
                ares_quality=lesson.ares_quality,
                athena_quality=lesson.athena_quality,
                regime=regime_ctx.get("label") if isinstance(regime_ctx, dict) else regime_ctx,
            )
            record = await session.get(Decision, decision_id)
            if record:
                record.state = TradeState.ANALYZED
                record.updated_at = utcnow()
                await session.commit()
        metrics.incr("trades_analyzed")
