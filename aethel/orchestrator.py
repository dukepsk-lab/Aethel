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
from aethel.venus.retrain import run_retrain_cycle

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
        # Helios has its own per-symbol cooldown — independent of Venus gate
        self._helios_last_consult: dict[str, float] = {}

    async def run_forever(self) -> None:
        log.info("aethel_started", shadow_mode=self.s.shadow_mode, symbols=SYMBOLS)
        await send_alert(f"✅ Aethel started — {'SHADOW' if self.s.shadow_mode else 'LIVE'} mode | symbols: {', '.join(SYMBOLS)}")
        asyncio.create_task(self._weekly_audit_loop())
        asyncio.create_task(self._scheduled_retrain_loop())
        asyncio.create_task(self._detect_closed_trades_loop())
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
                for symbol in self.helios:
                    try:
                        await self.process_helios_symbol(symbol)
                    except Exception as e:
                        log.error("helios_pipeline_error", symbol=symbol, error=str(e))
                        metrics.incr("helios_pipeline_errors")
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
        signal_log.record(signal, passed=consult, reason=reason,
                          helios_confidence=helios_conf,
                          helios_agreed=helios_agreed)
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

            if tier == "normal" and self.s.bypass_agents:
                # bypass_agents mode: skip Ares/Athena, use ATR-based SL/TP directly
                log.info("bypass_agents_path", symbol=symbol, confidence=signal.confidence)
                metrics.incr("bypass_agents_trades")
                try:
                    tick = await self.mt5.get_tick(symbol)
                    entry = tick.ask if signal.direction == Action.BUY else tick.bid
                    h1_candles = candles[Timeframe.H1]
                    highs = [c.high for c in h1_candles[-20:]]
                    lows = [c.low for c in h1_candles[-20:]]
                    atr = float(np.mean([h - l for h, l in zip(highs, lows)]))
                    if signal.direction == Action.BUY:
                        stop_loss = entry - 1.0 * atr
                        take_profit = entry + 2.0 * atr
                    else:
                        stop_loss = entry + 1.0 * atr
                        take_profit = entry - 2.0 * atr
                    proposal = AresProposal(
                        symbol=symbol, action=signal.direction,
                        entry=round(entry, 5),
                        stop_loss=round(stop_loss, 5),
                        take_profit=round(take_profit, 5),
                        risk_pct=self.s.base_risk_pct,
                        rationale=f"bypass_agents (conf={signal.confidence:.2f})",
                    )
                    decision = AthenaDecision(
                        verdict=Verdict.APPROVE,
                        proposal=proposal,
                        athena_rationale="bypass_agents — Ares/Athena skipped by config",
                        venus_confidence=signal.confidence,
                        expires_at=utcnow() + timedelta(seconds=self.s.decision_ttl_seconds),
                    )
                    record = Decision(
                        decision_id=decision.decision_id, symbol=symbol,
                        state=TradeState.APPROVED, venus_signal=signal.model_dump(mode="json"),
                        ares_proposal=proposal.model_dump(mode="json"),
                        athena_decision={"verdict": "BYPASS", "veto_reason": None},
                        created_at=utcnow(), updated_at=utcnow(),
                    )
                    session.add(record)
                    tier = "bypass_done"
                except Exception as e:
                    log.warning("bypass_agents_failed", symbol=symbol, error=str(e))
                    return

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

            # 5. Risk Gate — fresh tick, hard limits (runs for all tiers)
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

    async def process_helios_symbol(self, symbol: str) -> None:
        """Independent Helios pipeline — same flow as Venus but uses its own
        cooldown so a Venus trade does not suppress a Helios entry."""
        import time as _time

        # Helios cooldown (same duration as Venus signal_cooldown_seconds)
        now_ts = _time.monotonic()
        last = self._helios_last_consult.get(symbol, 0.0)
        if now_ts - last < self.s.signal_cooldown_seconds:
            return

        with metrics.time_stage("helios"):
            candles = {
                tf: await self.mt5.get_candles(symbol, tf, count)
                for tf, count in CANDLE_COUNTS.items()
            }
            h_signal = self.helios[symbol].predict(candles)

        if h_signal.confidence < self.s.helios_confidence_threshold:
            signal_log.record(h_signal, passed=False,
                              reason=f"helios below threshold ({h_signal.confidence:.3f} < {self.s.helios_confidence_threshold})",
                              helios_confidence=h_signal.confidence, helios_agreed=None)
            return

        self._helios_last_consult[symbol] = now_ts
        metrics.incr("helios_signals")
        signal_log.record(h_signal, passed=True, reason="helios gate passed",
                          helios_confidence=h_signal.confidence, helios_agreed=True)
        await send_alert(
            f"🔮 Helios | {symbol} | {h_signal.direction.value} | conf={h_signal.confidence:.2f}"
        )

        # Helios uses bypass_agents path — selective labels already encode high quality
        async with get_sessionmaker()() as session:
            risk_store = RiskStateStore(session)
            account = await self.mt5.get_account()
            positions = await self.mt5.get_positions()
            daily_pnl = await risk_store.daily_pnl_pct(account.equity)

            if daily_pnl <= -self.s.max_daily_loss_pct and not await risk_store.is_tripped():
                await risk_store.trip_kill_switch(f"daily loss {daily_pnl:.2f}%")
                await send_alert(f"🛑 KILL SWITCH tripped: daily loss {daily_pnl:.2f}%")
            if await risk_store.is_tripped():
                return

            try:
                tick = await self.mt5.get_tick(symbol)
                entry = tick.ask if h_signal.direction == Action.BUY else tick.bid
                h1_candles = candles[Timeframe.H1]
                highs = [c.high for c in h1_candles[-20:]]
                lows = [c.low for c in h1_candles[-20:]]
                atr = float(np.mean([h - l for h, l in zip(highs, lows)]))
                if h_signal.direction == Action.BUY:
                    stop_loss = entry - 1.0 * atr
                    take_profit = entry + 2.0 * atr
                else:
                    stop_loss = entry + 1.0 * atr
                    take_profit = entry - 2.0 * atr
                proposal = AresProposal(
                    symbol=symbol, action=h_signal.direction,
                    entry=round(entry, 5),
                    stop_loss=round(stop_loss, 5),
                    take_profit=round(take_profit, 5),
                    risk_pct=self.s.base_risk_pct,
                    rationale=f"helios selective (conf={h_signal.confidence:.2f})",
                )
                decision = AthenaDecision(
                    verdict=Verdict.APPROVE,
                    proposal=proposal,
                    athena_rationale="helios selective-label fast path",
                    venus_confidence=h_signal.confidence,
                    expires_at=utcnow() + timedelta(seconds=self.s.decision_ttl_seconds),
                )
            except Exception as e:
                log.warning("helios_proposal_failed", symbol=symbol, error=str(e))
                return

            record = Decision(
                decision_id=decision.decision_id, symbol=symbol,
                state=TradeState.APPROVED,
                venus_signal=h_signal.model_dump(mode="json"),
                ares_proposal=proposal.model_dump(mode="json"),
                athena_decision={"verdict": "HELIOS_FAST", "veto_reason": None},
                created_at=utcnow(), updated_at=utcnow(),
            )
            session.add(record)

            with metrics.time_stage("risk_gate"):
                tick = await self.mt5.get_tick(symbol)
                outcome = await self.risk_gate.validate(
                    decision, account, positions, tick, daily_pnl,
                    await risk_store.trades_today(), await risk_store.is_tripped())

            record.risk_outcome = outcome.model_dump(mode="json")
            if not outcome.approved:
                record.state = TradeState.RISK_REJECTED
                metrics.incr(f"risk_reject_{outcome.reason}")
                await session.commit()
                return

            record.state = TradeState.APPROVED
            with metrics.time_stage("hermes"):
                result = await self.hermes.execute(outcome, self.s.shadow_mode)

            record.execution = result.model_dump(mode="json")
            record.ticket = result.ticket
            if result.success:
                record.state = TradeState.EXECUTED
                metrics.incr("orders_executed")
                await risk_store.record_trade(account.equity)
                await send_alert(
                    f"{'👁' if self.s.shadow_mode else '✅'} Helios {'SHADOW' if self.s.shadow_mode else 'EXECUTED'} "
                    f"{symbol} {h_signal.direction.value} | conf={h_signal.confidence:.2f}"
                )
            else:
                metrics.incr("orders_failed")
                await send_alert(f"⚠️ Helios order failed {symbol}: {result.detail}")
            await session.commit()

    async def _detect_closed_trades_loop(self) -> None:
        """Poll MT5 deal history every 60s and persist newly-closed trades.

        Trades closed by SL/TP/broker are never seen by the Hermes executor,
        so without this loop they vanish from the DB. We match on ticket
        (position_id) against EXECUTED decisions to link the decision_id.
        """
        from aethel.db.models import ClosedTrade
        from datetime import timedelta as _td

        # look back 24h on first run to catch trades closed while bot was down
        look_back = utcnow() - _td(hours=24)

        while True:
            await asyncio.sleep(60)
            try:
                deals = await self.mt5.get_closed_deals(look_back)
                if not deals:
                    look_back = utcnow() - _td(minutes=10)
                    continue

                async with get_sessionmaker()() as session:
                    # load known tickets to avoid re-inserting
                    existing = set(
                        r[0] for r in (await session.execute(
                            select(ClosedTrade.ticket)
                        )).all()
                    )
                    # map ticket → decision_id from EXECUTED decisions
                    from aethel.db.models import Decision as _Dec
                    dec_rows = (await session.execute(
                        select(_Dec.ticket, _Dec.decision_id, _Dec.venus_signal)
                        .where(_Dec.state.in_(["EXECUTED", "MANAGED"]),
                               _Dec.ticket.isnot(None))
                    )).all()
                    ticket_to_dec = {r.ticket: (r.decision_id, r.venus_signal)
                                     for r in dec_rows}

                    new_trades = []
                    for d in deals:
                        t = d["ticket"]
                        if t in existing:
                            continue
                        dec_id, vsig = ticket_to_dec.get(t, (f"unknown-{t}", {}))
                        action = "BUY" if (d.get("type") == 1) else "SELL"
                        ct = ClosedTrade(
                            ticket=t,
                            decision_id=dec_id,
                            symbol=d["symbol"],
                            action=action,
                            lots=d.get("volume", 0.0),
                            entry=d.get("price_in") or 0.0,
                            exit_price=d.get("price_out") or 0.0,
                            profit=d.get("profit", 0.0),
                            opened_at=d.get("time_open") or utcnow(),
                            closed_at=d.get("time_close") or utcnow(),
                        )
                        session.add(ct)
                        new_trades.append(ct)

                    if new_trades:
                        await session.commit()
                        for ct in new_trades:
                            pnl = ct.profit
                            log.info("closed_trade_recorded", symbol=ct.symbol,
                                     ticket=ct.ticket, profit=pnl)
                            metrics.incr("closed_trades_detected")
                        await send_alert(
                            f"📋 {len(new_trades)} trade(s) closed | "
                            + " | ".join(
                                f"{t.symbol} {t.action} {t.profit:+.2f}" for t in new_trades
                            )
                        )

                look_back = utcnow() - _td(minutes=10)

            except Exception as e:
                log.error("detect_closed_trades_failed", error=str(e))

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

    async def _scheduled_retrain_loop(self) -> None:
        """Benchmark + promote Venus every ``retrain_interval_days`` days.

        Runs in the background, fail-soft — a failed retrain is logged and
        retried on the next interval. On promotion the in-process
        VenusInference is hot-reloaded so the bot picks up the new weights
        without a restart.
        """
        interval = timedelta(days=self.s.retrain_interval_days)
        await asyncio.sleep(interval.total_seconds())   # first run after one interval
        while True:
            promoted: list[str] = []
            skipped: list[str] = []
            errors:  list[str] = []
            data_dir = Path(self.s.retrain_data_dir)
            artifacts_dir = Path(self.s.retrain_artifacts_dir)
            for symbol in list(self.venus):
                data_path = data_dir / f"{symbol}_m5.parquet"
                if not data_path.exists():
                    log.warning("retrain_data_missing", symbol=symbol, path=str(data_path))
                    errors.append(symbol)
                    continue
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda sym=symbol, dp=data_path: run_retrain_cycle(
                            sym, dp, artifacts_dir / sym,
                            data_days=self.s.retrain_data_days),
                    )
                    if result["action"] == "promoted":
                        # hot-reload — replace the in-process VenusInference
                        try:
                            self.venus[symbol] = VenusInference(artifacts_dir, symbol)
                            log.info("venus_hot_reloaded", symbol=symbol)
                        except Exception as e:
                            log.error("venus_hot_reload_failed", symbol=symbol, error=str(e))
                        promoted.append(symbol)
                    else:
                        skipped.append(symbol)
                    metrics.incr("retrain_cycles")
                except Exception as e:
                    log.error("retrain_failed", symbol=symbol, error=str(e))
                    errors.append(symbol)
                    metrics.incr("retrain_errors")

            parts = []
            if promoted:
                parts.append(f"✅ promoted: {', '.join(promoted)}")
            if skipped:
                parts.append(f"↷ no improvement: {', '.join(skipped)}")
            if errors:
                parts.append(f"⚠️ errors: {', '.join(errors)}")
            msg = f"🔄 Retrain cycle complete | {' | '.join(parts)}"
            await send_alert(msg)
            log.info("retrain_cycle_complete", promoted=promoted, skipped=skipped, errors=errors)
            await asyncio.sleep(interval.total_seconds())

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
