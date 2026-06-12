from datetime import timedelta

import pytest

from aethel.core.schemas import (
    AccountState,
    Action,
    AresProposal,
    AthenaDecision,
    Position,
    RiskRejection,
    Tick,
    ValidatedOrder,
    Verdict,
    utcnow,
)
from aethel.risk_gate.gate import RiskGate


class NoNews:
    async def in_blackout(self, symbol, minutes):
        return None


class BlockingNews:
    async def in_blackout(self, symbol, minutes):
        from aethel.news.calendar import CalendarEvent

        return CalendarEvent(time=utcnow(), currency="USD", title="NFP", impact="high")


def make_decision(**proposal_overrides) -> AthenaDecision:
    base = dict(symbol="EURUSD", action=Action.BUY, entry=1.0850,
                stop_loss=1.0830, take_profit=1.0890, risk_pct=0.5,
                rationale="test")
    base.update(proposal_overrides)
    return AthenaDecision(
        verdict=Verdict.APPROVE, proposal=AresProposal(**base),
        athena_rationale="approved", venus_confidence=0.8,
        expires_at=utcnow() + timedelta(seconds=60),
    )


ACCOUNT = AccountState(balance=10_000, equity=10_000, margin_free=9_000, open_positions=0)
TICK = Tick(time=utcnow(), bid=1.08495, ask=1.08505)


async def run_gate(decision, *, news=None, positions=None, daily_pnl=0.0,
                   trades_today=0, tripped=False, tick=TICK):
    gate = RiskGate(news or NoNews())
    return await gate.validate(decision, ACCOUNT, positions or [], tick,
                               daily_pnl, trades_today, tripped)


async def test_valid_order_passes():
    result = await run_gate(make_decision())
    assert isinstance(result, ValidatedOrder)
    assert result.lots == 0.25  # $50 risk / (20 pips * $10)


async def test_kill_switch_blocks():
    result = await run_gate(make_decision(), tripped=True)
    assert isinstance(result, RiskRejection) and result.rule == "kill_switch"


async def test_daily_loss_blocks():
    result = await run_gate(make_decision(), daily_pnl=-3.5)
    assert isinstance(result, RiskRejection) and result.rule == "daily_loss"


async def test_expired_decision_blocks():
    d = make_decision()
    d.expires_at = utcnow() - timedelta(seconds=1)
    result = await run_gate(d)
    assert isinstance(result, RiskRejection) and result.rule == "ttl"


async def test_price_drift_blocks():
    drifted = Tick(time=utcnow(), bid=1.0880, ask=1.0881)
    result = await run_gate(make_decision(), tick=drifted)
    assert isinstance(result, RiskRejection) and result.rule == "price_drift"


async def test_wide_spread_blocks():
    wide = Tick(time=utcnow(), bid=1.0848, ask=1.0853)  # 5 pips > 3 max
    # entry near the ask so the drift check passes and spread is what rejects
    result = await run_gate(make_decision(entry=1.0852), tick=wide)
    assert isinstance(result, RiskRejection) and result.rule == "spread"


async def test_news_blackout_blocks():
    result = await run_gate(make_decision(), news=BlockingNews())
    assert isinstance(result, RiskRejection) and result.rule == "news_blackout"


async def test_max_positions_blocks():
    positions = [
        Position(ticket=i, symbol="GBPUSD", action=Action.BUY, lots=0.1,
                 entry=1.27, sl=1.26, tp=1.29, profit=0, magic=770001)
        for i in range(3)
    ]
    result = await run_gate(make_decision(), positions=positions)
    assert isinstance(result, RiskRejection) and result.rule == "max_positions"


async def test_usd_exposure_blocks():
    # Two existing BUYs on USD-quote pairs = -2.0 lots USD; one more BUY breaches
    positions = [
        Position(ticket=1, symbol="GBPUSD", action=Action.BUY, lots=1.0,
                 entry=1.27, sl=1.26, tp=1.29, profit=0, magic=770001),
        Position(ticket=2, symbol="XAUUSD", action=Action.BUY, lots=1.0,
                 entry=2300, sl=2280, tp=2340, profit=0, magic=770003),
    ]
    result = await run_gate(make_decision(), positions=positions)
    assert isinstance(result, RiskRejection) and result.rule == "usd_exposure"


async def test_trade_count_blocks():
    result = await run_gate(make_decision(), trades_today=10)
    assert isinstance(result, RiskRejection) and result.rule == "trade_count"
