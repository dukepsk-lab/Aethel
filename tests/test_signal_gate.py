from aethel.core.schemas import Action, VenusSignal
from aethel.signal_gate import SignalGate


def make_signal(confidence: float, symbol: str = "EURUSD") -> VenusSignal:
    return VenusSignal(symbol=symbol, direction=Action.BUY,
                       confidence=confidence, model_version="test")


def test_low_confidence_blocked():
    gate = SignalGate()
    ok, reason = gate.should_consult_agents(make_signal(0.3))  # below 0.45 threshold
    assert not ok and "threshold" in reason


def test_high_confidence_passes():
    gate = SignalGate()
    ok, _ = gate.should_consult_agents(make_signal(0.8))
    assert ok


def test_cooldown_blocks_repeat():
    gate = SignalGate()
    gate.record_consultation("EURUSD")
    ok, reason = gate.should_consult_agents(make_signal(0.8))
    assert not ok and "cooldown" in reason


def test_cooldown_is_per_symbol():
    gate = SignalGate()
    gate.record_consultation("EURUSD")
    ok, _ = gate.should_consult_agents(make_signal(0.8, symbol="USDJPY"))
    assert ok


def test_hourly_budget():
    gate = SignalGate()
    for i in range(12):
        gate.record_consultation(f"SYM{i}")
    ok, reason = gate.should_consult_agents(make_signal(0.8))
    assert not ok and "budget" in reason
