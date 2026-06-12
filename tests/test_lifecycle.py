import pytest

from aethel.core.lifecycle import InvalidTransition, TradeState, transition


def test_happy_path():
    state = TradeState.SIGNAL
    for nxt in [TradeState.GATED, TradeState.PROPOSED, TradeState.APPROVED,
                TradeState.RISK_CHECKED, TradeState.EXECUTED, TradeState.MANAGED,
                TradeState.CLOSED, TradeState.ANALYZED]:
        state = transition(state, nxt)
    assert state == TradeState.ANALYZED


def test_veto_is_terminal():
    with pytest.raises(InvalidTransition):
        transition(TradeState.VETOED, TradeState.APPROVED)


def test_cannot_skip_risk_gate():
    with pytest.raises(InvalidTransition):
        transition(TradeState.APPROVED, TradeState.EXECUTED)


def test_cannot_reexecute():
    with pytest.raises(InvalidTransition):
        transition(TradeState.EXECUTED, TradeState.RISK_CHECKED)
