import pytest
from pydantic import ValidationError

from aethel.core.schemas import Action, AresProposal, AthenaDecision, Verdict


def make_proposal(**overrides) -> AresProposal:
    base = dict(symbol="EURUSD", action=Action.BUY, entry=1.0850,
                stop_loss=1.0830, take_profit=1.0890, risk_pct=0.5,
                rationale="test")
    base.update(overrides)
    return AresProposal(**base)


def test_buy_geometry_enforced():
    with pytest.raises(ValidationError):
        make_proposal(stop_loss=1.0860)  # SL above entry on a BUY


def test_sell_geometry_enforced():
    with pytest.raises(ValidationError):
        make_proposal(action=Action.SELL, stop_loss=1.0830, take_profit=1.0890)


def test_risk_pct_capped_at_schema_level():
    with pytest.raises(ValidationError):
        make_proposal(risk_pct=5.0)


def test_risk_reward_computed():
    assert make_proposal().risk_reward == 2.0


def test_approve_requires_proposal():
    with pytest.raises(ValidationError):
        AthenaDecision(verdict=Verdict.APPROVE, proposal=None,
                       athena_rationale="x", venus_confidence=0.7)


def test_veto_requires_reason():
    with pytest.raises(ValidationError):
        AthenaDecision(verdict=Verdict.VETO, veto_reason=None,
                       athena_rationale="x", venus_confidence=0.7)


def test_decision_without_expiry_is_expired():
    # fail closed: no expires_at means not executable
    d = AthenaDecision(verdict=Verdict.VETO, veto_reason="r",
                       athena_rationale="x", venus_confidence=0.7)
    assert d.is_expired()
