"""Typed contracts between every stage of the pipeline.

Design invariants enforced here:
- Lot size is NEVER produced by an LLM. Ares proposes risk_pct; the Risk Gate
  computes lots deterministically from contract specs.
- Every approved decision carries decision_id, expires_at and max_deviation so
  Hermes can enforce TTL, idempotency and price-drift tolerance.
- SL is mandatory on every proposal — the Risk Gate rejects orders without one.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Action(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class Timeframe(StrEnum):
    M1 = "M1"
    M15 = "M15"
    H1 = "H1"


class Candle(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int


class Tick(BaseModel):
    time: datetime
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


class AccountState(BaseModel):
    balance: float
    equity: float
    margin_free: float
    open_positions: int


class Position(BaseModel):
    ticket: int
    symbol: str
    action: Action
    lots: float
    entry: float
    sl: float
    tp: float
    profit: float
    magic: int


class VenusSignal(BaseModel):
    """Output of the Venus ML pipeline. confidence is a calibrated
    P(TP hit before SL) from triple-barrier meta-labeling."""

    symbol: str
    direction: Action
    confidence: float = Field(ge=0.0, le=1.0)
    features: dict[str, float] = Field(default_factory=dict)
    model_version: str
    generated_at: datetime = Field(default_factory=utcnow)


class AresProposal(BaseModel):
    """Trade setup proposed by Ares (DeepSeek). Note: risk_pct, not lots."""

    symbol: str
    action: Action
    entry: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    risk_pct: float = Field(gt=0, le=2.0, description="% of equity risked if SL is hit")
    rationale: str = Field(max_length=2000)

    @model_validator(mode="after")
    def validate_geometry(self) -> AresProposal:
        if self.action == Action.BUY:
            if not (self.stop_loss < self.entry < self.take_profit):
                raise ValueError("BUY requires SL < entry < TP")
        else:
            if not (self.take_profit < self.entry < self.stop_loss):
                raise ValueError("SELL requires TP < entry < SL")
        return self

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop_loss)
        reward = abs(self.take_profit - self.entry)
        return round(reward / risk, 2) if risk else 0.0


class Verdict(StrEnum):
    APPROVE = "APPROVE"
    VETO = "VETO"


class AthenaDecision(BaseModel):
    """Final agent-layer decision from Athena (Claude). Strict schema —
    parsed with Pydantic, one bounded retry on invalid output, then fail closed."""

    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    verdict: Verdict
    veto_reason: str | None = None
    proposal: AresProposal | None = None
    athena_rationale: str = Field(max_length=2000)
    venus_confidence: float
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> AthenaDecision:
        if self.verdict == Verdict.APPROVE and self.proposal is None:
            raise ValueError("APPROVE requires a proposal")
        if self.verdict == Verdict.VETO and not self.veto_reason:
            raise ValueError("VETO requires a veto_reason")
        return self

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return True  # fail closed: no expiry set means not executable
        return (now or utcnow()) >= self.expires_at


class ValidatedOrder(BaseModel):
    """What the Risk Gate emits — the only thing Hermes will execute.
    lots was computed by deterministic Python, never by an LLM."""

    decision_id: str
    symbol: str
    action: Action
    entry: float
    stop_loss: float
    take_profit: float
    lots: float
    magic: int
    max_deviation_pips: float
    order_expiry_seconds: int


class RiskRejection(BaseModel):
    decision_id: str
    rule: str
    detail: str


class ExecutionResult(BaseModel):
    decision_id: str
    success: bool
    shadow: bool = False
    ticket: int | None = None
    retcode: int | None = None
    detail: str = ""
