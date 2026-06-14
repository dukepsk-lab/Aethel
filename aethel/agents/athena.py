"""Athena — Chief Risk Officer (Gemini 3.1 Pro Preview). Reviews Ares's proposal against
upcoming news, current drawdown and open exposure. Absolute veto power:
a VETO terminates the trade path with no re-negotiation.

Athena's approval is necessary but not sufficient — the hardcoded Risk Gate
still runs after her and she cannot override it.
"""

from __future__ import annotations

import json
from datetime import timedelta

from pydantic import BaseModel, Field

from aethel.agents.llm import LLMClient
from aethel.config import get_settings
from aethel.core.schemas import (
    AccountState,
    AresProposal,
    AthenaDecision,
    Position,
    Verdict,
    VenusSignal,
    utcnow,
)

SYSTEM = """You are Athena, Chief Risk Officer of the Aethel trading system.
You have ABSOLUTE veto power over every trade proposal.

Review the proposal against:
1. Upcoming high-impact economic events (provided) — veto entries shortly
   before major releases affecting the symbol's currencies.
2. Current daily drawdown and open positions — veto when the account is
   already stressed or the proposal concentrates directional USD exposure.
3. Proposal quality — veto if the stop placement, R:R or rationale is weak,
   or if it contradicts the Venus signal it claims to be based on.

You may also receive an H1 market-regime label and a news-sentiment brief
(from Apollo). Both are advisory context; either may be null/unavailable —
their absence is NOT a reason to veto.

model_consensus shows whether both Venus and Helios ML models agree. Disagreement is a signal of lower conviction and should increase scrutiny.

You may APPROVE the proposal as-is or VETO it. You may NOT modify its levels.
Approve only when you would defend this trade in a post-mortem.

Respond with ONLY a JSON object:
{"verdict": "APPROVE"|"VETO", "veto_reason": str|null, "athena_rationale": str}
"""


class _AthenaRaw(BaseModel):
    verdict: Verdict
    veto_reason: str | None = None
    athena_rationale: str = Field(max_length=2000)


class Athena:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.s = get_settings()

    async def review(
        self,
        signal: VenusSignal,
        proposal: AresProposal,
        account: AccountState,
        positions: list[Position],
        daily_pnl_pct: float,
        upcoming_events: list[dict],
        regime: dict | None = None,
        news_sentiment: dict | None = None,
    ) -> AthenaDecision:
        user = json.dumps({
            "venus_signal": signal.model_dump(mode="json"),
            "ares_proposal": proposal.model_dump(mode="json"),
            "proposal_risk_reward": proposal.risk_reward,
            "account": account.model_dump(),
            "open_positions": [p.model_dump(mode="json") for p in positions],
            "daily_pnl_pct": daily_pnl_pct,
            "upcoming_high_impact_events": upcoming_events,
            "h1_market_regime": regime,
            "news_sentiment": news_sentiment,  # Apollo's brief; null = unavailable
            "model_consensus": {
                "venus_confidence": signal.confidence,
                "helios_confidence": signal.helios_confidence,
                "consensus": signal.helios_agreed,
                "note": (
                    "Both Venus and Helios models agree on this signal." if signal.helios_agreed
                    else "Venus signals but Helios does NOT agree — lower conviction setup."
                    if signal.helios_agreed is False
                    else "Helios model unavailable — single-model signal."
                ),
            },
        })
        raw = await self.llm.structured("athena", SYSTEM, user, _AthenaRaw)
        return AthenaDecision(
            verdict=raw.verdict,
            veto_reason=raw.veto_reason,
            proposal=proposal if raw.verdict == Verdict.APPROVE else None,
            athena_rationale=raw.athena_rationale,
            venus_confidence=signal.confidence,
            expires_at=utcnow() + timedelta(seconds=self.s.decision_ttl_seconds),
        )
