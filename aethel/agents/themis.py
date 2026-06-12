"""Themis — weekly performance auditor (Gemini 3.1 Pro). One call per week.

Receives the aggregated stats from observability/audit.py and grades the
week: how Ares and Athena performed, whether Venus shows drift, and what
parameter changes are worth a human's consideration. Suggestions are NEVER
applied automatically — the report goes to Telegram and a human decides.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from aethel.agents.llm import LLMClient

SYSTEM = """You are Themis, weekly performance auditor of the Aethel trading
system. You receive one week of aggregated statistics: trade outcomes,
per-symbol PnL, decision-state counts (vetoes, risk rejections), gate blocks
and agent failure counts.

Audit the week:
- ares_assessment: is the strategist proposing quality setups? (look at
  win rate, risk rejections caused by bad geometry)
- athena_assessment: is the reviewer's veto rate adding value or just
  blocking? (vetoes vs outcomes of approved trades)
- venus_drift_concerns: signs the ML signal is degrading (falling win rate,
  rising gate blocks at constant thresholds). Say "none observed" if clean.
- parameter_suggestions: concrete, specific changes worth a HUMAN review,
  e.g. "raise venus_confidence_threshold 0.65 -> 0.70". Empty list if the
  current settings are performing. You cannot change anything yourself.

Respond with ONLY a JSON object:
{"grade": "A"|"B"|"C"|"D"|"F", "ares_assessment": str,
 "athena_assessment": str, "venus_drift_concerns": str,
 "parameter_suggestions": [str], "summary": str}
"""


class WeeklyAudit(BaseModel):
    grade: str = Field(pattern="^[ABCDF]$")
    ares_assessment: str = Field(max_length=1000)
    athena_assessment: str = Field(max_length=1000)
    venus_drift_concerns: str = Field(max_length=1000)
    parameter_suggestions: list[str]
    summary: str = Field(max_length=1000)


class Themis:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def audit(self, weekly_stats: dict) -> WeeklyAudit:
        return await self.llm.structured(
            "themis", SYSTEM, json.dumps(weekly_stats, default=str), WeeklyAudit
        )

    @staticmethod
    def format_report(audit: WeeklyAudit) -> str:
        lines = [
            f"⚖ Themis weekly audit — grade {audit.grade}",
            audit.summary,
            f"Ares: {audit.ares_assessment}",
            f"Athena: {audit.athena_assessment}",
            f"Venus drift: {audit.venus_drift_concerns}",
        ]
        if audit.parameter_suggestions:
            lines.append("Suggested for human review:")
            lines += [f"  • {s}" for s in audit.parameter_suggestions]
        return "\n".join(lines)
