"""Mnemosyne — post-trade analyst (Gemini). Runs asynchronously after a
trade closes; never blocks the signal pipeline.

Produces a structured lesson from the closed trade plus the full agent
transcript, which is embedded and stored in pgvector. Retrieval is curated:
the memory store returns a BALANCED set of similar wins and losses plus
aggregate stats, to avoid the recency-loss overcorrection spiral.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from aethel.agents.llm import LLMClient

SYSTEM = """You are Mnemosyne, post-trade analyst of the Aethel trading system.

Analyze one closed trade with its full decision context (Venus signal, Ares
proposal, Athena review, execution and outcome). Extract a transferable
lesson — what about the CONTEXT predicted this outcome, not a narrative of
the price action.

Respond with ONLY a JSON object:
{"outcome_summary": str,
 "context_tags": [str],          // regime/session/volatility/news tags
 "lesson": str,                  // one transferable, falsifiable lesson
 "ares_quality": int,            // 1-5, proposal quality in hindsight
 "athena_quality": int}          // 1-5, review quality in hindsight
"""


class TradeLesson(BaseModel):
    outcome_summary: str = Field(max_length=1000)
    context_tags: list[str]
    lesson: str = Field(max_length=1000)
    ares_quality: int = Field(ge=1, le=5)
    athena_quality: int = Field(ge=1, le=5)


class Mnemosyne:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def analyze(self, trade_context: dict) -> TradeLesson:
        return await self.llm.structured(
            "gemini", SYSTEM, json.dumps(trade_context, default=str), TradeLesson
        )
