"""Curated RAG retrieval over Mnemosyne's trade memories.

Retrieval returns a BALANCED set of similar wins and losses plus aggregate
stats. Feeding only recent losses to Ares produces a risk-averse spiral;
only wins produces overconfidence — balance is a design requirement, not
an optimization.

Embeddings use Gemini's embedding endpoint (cloud-only constraint).
"""

from __future__ import annotations

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aethel.config import get_settings
from aethel.core.schemas import utcnow
from aethel.db.models import EMBEDDING_DIM, TradeMemory


async def embed_text(text: str) -> list[float]:
    s = get_settings()
    async with httpx.AsyncClient(timeout=15) as http:
        r = await http.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "text-embedding-004:embedContent",
            headers={"x-goog-api-key": s.gemini_api_key},
            json={"content": {"parts": [{"text": text}]},
                  "outputDimensionality": EMBEDDING_DIM},
        )
        r.raise_for_status()
    return r.json()["embedding"]["values"]


class MemoryStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def store(self, decision_id: str, symbol: str, won: bool, profit: float,
                    lesson: str, context_tags: list[str]) -> None:
        embedding = await embed_text(f"{symbol} {' '.join(context_tags)} {lesson}")
        self.session.add(TradeMemory(
            decision_id=decision_id, symbol=symbol, won=won, profit=profit,
            lesson=lesson, context_tags={"tags": context_tags},
            embedding=embedding, created_at=utcnow(),
        ))
        await self.session.commit()

    async def recall(self, symbol: str, context: str, k_per_side: int = 3) -> list[str]:
        """Top-k similar wins AND top-k similar losses, plus aggregate stats."""
        try:
            query_vec = await embed_text(f"{symbol} {context}")
        except httpx.HTTPError:
            return []  # memory is advisory — its failure never blocks a trade

        notes: list[str] = []
        for won in (True, False):
            stmt = (
                select(TradeMemory)
                .where(TradeMemory.symbol == symbol, TradeMemory.won == won)
                .order_by(TradeMemory.embedding.cosine_distance(query_vec))
                .limit(k_per_side)
            )
            for m in (await self.session.execute(stmt)).scalars():
                outcome = "WIN" if won else "LOSS"
                notes.append(f"[{outcome} {m.profit:+.2f}] {m.lesson}")

        stats = await self.session.execute(
            select(func.count(TradeMemory.id),
                   func.avg(func.cast(TradeMemory.won, type_=None)))
            .where(TradeMemory.symbol == symbol)
        )
        row = stats.one_or_none()
        if row and row[0]:
            notes.append(f"[STATS] {row[0]} past {symbol} trades on record")
        return notes
