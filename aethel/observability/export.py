"""Export Themis audits + Mnemosyne lessons as markdown for the human.

This is the *only* sanctioned bridge to Obsidian / NotebookLM: one-way,
read-only, bot → markdown → human. The bot never reads these tools back into
the pipeline (they have no deterministic, auditable API — that would violate
the fail-closed contract). They are study material for the operator.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from aethel.db.models import TradeMemory

logger = logging.getLogger(__name__)
EXPORT_DIR = Path("exports/obsidian")


async def export_lessons_markdown(session: AsyncSession, limit: int = 200) -> Path:
    """Export the most recent TradeMemory lessons as a single markdown file."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = await session.execute(
        select(TradeMemory).order_by(desc(TradeMemory.created_at)).limit(limit)
    )
    memories = result.scalars().all()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"# Aethel Trade Lessons\n\nExported: {ts}\n\n---\n"]
    for m in memories:
        outcome = "✅ WIN" if m.won else "❌ LOSS"
        day = m.created_at.strftime("%Y-%m-%d") if m.created_at else "?"
        lines.append(f"## {outcome} | {m.symbol} | {day}")
        if m.regime:
            lines.append(f"**Regime:** {m.regime}")
        lines.append(f"**P&L:** {m.profit:+.2f}")
        if m.ares_quality is not None:
            lines.append(f"**Ares quality:** {m.ares_quality}/5  "
                         f"**Athena quality:** {m.athena_quality}/5")
        tags = (m.context_tags or {}).get("tags") if isinstance(m.context_tags, dict) else None
        if tags:
            lines.append(f"**Tags:** {', '.join(tags)}")
        lines.append(f"\n{m.lesson}\n\n---\n")

    out = EXPORT_DIR / f"lessons_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[export] lessons → %s (%d entries)", out, len(memories))
    return out


def export_audit_markdown(audit: dict) -> Path:
    """Export a Themis WeeklyAudit (as a dict) to markdown."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    suggestions = audit.get("parameter_suggestions", [])
    lines = [
        f"# Themis Weekly Audit\n\n**Date:** {ts}  **Grade:** {audit.get('grade', '?')}\n",
        f"## Summary\n{audit.get('summary', '')}\n",
        f"## Ares Assessment\n{audit.get('ares_assessment', '')}\n",
        f"## Athena Assessment\n{audit.get('athena_assessment', '')}\n",
        f"## Venus Drift Concerns\n{audit.get('venus_drift_concerns', '')}\n",
        "## Parameter Suggestions\n" + (
            "\n".join(f"- {s}" for s in suggestions) if suggestions else "_none_"
        ) + "\n",
    ]
    out = EXPORT_DIR / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[export] audit → %s", out)
    return out
