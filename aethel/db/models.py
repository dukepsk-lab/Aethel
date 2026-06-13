"""Database models. PostgreSQL is the single source of truth for trade
lifecycle state, daily risk counters and the kill switch — process memory is
never trusted across restarts. pgvector stores Mnemosyne's trade memories
in the same database (transactional with the trade log)."""

from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    pass


class Decision(Base):
    """Full lifecycle record for every Venus signal that reached the agents."""

    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    state: Mapped[str] = mapped_column(String(20), index=True)
    venus_signal: Mapped[dict] = mapped_column(JSON)
    ares_proposal: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    athena_decision: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    risk_outcome: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    execution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ticket: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ClosedTrade(Base):
    __tablename__ = "closed_trades"

    ticket: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[str] = mapped_column(String(36), index=True)
    symbol: Mapped[str] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(4))
    lots: Mapped[float] = mapped_column(Float)
    entry: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float] = mapped_column(Float)
    profit: Mapped[float] = mapped_column(Float)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DailyRiskState(Base):
    """Kill-switch and daily counters — survives restarts by design.
    One row per trading day (boundary = daily_reset_hour_utc)."""

    __tablename__ = "daily_risk_state"

    trading_day: Mapped[date] = mapped_column(Date, primary_key=True)
    start_equity: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    kill_switch_tripped: Mapped[bool] = mapped_column(Boolean, default=False)
    kill_switch_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class TradeMemory(Base):
    """Mnemosyne's episodic memory — embedded lessons for RAG retrieval."""

    __tablename__ = "trade_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[str] = mapped_column(String(36), index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    won: Mapped[bool] = mapped_column(Boolean, index=True)
    profit: Mapped[float] = mapped_column(Float)
    lesson: Mapped[str] = mapped_column(Text)
    context_tags: Mapped[dict] = mapped_column(JSON)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Mnemosyne's hindsight quality scores (1-5) — previously computed and
    # discarded; persisted now so Themis and retrieval can learn from them.
    ares_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)
    athena_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Regime tag at decision time — enables regime-aware memory retrieval.
    regime: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
