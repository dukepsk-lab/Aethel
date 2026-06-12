"""Central configuration. All risk limits live here and in the database —
never in prompts. LLM output cannot change anything in this module."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

# Net USD exposure sign for a BUY of one lot of each symbol.
# USD is the quote currency for EURUSD/GBPUSD/XAUUSD -> buying them sells USD.
# USD is the base currency for USDJPY -> buying it buys USD.
USD_EXPOSURE_SIGN = {"EURUSD": -1, "GBPUSD": -1, "USDJPY": +1, "XAUUSD": -1}


class SymbolSpec(BaseSettings):
    """Contract specification used for deterministic lot sizing."""

    symbol: str
    pip_size: float          # price increment of one pip
    pip_value_per_lot: float  # account-currency (USD) value of 1 pip for 1.0 lot
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_spread_pips: float = 3.0  # risk gate rejects above this


SYMBOL_SPECS: dict[str, SymbolSpec] = {
    "EURUSD": SymbolSpec(symbol="EURUSD", pip_size=0.0001, pip_value_per_lot=10.0),
    "GBPUSD": SymbolSpec(symbol="GBPUSD", pip_size=0.0001, pip_value_per_lot=10.0),
    "USDJPY": SymbolSpec(symbol="USDJPY", pip_size=0.01, pip_value_per_lot=6.7),
    "XAUUSD": SymbolSpec(symbol="XAUUSD", pip_size=0.1, pip_value_per_lot=10.0,
                         max_spread_pips=5.0),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHEL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://aethel:aethel@localhost:5432/aethel"

    # MT5 connectivity — the MetaTrader5 package is Windows-only, so the
    # default topology is a Linux brain talking to a Windows gateway.
    mt5_mode: Literal["gateway", "direct"] = "gateway"
    mt5_gateway_url: str = "http://localhost:8001"
    mt5_gateway_token: str = ""

    # LLM APIs (strictly cloud)
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    gemini_api_key: str = ""
    # Athena (Chief Risk Officer) — Gemini 3.1 Pro Preview, updated 2026-06-12
    athena_model: str = "gemini-3.1-pro-preview"
    # Mnemosyne (post-trade analyst) — Gemini 3.5 Flash
    mnemosyne_model: str = "gemini-3.5-flash"
    # Apollo (news sentiment, fail-soft advisory context) — Gemini 3.5 Flash
    apollo_model: str = "gemini-3.5-flash"
    # Themis (weekly auditor, one call/week) — Gemini 3.1 Pro Preview
    themis_model: str = "gemini-3.1-pro-preview"
    llm_timeout_seconds: float = 30.0

    # Execution
    shadow_mode: bool = True
    magic_base: int = 770000  # MT5 magic number base; +index per symbol
    order_expiry_seconds: int = 300       # pending limit order lifetime
    decision_ttl_seconds: int = 60        # approved decision dies after this
    max_deviation_pips: float = 2.0       # price-drift tolerance at execution

    # Signal gate — keeps agent call volume and noise bounded
    venus_confidence_threshold: float = 0.65
    signal_cooldown_seconds: int = 900    # per-symbol cooldown between consultations
    max_agent_calls_per_hour: int = 12

    # Risk gate (hard limits — AI cannot override)
    max_daily_loss_pct: float = 3.0
    max_risk_per_trade_pct: float = 1.0
    max_lots_per_trade: float = 1.0
    max_concurrent_positions: int = 3
    max_trades_per_day: int = 10
    max_net_usd_exposure_lots: float = 2.0
    daily_reset_hour_utc: int = 21        # ~5pm New York (broker rollover)
    news_blackout_minutes: int = 30       # no entries within this window of high-impact events

    # Trade management (rule-based, pure Python)
    breakeven_trigger_rr: float = 1.0     # move SL to BE at 1R profit
    trailing_start_rr: float = 1.5

    # Alerts
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
