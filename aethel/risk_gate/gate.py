"""The Risk Gate — hardcoded Python, the final checkpoint before execution.

Every rule here is a hard limit from configuration or the database. No LLM
output reaches this module except the (already schema-validated) decision
being checked, and nothing an LLM says can relax a rule. Every check fails
closed.

Checks, in order:
 1. decision approved, not expired (TTL)
 2. kill switch not tripped / daily loss within limit (DB-persisted)
 3. daily trade count within limit
 4. SL present and on the correct side (geometry re-checked here, not trusted)
 5. price drift from proposed entry within max deviation
 6. spread within per-symbol limit
 7. news blackout window clear
 8. concurrent position count within limit
 9. net USD exposure within limit (all four symbols are USD crosses)
10. lot size computed deterministically, within per-trade cap
"""

from __future__ import annotations

from aethel.config import SYMBOL_SPECS, SYMBOLS, USD_EXPOSURE_SIGN, get_settings
from aethel.core.schemas import (
    AccountState,
    Action,
    AthenaDecision,
    Position,
    RiskRejection,
    Tick,
    ValidatedOrder,
    Verdict,
)
from aethel.news.calendar import NewsService
from aethel.risk_gate.sizing import compute_lots


class RiskGate:
    def __init__(self, news: NewsService) -> None:
        self.s = get_settings()
        self.news = news

    def _reject(self, decision_id: str, rule: str, detail: str) -> RiskRejection:
        return RiskRejection(decision_id=decision_id, rule=rule, detail=detail)

    async def validate(
        self,
        decision: AthenaDecision,
        account: AccountState,
        positions: list[Position],
        tick: Tick,
        daily_pnl_pct: float,
        trades_today: int,
        kill_switch_tripped: bool,
    ) -> ValidatedOrder | RiskRejection:
        did = decision.decision_id

        # 1. approval + TTL
        if decision.verdict != Verdict.APPROVE or decision.proposal is None:
            return self._reject(did, "approval", "decision is not an approval")
        if decision.is_expired():
            return self._reject(did, "ttl", "decision expired before execution")
        p = decision.proposal

        # 2. kill switch / daily loss
        if kill_switch_tripped:
            return self._reject(did, "kill_switch", "kill switch is tripped")
        if daily_pnl_pct <= -self.s.max_daily_loss_pct:
            return self._reject(
                did, "daily_loss",
                f"daily PnL {daily_pnl_pct:.2f}% <= -{self.s.max_daily_loss_pct}%")

        # 3. trade count
        if trades_today >= self.s.max_trades_per_day:
            return self._reject(did, "trade_count",
                                f"{trades_today} trades today >= {self.s.max_trades_per_day}")

        # 4. SL geometry — re-checked here, never trusted from upstream
        if p.action == Action.BUY and not (p.stop_loss < p.entry < p.take_profit):
            return self._reject(did, "geometry", "BUY requires SL < entry < TP")
        if p.action == Action.SELL and not (p.take_profit < p.entry < p.stop_loss):
            return self._reject(did, "geometry", "SELL requires TP < entry < SL")

        spec = SYMBOL_SPECS[p.symbol]

        # 5. price drift
        market = tick.ask if p.action == Action.BUY else tick.bid
        drift_pips = abs(market - p.entry) / spec.pip_size
        if drift_pips > self.s.max_deviation_pips:
            return self._reject(did, "price_drift",
                                f"price drifted {drift_pips:.1f} pips from proposed entry")

        # 6. spread
        spread_pips = tick.spread / spec.pip_size
        if spread_pips > spec.max_spread_pips:
            return self._reject(did, "spread",
                                f"spread {spread_pips:.1f} pips > {spec.max_spread_pips}")

        # 7. news blackout
        blocking = await self.news.in_blackout(p.symbol, self.s.news_blackout_minutes)
        if blocking is not None:
            return self._reject(did, "news_blackout",
                                f"high-impact event within window: {blocking.title}")

        # 8. concurrent positions
        if len(positions) >= self.s.max_concurrent_positions:
            return self._reject(did, "max_positions",
                                f"{len(positions)} open >= {self.s.max_concurrent_positions}")

        # 9. net USD exposure
        lots = compute_lots(p.symbol, account.equity, p.risk_pct, p.entry, p.stop_loss)
        if lots <= 0:
            return self._reject(did, "sizing", "computed lot size below minimum")
        net_usd = sum(
            pos.lots * USD_EXPOSURE_SIGN[pos.symbol] * (1 if pos.action == Action.BUY else -1)
            for pos in positions if pos.symbol in SYMBOLS
        )
        new_usd = lots * USD_EXPOSURE_SIGN[p.symbol] * (1 if p.action == Action.BUY else -1)
        if abs(net_usd + new_usd) > self.s.max_net_usd_exposure_lots:
            return self._reject(
                did, "usd_exposure",
                f"net USD exposure would be {net_usd + new_usd:+.2f} lots "
                f"(limit {self.s.max_net_usd_exposure_lots})")

        # 10. lot cap
        if lots > self.s.max_lots_per_trade:
            lots = self.s.max_lots_per_trade  # cap, don't reject — risk only shrinks

        return ValidatedOrder(
            decision_id=did,
            symbol=p.symbol,
            action=p.action,
            entry=p.entry,
            stop_loss=p.stop_loss,
            take_profit=p.take_profit,
            lots=lots,
            magic=self.s.magic_base + SYMBOLS.index(p.symbol),
            max_deviation_pips=self.s.max_deviation_pips,
            order_expiry_seconds=self.s.order_expiry_seconds,
        )
