"""Trade lifecycle state machine.

Every decision is persisted with its state so the same approval can never be
executed twice and a crash-restart can reconcile open MT5 positions against
the database instead of starting blind.
"""

from enum import StrEnum


class TradeState(StrEnum):
    SIGNAL = "SIGNAL"            # Venus emitted a signal
    GATED = "GATED"              # passed the Signal Gate, agents consulted
    PROPOSED = "PROPOSED"        # Ares produced a proposal
    VETOED = "VETOED"            # Athena vetoed — terminal
    APPROVED = "APPROVED"        # Athena approved
    RISK_REJECTED = "RISK_REJECTED"  # Risk Gate rejected — terminal
    RISK_CHECKED = "RISK_CHECKED"    # ValidatedOrder produced
    EXECUTED = "EXECUTED"        # order placed (or shadow-logged)
    EXPIRED = "EXPIRED"          # TTL elapsed before execution — terminal
    MANAGED = "MANAGED"          # position open, rule-based management active
    CLOSED = "CLOSED"            # position closed
    ANALYZED = "ANALYZED"        # Mnemosyne post-trade analysis stored — terminal


_TRANSITIONS: dict[TradeState, set[TradeState]] = {
    TradeState.SIGNAL: {TradeState.GATED},
    TradeState.GATED: {TradeState.PROPOSED},
    TradeState.PROPOSED: {TradeState.VETOED, TradeState.APPROVED},
    TradeState.APPROVED: {TradeState.RISK_REJECTED, TradeState.RISK_CHECKED,
                          TradeState.EXPIRED},
    TradeState.RISK_CHECKED: {TradeState.EXECUTED, TradeState.EXPIRED},
    TradeState.EXECUTED: {TradeState.MANAGED, TradeState.CLOSED},
    TradeState.MANAGED: {TradeState.CLOSED},
    TradeState.CLOSED: {TradeState.ANALYZED},
    TradeState.VETOED: set(),
    TradeState.RISK_REJECTED: set(),
    TradeState.EXPIRED: set(),
    TradeState.ANALYZED: set(),
}


class InvalidTransition(Exception):
    pass


def transition(current: TradeState, new: TradeState) -> TradeState:
    if new not in _TRANSITIONS[current]:
        raise InvalidTransition(f"{current} -> {new} is not allowed")
    return new
