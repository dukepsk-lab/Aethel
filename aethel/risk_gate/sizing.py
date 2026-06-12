"""Deterministic lot sizing. This is the ONLY place lot size is computed.
LLMs propose risk_pct; this module turns it into lots using contract specs.
"""

from __future__ import annotations

import math

from aethel.config import SYMBOL_SPECS


def compute_lots(symbol: str, equity: float, risk_pct: float,
                 entry: float, stop_loss: float) -> float:
    spec = SYMBOL_SPECS[symbol]
    sl_pips = abs(entry - stop_loss) / spec.pip_size
    if sl_pips <= 0:
        return 0.0
    risk_amount = equity * (risk_pct / 100.0)
    lots = risk_amount / (sl_pips * spec.pip_value_per_lot)
    # round DOWN to lot step — never risk more than requested.
    # round() before floor() guards against float artifacts like 49.999999996.
    lots = math.floor(round(lots / spec.lot_step, 9)) * spec.lot_step
    if lots < spec.min_lot:
        return 0.0
    return round(lots, 2)
