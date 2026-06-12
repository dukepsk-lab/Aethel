from aethel.risk_gate.sizing import compute_lots


def test_eurusd_one_percent_risk():
    # 10,000 equity, 1% risk = $100; 20 pip SL at $10/pip/lot -> 0.5 lots
    lots = compute_lots("EURUSD", 10_000, 1.0, 1.0850, 1.0830)
    assert lots == 0.5


def test_rounds_down_to_lot_step():
    # $100 risk over 27 pips -> 0.3703 lots -> floored to 0.37
    lots = compute_lots("EURUSD", 10_000, 1.0, 1.08500, 1.08230)
    assert lots == 0.37


def test_below_min_lot_returns_zero():
    lots = compute_lots("EURUSD", 100, 0.1, 1.0850, 1.0750)
    assert lots == 0.0


def test_zero_sl_distance_returns_zero():
    assert compute_lots("EURUSD", 10_000, 1.0, 1.0850, 1.0850) == 0.0


def test_jpy_pair_uses_correct_pip_size():
    # USDJPY: 30 pip SL (0.30 in price), $6.7/pip/lot, $100 risk -> 0.49 lots
    lots = compute_lots("USDJPY", 10_000, 1.0, 150.00, 149.70)
    assert lots == 0.49
