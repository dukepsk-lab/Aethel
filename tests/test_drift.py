"""Tests for Venus drift detection."""
import numpy as np
import pytest
from aethel.venus.drift import (
    _psi, _ks, check_calibration_drift, check_feature_drift,
    run_drift_check, DriftReport, PSI_THRESHOLD, KS_THRESHOLD, CALIB_DRIFT_THRESHOLD
)


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 500)
    psi = _psi(data, data)
    assert psi < 0.01


def test_psi_different_distributions():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    cur = rng.normal(3, 1, 500)  # shifted by 3 sigma
    psi = _psi(ref, cur)
    assert psi > PSI_THRESHOLD


def test_ks_identical():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 300)
    ks = _ks(data, data)
    assert ks == 0.0


def test_ks_different():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 300)
    cur = rng.normal(5, 1, 300)
    ks = _ks(ref, cur)
    assert ks > KS_THRESHOLD


def test_calibration_drift_no_drift():
    probs = np.full(100, 0.6)
    outcomes = np.array([1] * 60 + [0] * 40)
    drift = check_calibration_drift(probs, outcomes)
    assert drift < CALIB_DRIFT_THRESHOLD


def test_calibration_drift_large():
    probs = np.full(100, 0.8)
    outcomes = np.array([1] * 40 + [0] * 60)  # actual hit rate 0.4
    drift = check_calibration_drift(probs, outcomes)
    assert drift > CALIB_DRIFT_THRESHOLD
    assert abs(drift - 0.4) < 0.01


def test_calibration_drift_too_few_samples():
    probs = np.array([0.8, 0.7])
    outcomes = np.array([0, 0])
    drift = check_calibration_drift(probs, outcomes)
    assert drift == 0.0


def test_run_drift_check_no_alert():
    probs = np.full(100, 0.55)
    outcomes = np.array([1] * 55 + [0] * 45)
    report = run_drift_check("BTCUSDT", probs, outcomes)
    assert report.symbol == "BTCUSDT"
    assert not report.calibration_alert
    assert not report.has_alert()


def test_run_drift_check_with_alert():
    probs = np.full(100, 0.9)
    outcomes = np.array([1] * 30 + [0] * 70)
    report = run_drift_check("ETHUSDT", probs, outcomes)
    assert report.calibration_alert
    assert report.has_alert()


def test_run_drift_check_feature_drift():
    rng = np.random.default_rng(0)
    ref = {"rsi": rng.uniform(30, 70, 200), "macd": rng.normal(0, 1, 200)}
    cur = {"rsi": rng.uniform(70, 100, 200), "macd": rng.normal(5, 1, 200)}
    probs = np.full(100, 0.55)
    outcomes = np.array([1] * 55 + [0] * 45)
    report = run_drift_check("BTCUSDT", probs, outcomes, ref_features=ref, cur_features=cur)
    assert "rsi" in report.feature_psi
    assert len(report.psi_alerts) > 0 or len(report.ks_alerts) > 0


def test_drift_report_summary_no_alert():
    report = DriftReport(symbol="X")
    assert report.summary() == "no drift"


def test_drift_report_summary_with_alert():
    report = DriftReport(symbol="X", calibration_drift=0.15, calibration_alert=True)
    assert "calibration drift" in report.summary()
