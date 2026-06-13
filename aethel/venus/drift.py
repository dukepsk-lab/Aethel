"""Drift detection for Venus — calibration drift + feature/concept drift.

Two silent failure modes degrade a frozen model:

* **calibration drift** — the predicted confidence no longer matches the
  realised hit-rate (e.g. Venus says 0.70 but only 0.55 of those trades win).
* **feature/concept drift** — the live feature distribution diverges from the
  one the model was trained on (regime change). Measured per-feature with the
  Population Stability Index (PSI) and a two-sample Kolmogorov–Smirnov stat.

All pure numpy — no LLM calls, no scipy dependency. A drift alert is advisory:
it surfaces in the Themis weekly audit and via Telegram, and can trigger a
retrain/recalibration; it never blocks live trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

PSI_THRESHOLD = 0.20          # PSI > 0.20 = significant population shift
KS_THRESHOLD = 0.10           # KS statistic > 0.10 = distribution divergence
CALIB_DRIFT_THRESHOLD = 0.10  # |mean predicted - actual hit-rate| > 10%


@dataclass
class DriftReport:
    symbol: str
    calibration_drift: float | None = None
    calibration_alert: bool = False
    feature_psi: dict[str, float] = field(default_factory=dict)
    feature_ks: dict[str, float] = field(default_factory=dict)
    psi_alerts: list[str] = field(default_factory=list)
    ks_alerts: list[str] = field(default_factory=list)

    def has_alert(self) -> bool:
        return self.calibration_alert or bool(self.psi_alerts) or bool(self.ks_alerts)

    def summary(self) -> str:
        parts: list[str] = []
        if self.calibration_alert and self.calibration_drift is not None:
            parts.append(f"calibration drift={self.calibration_drift:.3f}")
        if self.psi_alerts:
            parts.append(f"PSI alerts: {self.psi_alerts}")
        if self.ks_alerts:
            parts.append(f"KS alerts: {self.ks_alerts}")
        return "; ".join(parts) if parts else "no drift"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "calibration_drift": self.calibration_drift,
            "calibration_alert": self.calibration_alert,
            "feature_psi": self.feature_psi,
            "feature_ks": self.feature_ks,
            "psi_alerts": self.psi_alerts,
            "ks_alerts": self.ks_alerts,
            "has_alert": self.has_alert(),
        }


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions (binned on the
    combined range). Symmetric-ish and bounded away from 0 by an epsilon."""
    eps = 1e-8
    lo = float(min(expected.min(), actual.min()))
    hi = float(max(expected.max(), actual.max())) + eps
    edges = np.linspace(lo, hi, bins + 1)
    e = np.histogram(expected, bins=edges)[0].astype(float)
    a = np.histogram(actual, bins=edges)[0].astype(float)
    e = np.maximum(e / max(e.sum(), eps), eps)
    a = np.maximum(a / max(a.sum(), eps), eps)
    return float(np.sum((a - e) * np.log(a / e)))


def _ks(ref: np.ndarray, cur: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (max gap between empirical
    CDFs). Pure numpy — equivalent to scipy.stats.ks_2samp().statistic."""
    ref = np.sort(ref)
    cur = np.sort(cur)
    allv = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, allv, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, allv, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def check_calibration_drift(predicted_probs: np.ndarray,
                            actual_outcomes: np.ndarray) -> float:
    """|mean predicted confidence - actual hit-rate|. 0.0 when too few samples."""
    if len(predicted_probs) < 10:
        return 0.0
    return abs(float(predicted_probs.mean()) - float(actual_outcomes.mean()))


def check_feature_drift(ref_features: dict[str, np.ndarray],
                        cur_features: dict[str, np.ndarray]) -> DriftReport:
    """PSI + KS per feature for features present in both maps."""
    report = DriftReport(symbol="")
    for feat, ref_vals in ref_features.items():
        cur_vals = cur_features.get(feat)
        if cur_vals is None or len(ref_vals) < 30 or len(cur_vals) < 30:
            continue
        psi = _psi(np.asarray(ref_vals), np.asarray(cur_vals))
        ks = _ks(np.asarray(ref_vals), np.asarray(cur_vals))
        report.feature_psi[feat] = round(psi, 4)
        report.feature_ks[feat] = round(ks, 4)
        if psi > PSI_THRESHOLD:
            report.psi_alerts.append(feat)
        if ks > KS_THRESHOLD:
            report.ks_alerts.append(feat)
    return report


def run_drift_check(symbol: str,
                    predicted_probs: np.ndarray,
                    actual_outcomes: np.ndarray,
                    ref_features: dict[str, np.ndarray] | None = None,
                    cur_features: dict[str, np.ndarray] | None = None) -> DriftReport:
    """Full drift check for one symbol — calibration always, feature drift when
    reference + current feature maps are supplied."""
    report = DriftReport(symbol=symbol)
    drift = check_calibration_drift(np.asarray(predicted_probs),
                                    np.asarray(actual_outcomes))
    report.calibration_drift = round(drift, 4)
    report.calibration_alert = drift > CALIB_DRIFT_THRESHOLD

    if ref_features and cur_features:
        feat = check_feature_drift(ref_features, cur_features)
        report.feature_psi = feat.feature_psi
        report.feature_ks = feat.feature_ks
        report.psi_alerts = feat.psi_alerts
        report.ks_alerts = feat.ks_alerts

    if report.has_alert():
        logger.warning("[drift] %s: %s", symbol, report.summary())
    return report
