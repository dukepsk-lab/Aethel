"""Tests for Venus auto-retrain promotion logic."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from aethel.venus.retrain import _promotion_criteria, get_registry, _load_registry, _save_registry


def test_promotion_criteria_passes():
    bench = {"TCN (champion)": {"mean_auc": 0.55, "avg_r": 0.02, "trades": 100}}
    assert _promotion_criteria(bench) is True


def test_promotion_criteria_low_auc():
    bench = {"TCN (champion)": {"mean_auc": 0.50, "avg_r": 0.02, "trades": 100}}
    assert _promotion_criteria(bench) is False


def test_promotion_criteria_negative_avg_r():
    bench = {"TCN (champion)": {"mean_auc": 0.55, "avg_r": -0.01, "trades": 100}}
    assert _promotion_criteria(bench) is False


def test_promotion_criteria_zero_avg_r():
    bench = {"TCN (champion)": {"mean_auc": 0.55, "avg_r": 0.0, "trades": 100}}
    assert _promotion_criteria(bench) is False


def test_promotion_criteria_none_avg_r():
    bench = {"TCN (champion)": {"mean_auc": 0.55, "avg_r": None, "trades": 100}}
    assert _promotion_criteria(bench) is False


def test_promotion_criteria_missing_key():
    bench = {}
    assert _promotion_criteria(bench) is False


def test_get_registry_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import aethel.venus.retrain as retrain_mod
    monkeypatch.setattr(retrain_mod, "REGISTRY_FILE", tmp_path / "registry.json")
    result = get_registry()
    assert result == {}


def test_save_and_load_registry(tmp_path, monkeypatch):
    import aethel.venus.retrain as retrain_mod
    monkeypatch.setattr(retrain_mod, "REGISTRY_FILE", tmp_path / "models" / "registry.json")
    data = {"BTCUSDT": {"action": "promoted", "benchmark": {}}}
    retrain_mod._save_registry(data)
    loaded = retrain_mod._load_registry()
    assert loaded["BTCUSDT"]["action"] == "promoted"
