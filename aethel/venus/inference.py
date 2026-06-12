"""Live inference. Venus models are trained offline, versioned, and loaded
as frozen artifacts — no online learning during trading hours.

Artifact layout (per symbol):
    models/artifacts/{SYMBOL}/model.pt
    models/artifacts/{SYMBOL}/calibrator.pkl
    models/artifacts/{SYMBOL}/version.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aethel.core.schemas import Action, Candle, Timeframe, VenusSignal
from aethel.venus.calibration import Calibrator
from aethel.venus.features import FEATURE_COLUMNS, compute_features

SEQ_LEN = {Timeframe.M1: 120, Timeframe.M15: 96, Timeframe.H1: 72}


class VenusInference:
    def __init__(self, artifacts_dir: Path, symbol: str) -> None:
        import torch

        from aethel.venus.model import VenusNet

        self.symbol = symbol
        sym_dir = artifacts_dir / symbol
        self.version = (sym_dir / "version.txt").read_text().strip()
        self.model = VenusNet(n_features=len(FEATURE_COLUMNS))
        self.model.load_state_dict(torch.load(sym_dir / "model.pt", map_location="cpu"))
        self.model.eval()
        self.calibrator = Calibrator.load(sym_dir / "calibrator.pkl")
        self._torch = torch

    def predict(self, candles: dict[Timeframe, list[Candle]]) -> VenusSignal:
        torch = self._torch
        batch = {}
        latest_features: dict[str, float] = {}
        for tf, seq_len in SEQ_LEN.items():
            df = pd.DataFrame([c.model_dump() for c in candles[tf]]).set_index("time")
            feats = compute_features(df).ffill().fillna(0.0).tail(seq_len)
            if len(feats) < seq_len:
                raise ValueError(f"not enough {tf} history for {self.symbol}")
            batch[tf.value] = torch.tensor(
                feats.values[None, :, :], dtype=torch.float32
            )
            if tf == Timeframe.M15:
                latest_features = feats.iloc[-1].to_dict()

        with torch.no_grad():
            logit = self.model(batch).item()
        raw = 1 / (1 + np.exp(-logit))
        confidence = float(self.calibrator.transform(np.array([raw]))[0])

        # Direction from H1 trend context; confidence gates whether the
        # setup is worth taking in that direction.
        direction = Action.BUY if latest_features.get("dist_ema50", 0) >= 0 else Action.SELL
        return VenusSignal(
            symbol=self.symbol,
            direction=direction,
            confidence=confidence,
            features={k: round(float(v), 5) for k, v in latest_features.items()},
            model_version=self.version,
        )
