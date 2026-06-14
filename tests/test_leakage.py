"""Lookahead / leakage guards for the Venus+Helios feature pipeline.

These tests assert that the multi-timeframe windowing used at training time can
never see a higher-timeframe bar that closes after the trade's execution time.
Regression guard for the H1 resample lookahead fixed in this module.
"""

import numpy as np
import pandas as pd

from aethel.venus.train import resample_frames, remap_t_end_to_samples, SEQ


def _synth_m5(n: int = 5000) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    price = 1.10 + np.cumsum(rng.normal(0, 1e-4, n))
    high = price + np.abs(rng.normal(0, 5e-5, n))
    low = price - np.abs(rng.normal(0, 5e-5, n))
    return pd.DataFrame(
        {"open": price, "high": high, "low": low, "close": price,
         "tick_volume": rng.integers(50, 200, n)},
        index=idx,
    )


def test_h1_window_never_sees_future_bars():
    """For every M15 entry bar, the H1 bars selected by ``loc[:ts]`` must all
    close at or before the entry execution time (M15 bar's right edge)."""
    m5 = _synth_m5()
    frames = resample_frames(m5)
    h1 = frames["H1"]
    m15 = frames["M15"]

    # H1 bars are right-labeled => index == close time of the bar.
    for ts in m15.index[200::37]:  # sample a spread of entry bars
        entry_exec_time = ts + pd.Timedelta("15min")  # M15 left-labeled: close at +15m
        selected = h1.loc[:ts]
        if selected.empty:
            continue
        # every selected H1 bar must have closed by the execution time
        assert selected.index.max() <= entry_exec_time, (
            f"H1 lookahead at {ts}: selected bar closing "
            f"{selected.index.max()} > entry {entry_exec_time}"
        )


def test_h1_right_label_is_close_time():
    """H1 bar timestamp equals the close time (right edge), one hour after the
    M5 bars it aggregates begin."""
    m5 = _synth_m5(1000)
    h1 = resample_frames(m5)["H1"]
    # first H1 bar aggregates [00:00, 01:00) and is labeled 01:00
    assert h1.index[0] == m5.index[0] + pd.Timedelta("1h")


def test_remap_t_end_to_sample_coords():
    """t_end must be expressed in sample-array coordinates so the purge in
    PurgedWalkForward compares like-with-like even when samples are a subset."""
    # frame of 100 M15 bars, but only every 3rd becomes a sample (Helios-like)
    frame_index = pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC")
    sample_positions = np.arange(0, 100, 3)          # frame positions kept
    sample_times = list(frame_index[sample_positions])
    # each sample's barrier closes 9 frame-bars later (frame coords)
    t_end_frame = np.minimum(sample_positions + 9, 99)

    t_end_sample = remap_t_end_to_samples(t_end_frame, frame_index, sample_times)

    # result must index into the sample array (0..len(samples)), not the frame
    assert t_end_sample.max() < len(sample_times)
    # barrier of sample k closes 9 frame-bars = 3 sample-bars later -> ~k+3
    assert t_end_sample[0] == 3
