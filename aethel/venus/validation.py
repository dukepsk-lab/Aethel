"""Purged walk-forward cross-validation.

Random splits leak with overlapping triple-barrier labels: a sample whose
barrier window extends into the test period sees the future. We therefore:
- split strictly forward in time,
- PURGE training samples whose label window (t_end) overlaps the test start,
- EMBARGO a buffer of bars after each test fold before training resumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedWalkForward:
    n_splits: int = 5
    embargo_bars: int = 100

    def split(
        self, n_samples: int, t_end: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        fold = n_samples // (self.n_splits + 1)
        for k in range(self.n_splits):
            test_start = fold * (k + 1)
            test_stop = min(test_start + fold, n_samples)
            test_idx = np.arange(test_start, test_stop)

            train_mask = np.zeros(n_samples, dtype=bool)
            train_mask[:test_start] = True
            # purge: drop training samples whose label window reaches the test fold
            train_mask &= t_end < test_start
            # embargo after the test fold (for any post-test training data)
            post = test_stop + self.embargo_bars
            train_mask[test_stop:post] = False

            yield np.where(train_mask)[0], test_idx
