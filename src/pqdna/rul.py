"""Remaining-Useful-Life estimator.

We frame RUL as a per-component regression: given the PQ-DNA snapshot S_t
and the per-component CHS_t trajectory observed in the recent window,
predict the number of days until CHS_i crosses the maintenance threshold
(default 0.20).

Two model heads are provided:

  * `RULEstimator(model='ridge')`  - per-component closed-form linear model
                                      that fits cleanly on a meter MCU
                                      (8 ints + 1 bias per component).
  * `RULEstimator(model='gbr')`    - sklearn GradientBoostingRegressor for
                                      higher accuracy in the cloud-side
                                      reference.

The ridge variant is the firmware-resident model. The GBR variant is a
cloud-side reference for benchmarking only.
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAVE_SK = True
except Exception:                       # pragma: no cover
    _HAVE_SK = False

from .chs import COMPONENTS


@dataclass
class RULEstimator:
    model: Literal["ridge", "gbr"] = "ridge"
    threshold: float = 0.20
    l2: float = 1e-1
    coefs_: Optional[np.ndarray] = field(default=None, repr=False)       # ridge: (n_components, n_features+1)
    gbrs_: Optional[Dict[str, GradientBoostingRegressor]] = field(default=None, repr=False)

    def _features(self, S: np.ndarray, chs: np.ndarray, age_years: np.ndarray = None) -> np.ndarray:
        """Concatenate (N, n_feat), (N, n_comp) and the (N,) age column.

        Age is included for the same reason as in
        :class:`~pqdna.chs.ComponentHealthScorer`: the EWMA state S is a
        stress-*level* estimate and carries no exposure-duration information,
        which time-to-failure obviously depends on.
        """
        blocks = [np.asarray(S, dtype=np.float64), np.asarray(chs, dtype=np.float64)]
        if age_years is not None:
            blocks.append(np.asarray(age_years, dtype=np.float64).reshape(-1, 1))
        return np.hstack(blocks)

    def fit(
        self,
        S: np.ndarray,
        chs: np.ndarray,
        ttf_days: np.ndarray,
        age_years: np.ndarray = None,
    ) -> RULEstimator:
        """ttf_days is the *minimum* across components for a given snapshot, but
        we fit one regressor per component using the per-component CHS slope
        as the discriminator.
        """
        X = self._features(S, chs, age_years)
        # We use the same ttf_days target for each component head; the per-
        # component CHS columns let each head distinguish which component is
        # actually closest to threshold.
        y = ttf_days.astype(np.float64)
        if self.model == "ridge":
            X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
            A = X_aug.T @ X_aug + self.l2 * np.eye(X_aug.shape[1])
            B = X_aug.T @ y
            sol = np.linalg.solve(A, B)
            self.coefs_ = sol.astype(np.float32)
        elif self.model == "gbr":
            if not _HAVE_SK:
                raise RuntimeError("sklearn not installed; pip install scikit-learn")
            self.gbrs_ = {}
            # All component heads regress the *same* target (min-ttf); a single
            # GBR is sufficient. Train one and alias it for every component.
            m = GradientBoostingRegressor(
                n_estimators=80, max_depth=3, learning_rate=0.08, random_state=0,
            )
            m.fit(X, y)
            for c in COMPONENTS:
                self.gbrs_[c] = m
        else:
            raise ValueError(f"Unknown model {self.model!r}")
        return self

    def predict_days(
        self,
        S: np.ndarray,
        chs: np.ndarray,
        age_years: np.ndarray = None,
    ) -> np.ndarray:
        X = self._features(S, chs, age_years)
        if self.model == "ridge":
            assert self.coefs_ is not None, "call fit() first"
            X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
            pred = X_aug @ self.coefs_
            return np.maximum(pred, 0.0)
        else:
            assert self.gbrs_ is not None, "call fit() first"
            preds = np.stack([self.gbrs_[c].predict(X) for c in COMPONENTS], axis=1)
            return preds.min(axis=1).clip(min=0.0)
