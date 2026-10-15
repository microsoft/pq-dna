"""Remaining-Useful-Life estimator.

RUL is framed as a regression from the current snapshot to the number of days
until failure. The inputs are the PQ-DNA state S (12 values), the per-component
health scores CHS (5 values) and the device age in years (run-days / 365); the
target ``ttf_days`` is supplied by the caller. In `pqdna.demo` it is the number
of days to the first weekly snapshot at which any component's CHS falls below
0.05 (devices that never fail contribute the days to the end of the horizon,
right-censored), and the CHS inputs are the simulator's ground-truth scores
rather than the output of `ComponentHealthScorer`.

Two model heads are provided, both fitted to the same target:

  * `RULEstimator(model='ridge')`  - one closed-form linear model,
        RUL = max(0, beta . [S, CHS, age_years, 1]),
    with 19 coefficients (12 + 5 + 1 + intercept) shared by all components,
    which the fixed-point export of `pqdna.edge_export` stores in 77 bytes.
  * `RULEstimator(model='gbr')`    - sklearn GradientBoostingRegressor for
    higher accuracy in the cloud-side reference.

The ridge variant is the firmware-resident model. The GBR variant is a
cloud-side reference for benchmarking only. See docs/METHOD.md, Section 3.3.
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
    """Ridge or gradient-boosting regressor of days-to-failure.

    Parameters
    ----------
    model : {"ridge", "gbr"}, default "ridge"
        Model head (see the module docstring). The failure criterion is not a
        parameter of the estimator: it is defined by the ``ttf_days`` target
        passed to `fit` (CHS < 0.05 in `pqdna.demo`).
    l2 : float, default 0.1
        Ridge penalty, applied to the intercept as well as the coefficients
        (`pqdna.demo` uses 10).
    """
    model: Literal["ridge", "gbr"] = "ridge"
    l2: float = 1e-1
    coefs_: Optional[np.ndarray] = field(default=None, repr=False)       # ridge: (19,) = 12 S + 5 CHS + 1 age + intercept
    gbrs_: Optional[Dict[str, GradientBoostingRegressor]] = field(default=None, repr=False)

    def _features(self, S: np.ndarray, chs: np.ndarray, age_years: np.ndarray = None) -> np.ndarray:
        """Concatenate (N, n_feat), (N, n_comp) and the (N,) age column.

        S is used raw (not scaled) and ``age_years`` is in years (run-days / 365).
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
        """Fit the selected head to ``ttf_days``, the days from each snapshot to failure.

        A single regressor is fitted, not one per component: every component
        head regresses the same target, and the CHS columns let the model tell
        which component is closest to failure. For ``model='ridge'`` the closed
        form is ``beta = (Z'Z + l2 I)^-1 Z'y`` with ``Z = [S, CHS, age_years, 1]``,
        so the intercept is penalised like the other 18 coefficients and
        ``coefs_`` has length 19. ``age_years`` is in years (run-days / 365).
        """
        X = self._features(S, chs, age_years)
        # We use the same ttf_days target for each component head; the per-
        # component CHS columns let each head distinguish which component is
        # actually closest to failure.
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
        """Predicted days to failure, clamped at zero.

        Ridge: ``max(0, [S, CHS, age_years, 1] @ coefs_)``. GBR: the minimum over
        the component heads (which all alias one model), clamped at zero.
        """
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
