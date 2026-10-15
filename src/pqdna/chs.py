"""Component Health Scores (CHS).

Maps the lifetime PQ-DNA stress vector to a per-subsystem health score in
[0.0, 1.0], where 1.0 = factory-new and 0.0 = end-of-life.

For component ``c`` the mapping is

    CHS_c = clip(1 - W_c . phi - b_c, 0, 1)
    phi   = scale * [S, age_years * S]            (24 inputs; scale = 1e-3)

where ``S`` is the 12-value PQ-DNA state, ``age_years`` is the device age in
years (run-days / 365) and ``[A, B]`` concatenates. Without the age block
(``use_age=False``) ``phi = scale * S`` has 12 inputs. W is learned from
accelerated-life-test (ALT) data. In this reference implementation the first 12
weights are hand-wired from documented physics-of-failure correlations
(`FAILURE_MODES`) and the age block starts at zero; `fit` then replaces them
using the synthetic survival data in `synth.py`.

See docs/METHOD.md, Sections 3.1-3.2.
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES
from .pqdna import PQDNAState

# Subsystems we monitor inside the meter.
COMPONENTS: List[str] = ["relay", "smps", "terminal_block", "mcu", "display"]


# Physics-of-failure mapping. For each subsystem we list the PQ-DNA dimensions
# that drive its dominant failure modes, with a (qualitative) weight. The
# weights of each subsystem sum to 1. They are the prior for the first 12
# entries of W (see ComponentHealthScorer) and the Miner's-rule weights w_ck of
# the latent ground truth in synth.py.
FAILURE_MODES: Dict[str, Dict[str, float]] = {
    "relay": {
        # Contact erosion from arc events on inrush; transient overvoltages stress coil.
        "inrush_count":   0.45,
        "transient_pu":   0.35,
        "i_crest":        0.10,
        "i_rms":          0.10,
    },
    "smps": {
        # Bulk electrolytic cap dry-out and inductor saturation drivers.
        "v_thd":          0.30,
        "i_thd":          0.25,
        "transient_pu":   0.20,
        "v_rms":          0.15,
        "freq_dev":       0.10,
    },
    "terminal_block": {
        # I^2R hot-spot fatigue and connection thermal cycling.
        "i_rms":          0.45,
        "i_crest":        0.30,
        "inrush_count":   0.15,
        "i_thd":          0.10,
    },
    "mcu": {
        # Decap aging from supply transients + brown-out from sags.
        "transient_pu":   0.35,
        "sag_dwell":      0.30,
        "swell_dwell":    0.15,
        "v_thd":          0.20,
    },
    "display": {
        # LCD bias-cap aging + flicker.
        "flicker_pst":    0.40,
        "v_rms":          0.25,
        "freq_dev":       0.15,
        "transient_pu":   0.20,
    },
}


def _weight_vector(component: str) -> np.ndarray:
    """Project the dict mapping into a length-N_FEATURES weight vector."""
    w = np.zeros(N_FEATURES, dtype=np.float32)
    table = FAILURE_MODES[component]
    for name, val in table.items():
        w[FEATURE_NAMES.index(name)] = val
    return w


@dataclass
class ComponentHealthScorer:
    """Linear scorer over the PQ-DNA state, fitted from labelled survival data.

    The estimator predicts the accumulated damage of component ``c`` as a linear
    function of the design vector ``phi = scale * [S, age_years * S]`` (or
    ``scale * S`` when ``use_age`` is False), ``damage_c = W_c . phi + b_c``, and
    maps it to health with ``CHS_c = clip(1 - damage_c, 0, 1)``. ``S`` is the raw
    (unscaled) PQ-DNA state and ``age_years`` the device age in years
    (run-days / 365).

    Parameters
    ----------
    W : np.ndarray, optional
        Weights, shape ``(n_components, n_inputs)``. Defaults to the
        physics-of-failure prior: `FAILURE_MODES` in the first 12 columns and
        zeros in the age block.
    bias : np.ndarray, optional
        Bias ``b``, shape ``(n_components,)``. Defaults to zeros.
    scale : float, default 1e-3
        Fixed stress-to-health scale ``s`` applied to ``S``. `pqdna.edge_export`
        folds it into the exported weights.
    use_age : bool, default True
        Include the ``S x age`` interaction block in the design matrix.

        This matters more than it looks. The PQ-DNA state ``S`` is an
        exponentially-weighted moving average, so it estimates the *current
        stress level* and reaches steady state within a few decay horizons.
        Accumulated fatigue damage, by contrast, is a monotone integral of that
        stress over the device's life. ``S`` alone therefore carries no
        information about how long the device has been exposed, and cannot
        identify accumulated damage no matter how the weights are chosen.
        Multiplying by elapsed age recovers the missing dose-time product.

        Age is free on real hardware - every device of this class already keeps
        a run-hour counter - and costs one extra ``uint32`` of NVM.

        Set to ``False`` only to reproduce the S-only ablation.
    """

    W: np.ndarray = None          # shape (n_components, n_inputs)
    bias: np.ndarray = None       # shape (n_components,)
    scale: float = 1.0e-3         # scale factor from stress-units to health-loss
    use_age: bool = True

    def __post_init__(self):
        if self.W is None:
            base = np.stack([_weight_vector(c) for c in COMPONENTS], axis=0)
            if self.use_age:
                # The age-interaction block is calibrated by `fit`; the hand-
                # wired prior only covers the instantaneous-stress block.
                self.W = np.hstack([base, np.zeros_like(base)]).astype(np.float32)
            else:
                self.W = base.astype(np.float32)
        if self.bias is None:
            self.bias = np.zeros(len(COMPONENTS), dtype=np.float32)

    @property
    def n_inputs(self) -> int:
        """Width of the design matrix (excluding bias)."""
        return N_FEATURES * (2 if self.use_age else 1)

    def _design(self, S: np.ndarray, age_years) -> np.ndarray:
        """Build the (N, n_inputs) design matrix ``phi`` from raw S and age.

        ``phi = scale * [S, age_years * S]``, or ``scale * S`` when ``use_age``
        is False. ``age_years`` is the device age in years (run-days / 365).
        """
        Sm = np.atleast_2d(np.asarray(S, dtype=np.float64)) * self.scale
        if not self.use_age:
            return Sm
        age = np.asarray(age_years, dtype=np.float64).reshape(-1, 1)
        return np.hstack([Sm, Sm * age])

    def score_batch(self, S: np.ndarray, age_years=0.0) -> np.ndarray:
        """Vectorised CHS for a batch of states. Returns (N, n_components).

        ``age_years`` is a scalar or an (N,) array of device ages in years.
        """
        X = self._design(S, age_years)
        damage = X @ self.W.T + self.bias
        return np.clip(1.0 - damage, 0.0, 1.0)

    def score(self, state: PQDNAState, age_years: float = 0.0) -> Dict[str, float]:
        """Return CHS_c in [0, 1] for each component; ``age_years`` is run-days / 365."""
        chs = self.score_batch(state.S[None, :], age_years)[0]
        return {c: float(chs[i]) for i, c in enumerate(COMPONENTS)}

    # ---- supervised refinement ---------------------------------------------
    def fit(
        self,
        S_matrix: np.ndarray,         # (N, n_features) PQ-DNA snapshots
        chs_target: np.ndarray,       # (N, n_components) observed CHS
        age_years: np.ndarray = None, # (N,) device age in years
        l2: float = 1e-3,
    ) -> ComponentHealthScorer:
        """Refine W and bias by ridge regression on ``damage = 1 - CHS_target``.

        Closed form: minimise ``sum_n (damage_n - W . phi_n - b)^2 + l2 * (|W|^2 + b^2)``.
        The bias is penalised like the weights. The clip is ignored during the
        fit; it is re-applied at scoring time.
        """
        if self.use_age and age_years is None:
            raise ValueError("age_years is required when use_age=True")
        X = self._design(S_matrix, 0.0 if age_years is None else age_years)
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        damage_target = 1.0 - chs_target            # (N, n_components)
        A = X.T @ X + l2 * np.eye(X.shape[1])
        B = X.T @ damage_target
        sol = np.linalg.solve(A, B)                 # (n_inputs+1, n_components)
        self.W = sol[:-1, :].T.astype(np.float32)
        self.bias = sol[-1, :].astype(np.float32)
        return self
