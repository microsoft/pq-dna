"""Lifetime PQ-DNA accumulator.

`PQDNAState` is the persistent, NVM-resident state. It is intentionally tiny:
12 floats x 4 bytes = 48 bytes per stress channel x the number of channels we
track.

The aggregator implements the core accumulation rule:

    S_t = alpha * S_{t-1} + (1 - alpha) * g(features_t)

where `g(.)` is a per-dimension, monotone non-linearity that maps the raw PQ
feature value to "damage equivalent dose". The non-linearities reflect
physics-of-failure curves (Coffin-Manson, Arrhenius, contact-erosion) - they
are NOT linear in the stress driver.
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES, PQFeatures

# Per-feature decay time constant (seconds). Tuned so each PQ axis has a
# memory horizon matched to the physics of the dominant failure modes it
# drives. Long horizons (months) reflect slow degradation; short ones
# (hours) reflect fast wear-out drivers.
DEFAULT_TAU_S = {
    "v_rms":         30 * 24 * 3600.0,   # 30 days  - electrolytic dry-out
    "i_rms":         30 * 24 * 3600.0,   # 30 days  - I^2R heating
    "v_thd":         60 * 24 * 3600.0,   # 60 days  - SMPS / cap stress
    "i_thd":         60 * 24 * 3600.0,
    "i_crest":        7 * 24 * 3600.0,   #  7 days  - terminal-block hot-spots
    "transient_pu":   1 * 24 * 3600.0,   #  1 day   - relay contact erosion
    "sag_dwell":      7 * 24 * 3600.0,
    "swell_dwell":    7 * 24 * 3600.0,
    "flicker_pst":   30 * 24 * 3600.0,
    "freq_dev":      30 * 24 * 3600.0,
    "unbalance":     30 * 24 * 3600.0,
    "inrush_count":   1 * 24 * 3600.0,
}


def _damage_nonlinearity(name: str, x: float) -> float:
    """Map a raw feature value to a per-window damage dose in [0, ~10]."""
    if name == "v_rms":
        # Damage from sustained off-nominal voltage, symmetric in |v-1|.
        return float(50.0 * (x - 1.0) ** 2)
    if name == "i_rms":
        # I^2 heating.
        return float(x ** 2)
    if name == "v_thd" or name == "i_thd":
        # SMPS / cap stress grows ~ THD^1.5.
        return float(20.0 * (x ** 1.5))
    if name == "i_crest":
        # Crest factor above sinusoidal sqrt(2) is the damaging part.
        excess = max(0.0, x - 1.414)
        return float(excess ** 2)
    if name == "transient_pu":
        # Coffin-Manson-like power-law on transient amplitude.
        return float(10.0 * (x ** 2.5))
    if name in ("sag_dwell", "swell_dwell"):
        return float(10.0 * x)
    if name == "flicker_pst":
        return float(x)
    if name == "freq_dev":
        return float(1000.0 * x)
    if name == "unbalance":
        return float(100.0 * x)
    if name == "inrush_count":
        # Arc / contact erosion grows linearly with arc events.
        return float(x)
    return float(x)


def vectorised_damage(features: np.ndarray) -> np.ndarray:
    """Vectorised version of `_damage_nonlinearity` matching `FEATURE_NAMES`
    order. Accepts shape (...,N_FEATURES) and returns same shape.

    Indices (must match `features.FEATURE_NAMES`):
        0 v_rms, 1 i_rms, 2 v_thd, 3 i_thd, 4 i_crest, 5 transient_pu,
        6 sag_dwell, 7 swell_dwell, 8 flicker_pst, 9 freq_dev,
        10 unbalance, 11 inrush_count
    """
    f = np.asarray(features, dtype=np.float32)
    out = np.empty_like(f)
    out[..., 0] = 50.0 * (f[..., 0] - 1.0) ** 2          # v_rms
    out[..., 1] = f[..., 1] ** 2                         # i_rms
    out[..., 2] = 20.0 * np.power(np.maximum(f[..., 2], 0.0), 1.5)   # v_thd
    out[..., 3] = 20.0 * np.power(np.maximum(f[..., 3], 0.0), 1.5)   # i_thd
    out[..., 4] = np.maximum(f[..., 4] - 1.414, 0.0) ** 2            # i_crest
    out[..., 5] = 10.0 * np.power(np.maximum(f[..., 5], 0.0), 2.5)   # transient
    out[..., 6] = 10.0 * f[..., 6]                                   # sag_dwell
    out[..., 7] = 10.0 * f[..., 7]                                   # swell_dwell
    out[..., 8] = f[..., 8]                                          # flicker
    out[..., 9] = 1000.0 * f[..., 9]                                 # freq_dev
    out[..., 10] = 100.0 * f[..., 10]                                # unbalance
    out[..., 11] = f[..., 11]                                        # inrush
    return out


@dataclass
class PQDNAState:
    """The compact NVM-resident state. ~48 + 8 = 56 bytes when packed."""
    S: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES, dtype=np.float32))
    t_last: float = 0.0       # last update time, seconds since meter birth.
    n_updates: int = 0

    def as_dict(self) -> dict:
        return {name: float(val) for name, val in zip(FEATURE_NAMES, self.S)}

    def copy(self) -> "PQDNAState":
        return PQDNAState(S=self.S.copy(), t_last=self.t_last, n_updates=self.n_updates)


class PQDNAAggregator:
    """Updates a `PQDNAState` from a `PQFeatures` observation."""

    def __init__(self, dt_window_s: float = 1.0, tau_s: Optional[dict] = None) -> None:
        self.dt = float(dt_window_s)
        taus = tau_s or DEFAULT_TAU_S
        self.tau = np.array(
            [taus[n] for n in FEATURE_NAMES], dtype=np.float32
        )
        # Per-dimension EWMA decay coefficient alpha = exp(-dt/tau).
        self.alpha = np.exp(-self.dt / self.tau).astype(np.float32)

    def update(self, state: PQDNAState, feats: PQFeatures) -> PQDNAState:
        f = feats.as_vector()
        g = np.array(
            [_damage_nonlinearity(n, float(v)) for n, v in zip(FEATURE_NAMES, f)],
            dtype=np.float32,
        )
        state.S = self.alpha * state.S + (1.0 - self.alpha) * g
        state.t_last += self.dt
        state.n_updates += 1
        return state
