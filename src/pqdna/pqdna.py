"""Lifetime PQ-DNA accumulator.

`PQDNAState` is the persistent, NVM-resident state: one value per PQ channel
(12 channels; 96 bytes as float64 in this NumPy reference, the same size as the
12 x int64 state of the fixed-point export in `pqdna.edge_export`).

The aggregator implements the core accumulation rule, channel by channel
(``*`` is the element-wise product):

    S_t = alpha * S_{t-1} + (1 - alpha) * g(x_t),    S_0 = 0
    alpha_k = exp(-dt / tau_k)            (dt and tau_k in seconds)

where ``x_t`` is the 12-channel feature vector of update ``t`` and ``g`` applies
the per-channel damage kernel ``g_k`` (see `vectorised_damage`) that maps the raw
PQ feature value to "damage equivalent dose". The kernels follow
physics-of-failure curves (I^2 R heating, Coffin-Manson power laws, contact
erosion). Each is non-decreasing in its stress driver, but only some are
non-linear: channels 6-11 (sag_dwell ... inrush_count) are linear.

The update is evaluated as ``S_t = S_{t-1} + gain * (g(x_t) - S_{t-1})`` with the
gain ``1 - alpha = -expm1(-dt / tau)``, in float64, so that it keeps its precision
when ``dt`` is much shorter than ``tau`` (a 1 s update of a 60-day channel has a
gain of 1.9e-7).

See docs/METHOD.md, Sections 2.2-2.3, for the kernel table and the units.
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
# drives. Long horizons (up to 60 days) reflect slow degradation; short ones
# (down to 1 day) reflect fast wear-out drivers.
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
    """Map a raw feature value ``x`` to a non-negative per-window damage dose.

    Scalar reference for `vectorised_damage`; the kernels g_0..g_11 and their
    constants are tabulated in docs/METHOD.md, Section 2.2. ``x`` must be
    non-negative for the fractional-power channels.
    """
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
    order. Accepts shape (...,N_FEATURES) and returns same shape (float32).
    The kernels g_0..g_11 are tabulated in docs/METHOD.md, Section 2.2.

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
    """The compact NVM-resident state.

    ``S`` holds the 12 PQ-DNA values (float64 here, 96 bytes), ``t_last`` the
    time of the last update in seconds since manufacture (8 bytes) and
    ``n_updates`` the number of updates. The fixed-point footprint (12 x int64
    for ``S`` plus a 4-byte run-day counter) is accounted in
    `pqdna.edge_export`. `PQDNAAggregator.update` keeps the dtype of ``S``, so a
    float32 state can be passed in but rounds every update to float32, which
    swallows the 1 s increments of the slowest channels.
    """
    S: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES, dtype=np.float64))
    t_last: float = 0.0       # last update time, seconds since meter birth.
    n_updates: int = 0

    def as_dict(self) -> dict:
        return {name: float(val) for name, val in zip(FEATURE_NAMES, self.S)}

    def copy(self) -> "PQDNAState":
        return PQDNAState(S=self.S.copy(), t_last=self.t_last, n_updates=self.n_updates)


class PQDNAAggregator:
    """Updates a `PQDNAState` from a `PQFeatures` observation.

    Parameters
    ----------
    dt_window_s : float, default 1.0
        Interval between updates, in seconds (the 1 s observation window on a
        device). The per-channel decay is ``alpha = exp(-dt_window_s / tau)``.
    tau_s : dict, optional
        Per-channel decay time constants in seconds; defaults to
        `DEFAULT_TAU_S`.

    Attributes
    ----------
    tau : np.ndarray
        Per-channel time constants in seconds, float64, in `FEATURE_NAMES` order.
    alpha : np.ndarray
        ``exp(-dt / tau)``, float64.
    gain : np.ndarray
        ``1 - alpha = -expm1(-dt / tau)``, float64. This is the quantity the
        fixed-point export stores, as ``round(2**32 * gain)``.

    Notes
    -----
    Everything is float64. At ``dt_window_s = 1`` the decay coefficient is within
    1.2e-5 of 1 (within 2e-7 for the 60-day channels), so evaluating or storing
    it in float32 distorts the effective time constants (by tens of percent for
    the 60-day channels, by an amount that depends on the platform's float32
    ``exp``). The float32-free update below is exact up to float64 rounding.
    """

    def __init__(self, dt_window_s: float = 1.0, tau_s: Optional[dict] = None) -> None:
        self.dt = float(dt_window_s)
        taus = tau_s or DEFAULT_TAU_S
        self.tau = np.array([taus[n] for n in FEATURE_NAMES], dtype=np.float64)
        # Per-dimension EWMA decay coefficient alpha = exp(-dt/tau) and its
        # complement, the gain 1 - alpha, which expm1 keeps exact for dt << tau.
        self.alpha = np.exp(-self.dt / self.tau)
        self.gain = -np.expm1(-self.dt / self.tau)

    def update(self, state: PQDNAState, feats: PQFeatures) -> PQDNAState:
        """Advance ``state`` by one update of ``dt_window_s`` seconds, in place."""
        f = feats.as_vector()
        g = np.array(
            [_damage_nonlinearity(n, float(v)) for n, v in zip(FEATURE_NAMES, f)],
            dtype=np.float64,
        )
        S = state.S
        state.S = (S + self.gain * (g - S)).astype(S.dtype, copy=False)
        state.t_last += self.dt
        state.n_updates += 1
        return state
