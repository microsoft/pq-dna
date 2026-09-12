"""Synthetic dataset generator for a fleet of smart meters.

The generator instantiates `N` virtual meters each with a randomly drawn
electrical environment (urban / EV-corridor / industrial / clean-grid),
streams plausible voltage / current waveforms for a multi-year horizon,
runs them through the same feature extractor and PQ-DNA aggregator that the
firmware would use, and emits per-meter snapshots of:

    (timestamp, PQ-DNA state, true component-health, failure events)

so downstream models (CHS, RUL) have labelled supervision.

Ground-truth generators
-----------------------
Two generators are provided, selected with the ``generator`` argument:

``"mismatched"`` (default)
    The latent degradation process is deliberately **structurally different**
    from the estimator in :mod:`pqdna.chs`. Specifically it differs in four
    ways, each of which is a known property of real component degradation:

    1. *Different per-channel damage exponents.* The device-side kernel
       ``pqdna.vectorised_damage`` uses one set of exponents; the latent
       process uses another (:data:`LATENT_DOSE_EXP`). No estimator can
       recover the latent exponents from S alone.
    2. *Cumulative rather than exponentially-weighted damage.* True fatigue
       damage is a monotone integral (Miner's rule). The device's S vector is
       an EWMA, i.e. a *stress-level* estimate that reaches steady state and
       then stays flat. The two are fundamentally different functionals of the
       same input stream, and this is the single most important mismatch.
    3. *Non-linear dose-to-damage link.* Health follows ``exp(-(L/L_ref)**m)``
       with a per-component shape exponent ``m``, not ``clip(1 - W.S)``.
    4. *Cross-component coupling.* A degrading SMPS accelerates MCU and
       display aging, and so on. The estimator treats components independently.

    Additionally the device-visible S vector carries multiplicative sensor
    noise, so the estimator does not even observe the same S the latent
    process was driven by.

``"matched"``
    The original generator: ground truth is ``clip(1 - W.S * scale / r)`` using
    the *same* ``W`` the estimator is initialised with and later fits. This is
    retained **only** so the demo can quantify how much of the previously
    reported accuracy was an artefact of model matching. It must not be used
    to report headline accuracy.
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chs import COMPONENTS, FAILURE_MODES
from .features import (
    FEATURE_NAMES,
    N_FEATURES,
)
from .pqdna import PQDNAAggregator, PQDNAState, vectorised_damage

# A small catalogue of typical environments. Each describes the *intensity*
# of the various PQ stressors that meters in that environment experience.
ENVIRONMENTS: Dict[str, Dict[str, float]] = {
    "clean_grid": {
        "v_thd_mu": 0.015, "i_thd_mu": 0.05, "transient_rate_per_day": 0.5,
        "inrush_rate_per_day": 1.0, "sag_rate_per_day": 0.05,
        "load_pu_mu": 0.4, "load_pu_sigma": 0.15,
    },
    "urban_residential": {
        "v_thd_mu": 0.035, "i_thd_mu": 0.12, "transient_rate_per_day": 4.0,
        "inrush_rate_per_day": 12.0, "sag_rate_per_day": 0.4,
        "load_pu_mu": 0.55, "load_pu_sigma": 0.20,
    },
    "ev_corridor": {
        "v_thd_mu": 0.05, "i_thd_mu": 0.22, "transient_rate_per_day": 30.0,
        "inrush_rate_per_day": 80.0, "sag_rate_per_day": 1.0,
        "load_pu_mu": 0.75, "load_pu_sigma": 0.30,
    },
    "industrial_welder": {
        "v_thd_mu": 0.07, "i_thd_mu": 0.30, "transient_rate_per_day": 120.0,
        "inrush_rate_per_day": 300.0, "sag_rate_per_day": 4.0,
        "load_pu_mu": 0.85, "load_pu_sigma": 0.25,
    },
}


@dataclass
class MeterEnvironment:
    """Realised draw of a single meter's environment + intrinsic strength."""
    env_name: str
    params: Dict[str, float]
    # Per-component intrinsic robustness multiplier. 1.0 = nominal; lower means
    # this unit is weaker than spec (production variation).
    robustness: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Latent degradation model (the "mismatched" ground truth).
#
# These constants are deliberately NOT the ones used by
# `pqdna.vectorised_damage`. See the module docstring.
# ---------------------------------------------------------------------------

# Per-channel gain and exponent of the latent damage dose, in FEATURE_NAMES
# order. Compare against `pqdna.vectorised_damage` to see the mismatch.
LATENT_DOSE_GAIN = np.array(
    [60.0, 1.0, 14.0, 26.0, 1.0, 8.0, 14.0, 7.0, 1.0, 800.0, 120.0, 1.0],
    dtype=np.float64,
)
LATENT_DOSE_EXP = np.array(
    [1.6, 2.6, 1.2, 1.8, 1.5, 3.1, 0.8, 0.9, 1.3, 0.9, 1.1, 1.15],
    dtype=np.float64,
)

# Weibull-like shape exponent of the dose-to-damage link, per component.
LATENT_SHAPE_M: Dict[str, float] = {
    "relay": 1.5,
    "smps": 1.7,
    "terminal_block": 1.4,
    "mcu": 1.6,
    "display": 1.3,
}

# Characteristic life expressed as years of nominal `urban_residential`
# exposure required to reach unit normalised damage (CHS = 1/e).
# Calibrated so a 15-year fleet run yields a ~13% cumulative failure rate,
# consistent with the ~0.6%/yr premature-failure rate reported for this class
# of equipment. Harsh environments dominate the failures.
LATENT_LIFE_URBAN_YEARS: Dict[str, float] = {
    "relay": 105.0,
    "smps": 120.0,
    "terminal_block": 112.0,
    "mcu": 125.0,
    "display": 135.0,
}

# Cross-component coupling: degrading `source` accelerates `target`.
# (target -> (source, kappa)). The CHS estimator models components
# independently and therefore cannot represent this term.
LATENT_COUPLING: Dict[str, Tuple[str, float]] = {
    "mcu": ("smps", 0.35),
    "display": ("smps", 0.20),
    "relay": ("terminal_block", 0.25),
    "terminal_block": ("relay", 0.15),
}

# Multiplicative sensor noise on the device-visible S vector (1 sigma).
DEFAULT_OBS_NOISE = 0.04


def latent_dose(features: np.ndarray) -> np.ndarray:
    """Per-day latent damage dose for each PQ channel.

    Deliberately uses different gains and exponents from the device-side
    :func:`pqdna.vectorised_damage`, so that no exact linear recovery of the
    latent process from S is possible.

    Parameters
    ----------
    features : np.ndarray
        Shape ``(..., N_FEATURES)`` in :data:`FEATURE_NAMES` order.

    Returns
    -------
    np.ndarray
        Same shape as ``features``, non-negative.
    """
    f = np.asarray(features, dtype=np.float64)
    x = f.copy()
    # Channels measured as deviation from nominal rather than absolute level.
    x[..., 0] = np.abs(x[..., 0] - 1.0)                    # v_rms
    x[..., 4] = np.maximum(x[..., 4] - 1.414, 0.0)         # i_crest excess
    x = np.maximum(x, 0.0)
    return LATENT_DOSE_GAIN * np.power(x, LATENT_DOSE_EXP)


def _component_weight_matrix() -> np.ndarray:
    """(n_components, N_FEATURES) physics-of-failure weights."""
    W = np.zeros((len(COMPONENTS), N_FEATURES), dtype=np.float64)
    for ci, c in enumerate(COMPONENTS):
        for name, w in FAILURE_MODES[c].items():
            W[ci, FEATURE_NAMES.index(name)] = w
    return W


def _reference_characteristic_life(comp_W: np.ndarray) -> np.ndarray:
    """Characteristic life L_ref per component, in latent damage units.

    Anchored to the mean daily dose of a nominal ``urban_residential`` meter so
    the constants in :data:`LATENT_LIFE_URBAN_YEARS` are interpretable as
    "years of nominal urban exposure to reach CHS = 1/e".
    """
    p = ENVIRONMENTS["urban_residential"]
    mean_feats = np.array(
        [
            1.0,                              # v_rms
            p["load_pu_mu"],                  # i_rms
            p["v_thd_mu"],                    # v_thd
            p["i_thd_mu"],                    # i_thd
            1.5 + p["i_thd_mu"],              # i_crest
            0.9,                              # transient_pu (typical daily max)
            p["sag_rate_per_day"] * 0.001,    # sag_dwell
            p["sag_rate_per_day"] * 0.0003,   # swell_dwell
            0.5,                              # flicker_pst
            0.0008,                           # freq_dev
            0.0,                              # unbalance
            p["inrush_rate_per_day"],         # inrush_count
        ],
        dtype=np.float64,
    )
    ref_daily = comp_W @ latent_dose(mean_feats)          # (n_comp,)
    years = np.array([LATENT_LIFE_URBAN_YEARS[c] for c in COMPONENTS], dtype=np.float64)
    return np.maximum(ref_daily, 1e-12) * 365.0 * years


def _latent_chs_trajectory(
    feats: np.ndarray,
    comp_W: np.ndarray,
    L_ref: np.ndarray,
    robustness: np.ndarray,
) -> np.ndarray:
    """Ground-truth CHS trajectory from the latent degradation process.

    Parameters
    ----------
    feats : np.ndarray
        ``(n_days, N_FEATURES)`` daily feature draws.
    comp_W : np.ndarray
        ``(n_comp, N_FEATURES)`` physics-of-failure weights.
    L_ref : np.ndarray
        ``(n_comp,)`` characteristic life in latent damage units.
    robustness : np.ndarray
        ``(n_comp,)`` per-unit manufacturing strength multiplier.

    Returns
    -------
    np.ndarray
        ``(n_days, n_comp)`` CHS in (0, 1], monotonically non-increasing.
    """
    dose_daily = latent_dose(feats)                         # (n_days, N_FEATURES)
    comp_dose = dose_daily @ comp_W.T                       # (n_days, n_comp)

    # Cumulative (Miner's rule) damage - monotone, unlike the device's EWMA.
    L = np.cumsum(comp_dose, axis=0)                        # (n_days, n_comp)
    u = L / (L_ref[None, :] * robustness[None, :])          # normalised damage

    m = np.array([LATENT_SHAPE_M[c] for c in COMPONENTS], dtype=np.float64)
    D = np.power(np.maximum(u, 0.0), m[None, :])            # (n_days, n_comp)

    # Cross-component coupling, applied on the normalised damage.
    D_coupled = D.copy()
    for ci, c in enumerate(COMPONENTS):
        if c in LATENT_COUPLING:
            src, kappa = LATENT_COUPLING[c]
            D_coupled[:, ci] += kappa * D[:, COMPONENTS.index(src)]

    return np.exp(-D_coupled).astype(np.float32)


def _matched_chs_trajectory(
    S_hist: np.ndarray,
    comp_W: np.ndarray,
    robustness: np.ndarray,
) -> np.ndarray:
    """Legacy model-matched ground truth. Retained for ablation only.

    This reproduces the original generator: CHS is an exactly linear function
    of the same S the estimator sees, using the same weight matrix the
    estimator fits. Accuracy measured against this target is close to
    tautological and must not be reported as a headline result.
    """
    damage = (S_hist @ comp_W.T) * 5.5e-3 / robustness[None, :]
    return np.clip(1.0 - damage, 0.0, 1.0).astype(np.float32)


@dataclass
class MeterTrace:
    """Outputs of one simulated meter."""
    meter_id: str
    env: MeterEnvironment
    times_days: np.ndarray
    S_history: np.ndarray             # (T, N_FEATURES)
    chs_history: np.ndarray           # (T, len(COMPONENTS))
    failed_component: Optional[str] = None
    failure_day: Optional[float] = None


class _AcceleratedSimulator:
    """Closed-form damage simulator that bypasses 12,000 hr of waveform.

    Rather than synthesising literal 1-second waveforms for years of meter
    life (computationally prohibitive), we draw the EXPECTED per-day damage
    dose for each PQ-DNA dimension given the environment, and use the same
    EWMA update rule the on-meter aggregator uses, but with a 1-day window.

    This keeps the math IDENTICAL to the on-meter aggregator (same alpha,
    same g(.)) while making 30-year fleet simulation tractable.
    """

    def __init__(self, agg_for_feature_order: PQDNAAggregator) -> None:
        # alpha_day = alpha_sec ** (seconds_per_day)
        self.alpha = agg_for_feature_order.alpha ** 86400.0

    def _draw_features_batch(
        self, env: MeterEnvironment, n_days: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Vectorised version: generate `n_days` rows of expected features.

        Output shape: (n_days, N_FEATURES) in the order of FEATURE_NAMES.
        """
        p = env.params
        v_rms = np.clip(rng.normal(1.0, 0.005, n_days), 0.95, 1.05).astype(np.float32)
        i_rms = np.clip(
            rng.normal(p["load_pu_mu"], p["load_pu_sigma"], n_days), 0.05, 1.5
        ).astype(np.float32)
        v_thd = np.clip(
            rng.normal(p["v_thd_mu"], p["v_thd_mu"] * 0.3, n_days), 0.005, 0.15
        ).astype(np.float32)
        i_thd = np.clip(
            rng.normal(p["i_thd_mu"], p["i_thd_mu"] * 0.3, n_days), 0.01, 0.50
        ).astype(np.float32)
        i_crest = np.clip(rng.normal(1.5 + i_thd, 0.1), 1.4, 4.0).astype(np.float32)

        # Worst transient per day from a Gumbel peak distribution.
        n_trans = rng.poisson(p["transient_rate_per_day"], n_days)
        transient = np.full(n_days, 0.05, dtype=np.float32)
        max_n = int(n_trans.max(initial=0))
        if max_n > 0:
            block = rng.gumbel(loc=0.5, scale=0.3, size=(n_days, max_n)).astype(np.float32)
            mask = np.arange(max_n)[None, :] < n_trans[:, None]
            block = np.where(mask, block, -np.inf)
            day_max = block.max(axis=1)
            day_max = np.where(np.isfinite(day_max), day_max, 0.05)
            transient = np.clip(day_max, 0.05, 4.0).astype(np.float32)

        sag_evt = rng.poisson(p["sag_rate_per_day"], n_days)
        sag_dwell = np.minimum(sag_evt * 0.001, 0.5).astype(np.float32)
        swell_dwell = np.minimum(
            rng.poisson(p["sag_rate_per_day"] * 0.3, n_days) * 0.001, 0.3
        ).astype(np.float32)
        flicker = np.clip(rng.lognormal(np.log(0.5), 0.5, n_days), 0.05, 5.0).astype(np.float32)
        freq_dev = np.clip(np.abs(rng.normal(0, 0.001, n_days)), 0.0, 0.02).astype(np.float32)
        unbalance = np.zeros(n_days, dtype=np.float32)
        inrush = rng.poisson(p["inrush_rate_per_day"], n_days).astype(np.float32)

        # Stack in the exact FEATURE_NAMES order.
        return np.stack(
            [v_rms, i_rms, v_thd, i_thd, i_crest, transient,
             sag_dwell, swell_dwell, flicker, freq_dev, unbalance, inrush],
            axis=1,
        )


def _true_chs_from_state(state: PQDNAState, robustness: Dict[str, float]) -> np.ndarray:
    """Legacy single-snapshot model-matched CHS. Deprecated.

    Retained so existing callers keep working. New code should use
    :meth:`SyntheticMeterFleet.simulate`, which drives the latent
    (``"mismatched"``) generator by default.
    """
    out = np.zeros(len(COMPONENTS), dtype=np.float32)
    for ci, c in enumerate(COMPONENTS):
        damage = 0.0
        for name, w in FAILURE_MODES[c].items():
            idx = FEATURE_NAMES.index(name)
            damage += w * float(state.S[idx])
        r = robustness.get(c, 1.0)
        out[ci] = float(np.clip(1.0 - (damage * 1.5e-3) / r, 0.0, 1.0))
    return out


class SyntheticMeterFleet:
    """Generates a fleet of N simulated meters with daily CHS snapshots."""

    def __init__(
        self,
        n_meters: int = 200,
        years: float = 12.0,
        seed: int = 7,
        env_mix: Optional[Dict[str, float]] = None,
        generator: str = "mismatched",
        obs_noise: float = DEFAULT_OBS_NOISE,
    ) -> None:
        """
        Parameters
        ----------
        generator : {"mismatched", "matched"}
            Ground-truth degradation model. ``"mismatched"`` (default) uses the
            latent process described in the module docstring. ``"matched"``
            reproduces the legacy linear generator and exists only so the
            model-matching artefact can be quantified.
        obs_noise : float
            Multiplicative 1-sigma sensor noise applied to the device-visible S
            vector. Set to 0.0 for a noiseless study.
        """
        if generator not in ("mismatched", "matched"):
            raise ValueError(f"generator must be 'mismatched' or 'matched', got {generator!r}")
        self.n_meters = n_meters
        self.years = years
        self.generator = generator
        self.obs_noise = float(obs_noise)
        self.rng_master = np.random.default_rng(seed)
        self.env_mix = env_mix or {
            "clean_grid": 0.20,
            "urban_residential": 0.45,
            "ev_corridor": 0.20,
            "industrial_welder": 0.15,
        }
        # one aggregator instance (carries alpha/tau);
        # we'll repeatedly clone PQDNAState.
        self.agg = PQDNAAggregator(dt_window_s=1.0)
        self._sim = _AcceleratedSimulator(self.agg)

    def _draw_env(self, rng: np.random.Generator) -> MeterEnvironment:
        names = list(self.env_mix.keys())
        probs = np.array([self.env_mix[n] for n in names], dtype=np.float64)
        probs /= probs.sum()
        env_name = rng.choice(names, p=probs)
        robustness = {
            c: float(np.clip(rng.normal(1.0, 0.12), 0.5, 1.5))
            for c in COMPONENTS
        }
        return MeterEnvironment(env_name=env_name, params=ENVIRONMENTS[env_name],
                                robustness=robustness)

    def simulate(self) -> List[MeterTrace]:
        traces: List[MeterTrace] = []
        days = int(self.years * 365)
        sim_dt_days = 7
        snap_idx = np.arange(0, days, sim_dt_days, dtype=np.int32)
        alpha_day = self._sim.alpha            # (N_FEATURES,)

        comp_W = _component_weight_matrix()
        L_ref = _reference_characteristic_life(comp_W)

        for m in range(self.n_meters):
            rng = np.random.default_rng(int(self.rng_master.integers(0, 2 ** 31)))
            env = self._draw_env(rng)
            rob = np.array([env.robustness[c] for c in COMPONENTS], dtype=np.float64)

            # All days' expected features at once.
            feats = self._sim._draw_features_batch(env, days, rng)   # (days, N_FEATURES)
            damage = vectorised_damage(feats)                        # (days, N_FEATURES)

            # Device-side EWMA. This is the PQ-DNA state the firmware holds.
            S = np.zeros(N_FEATURES, dtype=np.float32)
            S_hist = np.empty((days, N_FEATURES), dtype=np.float32)
            for d in range(days):
                S = alpha_day * S + (1.0 - alpha_day) * damage[d]
                S_hist[d] = S

            # Ground truth from a process the device does not observe directly.
            if self.generator == "mismatched":
                chs_full = _latent_chs_trajectory(feats, comp_W, L_ref, rob)
            else:
                chs_full = _matched_chs_trajectory(S_hist, comp_W, rob)

            S_snap = S_hist[snap_idx]                                # (n_snap, N_FEATURES)
            chs_snap = chs_full[snap_idx]

            # Sensor imperfection: the estimator never sees the exact S that
            # drove the latent process.
            if self.obs_noise > 0.0:
                noise = rng.normal(1.0, self.obs_noise, S_snap.shape).astype(np.float32)
                S_snap = (S_snap * np.maximum(noise, 0.0)).astype(np.float32)

            # Failure detection: first snapshot where any CHS < 0.05.
            failed_comp = None
            failure_day = None
            below = chs_snap < 0.05
            if below.any():
                # first row index where any component below threshold.
                row = int(np.argmax(below.any(axis=1)))
                ci = int(np.argmin(chs_snap[row]))
                failed_comp = COMPONENTS[ci]
                failure_day = float(snap_idx[row])
                # truncate history at +1 snapshot past failure for realism.
                trunc = min(len(snap_idx), row + 2)
                snap_used = snap_idx[:trunc]
                S_snap = S_snap[:trunc]
                chs_snap = chs_snap[:trunc]
            else:
                snap_used = snap_idx

            traces.append(MeterTrace(
                meter_id=f"m{m:04d}",
                env=env,
                times_days=snap_used.astype(np.float32),
                S_history=S_snap,
                chs_history=chs_snap,
                failed_component=failed_comp,
                failure_day=failure_day,
            ))
        return traces

    # ---- handy conversion helpers --------------------------------------
    @staticmethod
    def to_arrays(
        traces: Iterable[MeterTrace],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten all snapshots into ``(S, CHS, ttf_days, age_days)``.

        ``age_days`` is the device's own elapsed operating time, which is
        available on any real device for free (it already keeps a run-hour
        counter) and is required because the EWMA state S is a stress-*level*
        estimate that carries no information about accumulated exposure.
        """
        S_all, C_all, t_all, age_all = [], [], [], []
        for tr in traces:
            S_all.append(tr.S_history)
            C_all.append(tr.chs_history)
            t = tr.times_days
            if tr.failure_day is not None:
                ttf = tr.failure_day - t
            else:
                ttf = float(tr.times_days[-1]) - t   # censored
            t_all.append(ttf.astype(np.float32))
            age_all.append(t.astype(np.float32))
        return (
            np.concatenate(S_all, axis=0),
            np.concatenate(C_all, axis=0),
            np.concatenate(t_all, axis=0),
            np.concatenate(age_all, axis=0),
        )
