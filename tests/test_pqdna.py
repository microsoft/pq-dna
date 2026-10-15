# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from pqdna.chs import COMPONENTS, ComponentHealthScorer
from pqdna.edge_export import memory_footprint_bytes
from pqdna.features import FEATURE_NAMES, PQFeatureExtractor
from pqdna.pqdna import (
    DEFAULT_TAU_S,
    PQDNAAggregator,
    PQDNAState,
    _damage_nonlinearity,
    vectorised_damage,
)
from pqdna.rul import RULEstimator
from pqdna.synth import LATENT_DOSE_EXP, SyntheticMeterFleet, _AcceleratedSimulator, latent_dose


def _clean_sine(fs=7680, line=60.0, dur_s=1.0, vp=325.0, ip=42.0):
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    v = vp * np.sin(2 * np.pi * line * t)
    i = ip * np.sin(2 * np.pi * line * t)
    return v, i


def test_feature_extractor_shapes_and_ranges():
    ext = PQFeatureExtractor()
    v, i = _clean_sine()
    f = ext.extract(v, i)
    vec = f.as_vector()
    assert vec.shape == (len(FEATURE_NAMES),)
    # clean sine -> THD ~ 0, crest ~ sqrt(2), v_rms ~ 1 pu
    assert f.v_thd < 0.05
    assert abs(f.i_crest - 2 ** 0.5) < 0.05
    assert 0.95 < f.v_rms < 1.05


def test_pqdna_aggregator_accumulates_and_decays():
    ext = PQFeatureExtractor()
    agg = PQDNAAggregator(dt_window_s=1.0)
    v, i = _clean_sine()
    feats = ext.extract(v, i)
    s = PQDNAState()
    for _ in range(10):
        agg.update(s, feats)
    # after many updates, state should be positive and bounded
    assert np.all(s.S >= 0)
    assert float(np.linalg.norm(s.S)) > 0


def test_chs_in_unit_interval():
    scorer = ComponentHealthScorer()
    s = PQDNAState()
    # nominal stress -> CHS ~ 1
    out = scorer.score(s)
    for c in COMPONENTS:
        assert 0.0 <= out[c] <= 1.0
        assert out[c] > 0.95


def test_fleet_simulation_minimal():
    fleet = SyntheticMeterFleet(n_meters=10, years=3.0, seed=1)
    traces = fleet.simulate()
    assert len(traces) == 10
    S, CHS, ttf, age = SyntheticMeterFleet.to_arrays(traces)
    assert S.shape[1] == len(FEATURE_NAMES)
    assert CHS.shape[1] == len(COMPONENTS)
    assert len(S) == len(CHS) == len(ttf) == len(age)
    assert np.all(age >= 0)


def test_chs_fit_improves_mae():
    fleet = SyntheticMeterFleet(n_meters=60, years=8.0, seed=2)
    traces = fleet.simulate()
    S, CHS, _, age = SyntheticMeterFleet.to_arrays(traces)
    age_y = age / 365.0

    scorer_default = ComponentHealthScorer()
    mae_default = float(np.mean(np.abs(scorer_default.score_batch(S, age_y) - CHS)))

    scorer_fit = ComponentHealthScorer().fit(S, CHS, age_y)
    mae_fit = float(np.mean(np.abs(scorer_fit.score_batch(S, age_y) - CHS)))

    assert mae_fit <= mae_default


def test_rul_runs():
    fleet = SyntheticMeterFleet(n_meters=40, years=6.0, seed=3)
    traces = fleet.simulate()
    S, CHS, ttf, age = SyntheticMeterFleet.to_arrays(traces)
    rul = RULEstimator(model="ridge", l2=1.0).fit(S, CHS, ttf, age / 365.0)
    pred = rul.predict_days(S, CHS, age / 365.0)
    assert pred.shape == ttf.shape
    assert np.all(pred >= 0)


def test_memory_footprint_small():
    agg = PQDNAAggregator(dt_window_s=1.0)
    fleet = SyntheticMeterFleet(n_meters=20, years=4.0, seed=4)
    traces = fleet.simulate()
    S, CHS, ttf, age = SyntheticMeterFleet.to_arrays(traces)
    age_y = age / 365.0
    scorer = ComponentHealthScorer().fit(S, CHS, age_y)
    rul = RULEstimator(model="ridge", l2=1.0).fit(S, CHS, ttf, age_y)
    foot = memory_footprint_bytes(agg, scorer, rul)
    assert foot["total_bytes"] < 1024
    assert foot["age_counter_bytes"] == 4


# ---------------------------------------------------------------------------
# Regression guards for the validation-integrity fixes.
# ---------------------------------------------------------------------------


def test_latent_generator_is_not_the_device_kernel():
    """The ground-truth dose must not be the device's own damage kernel.

    If these ever coincide, the CHS evaluation degenerates into a
    self-consistency check rather than a test. This guard exists to stop that
    regression from silently reappearing.
    """
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(0.5, 0.2, (64, len(FEATURE_NAMES)))).astype(np.float32)
    device = vectorised_damage(x)
    latent = latent_dose(x)
    assert device.shape == latent.shape
    assert not np.allclose(device, latent, rtol=0.05)
    assert not np.allclose(LATENT_DOSE_EXP, np.ones_like(LATENT_DOSE_EXP))


def test_ground_truth_is_monotone_and_bounded():
    """Latent health must decrease monotonically and stay in [0, 1]."""
    fleet = SyntheticMeterFleet(n_meters=8, years=10.0, seed=11)
    for tr in fleet.simulate():
        chs = tr.chs_history
        assert np.all(chs >= 0.0) and np.all(chs <= 1.0)
        assert np.all(np.diff(chs, axis=0) <= 1e-6)


def test_age_feature_is_necessary():
    """S alone cannot identify cumulative damage; S x age must do better.

    This encodes the core methodological finding: the EWMA state is a
    stress-*level* estimate that reaches steady state, so it carries no
    exposure-duration information.
    """
    fleet = SyntheticMeterFleet(n_meters=80, years=12.0, seed=5)
    traces = fleet.simulate()
    S, CHS, _, age = SyntheticMeterFleet.to_arrays(traces)
    age_y = age / 365.0

    s_only = ComponentHealthScorer(use_age=False).fit(S, CHS)
    mae_s_only = float(np.mean(np.abs(s_only.score_batch(S) - CHS)))

    with_age = ComponentHealthScorer(use_age=True).fit(S, CHS, age_y)
    mae_with_age = float(np.mean(np.abs(with_age.score_batch(S, age_y) - CHS)))

    assert mae_with_age < mae_s_only


def test_matched_generator_is_easier_than_mismatched():
    """Quantifies the model-matching artefact the original evaluation had.

    The legacy 'matched' generator produces CHS as an exactly linear function
    of the same S the estimator fits, so it must be substantially easier.
    """
    def mae_for(generator):
        fleet = SyntheticMeterFleet(n_meters=60, years=10.0, seed=9,
                                    generator=generator)
        traces = fleet.simulate()
        S, CHS, _, age = SyntheticMeterFleet.to_arrays(traces)
        age_y = age / 365.0
        sc = ComponentHealthScorer(use_age=True).fit(S, CHS, age_y)
        return float(np.mean(np.abs(sc.score_batch(S, age_y) - CHS)))

    assert mae_for("matched") < mae_for("mismatched")


def test_observation_noise_is_applied():
    """The estimator must not see the exact S that drove the latent process."""
    clean = SyntheticMeterFleet(n_meters=4, years=5.0, seed=3, obs_noise=0.0).simulate()
    noisy = SyntheticMeterFleet(n_meters=4, years=5.0, seed=3, obs_noise=0.10).simulate()
    assert not np.allclose(clean[0].S_history, noisy[0].S_history)


# ---------------------------------------------------------------------------
# Regression guards for the time constants: the 1 s decay of a 60-day channel
# is 1 - 1.9e-7, which float32 cannot represent.
# ---------------------------------------------------------------------------


def _tau_s():
    return np.array([DEFAULT_TAU_S[n] for n in FEATURE_NAMES], dtype=np.float64)


def test_aggregator_gain_is_exact_at_one_second():
    agg = PQDNAAggregator(dt_window_s=1.0)
    x = 1.0 / _tau_s()
    # 1 - exp(-x) by its Taylor series, independent of expm1; x <= 1.2e-5.
    series = x - x ** 2 / 2 + x ** 3 / 6
    assert agg.alpha.dtype == np.float64 and agg.gain.dtype == np.float64
    np.testing.assert_allclose(agg.gain, series, rtol=1e-12)
    np.testing.assert_allclose(agg.alpha + agg.gain, 1.0, rtol=0, atol=1e-15)
    # the effective time constant of the 1 s recursion is the nominal one
    np.testing.assert_allclose(-1.0 / np.log(agg.alpha), _tau_s(), rtol=1e-6)


def test_aggregator_update_matches_closed_form_and_keeps_dtype():
    agg = PQDNAAggregator(dt_window_s=1.0)
    v, i = _clean_sine()
    feats = PQFeatureExtractor().extract(v, i)
    # the aggregator's own (float64) kernel, so that only the update rule is tested
    g = np.array(
        [_damage_nonlinearity(n, float(x)) for n, x in zip(FEATURE_NAMES, feats.as_vector())]
    )
    state = PQDNAState()
    assert state.S.dtype == np.float64
    n = 500
    for _ in range(n):
        agg.update(state, feats)
    expected = -np.expm1(-n * agg.dt / agg.tau) * g
    np.testing.assert_allclose(state.S, expected, rtol=1e-5, atol=1e-12)
    assert state.n_updates == n and state.t_last == n * agg.dt

    # a caller-supplied float32 state stays float32
    f32 = agg.update(PQDNAState(S=np.zeros(len(FEATURE_NAMES), dtype=np.float32)), feats)
    assert f32.S.dtype == np.float32


def test_simulator_daily_decay_is_the_exact_one_day_decay():
    agg = PQDNAAggregator(dt_window_s=1.0)
    alpha_day = _AcceleratedSimulator(agg).alpha
    # every channel has its nominal time constant: 1, 7, 30 and 60 days
    np.testing.assert_allclose(-1.0 / np.log(alpha_day), _tau_s() / 86400.0, rtol=1e-12)

    # and it equals 86,400 one-second updates with a constant input
    g = np.linspace(0.5, 3.0, len(FEATURE_NAMES))
    S = np.zeros_like(g)
    for _ in range(86400):
        S += agg.gain * (g - S)
    np.testing.assert_allclose(S, (1.0 - alpha_day) * g, rtol=1e-6)
