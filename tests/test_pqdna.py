# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from pqdna.chs import COMPONENTS, ComponentHealthScorer
from pqdna.edge_export import memory_footprint_bytes
from pqdna.features import FEATURE_NAMES, PQFeatureExtractor
from pqdna.pqdna import PQDNAAggregator, PQDNAState, vectorised_damage
from pqdna.rul import RULEstimator
from pqdna.synth import LATENT_DOSE_EXP, SyntheticMeterFleet, latent_dose


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
