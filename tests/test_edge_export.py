"""Fixed-point export: formats, fidelity to the float model and the generated C header."""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import shlex
import shutil
import subprocess

import numpy as np
import pytest

from pqdna.chs import ComponentHealthScorer
from pqdna.edge_export import FixedPointModel, export_c_header, memory_footprint_bytes
from pqdna.features import FEATURE_NAMES
from pqdna.pqdna import PQDNAAggregator
from pqdna.rul import RULEstimator
from pqdna.synth import SyntheticMeterFleet

N_FEATURES = len(FEATURE_NAMES)
N_COMPONENTS = 5
ONE = 1 << 16  # 1.0 in Q15.16


@pytest.fixture(scope="module")
def fitted():
    fleet = SyntheticMeterFleet(n_meters=40, years=8.0, seed=21)
    S, CHS, ttf, age_days = SyntheticMeterFleet.to_arrays(fleet.simulate())
    age_y = age_days / 365.0
    return {
        "agg": PQDNAAggregator(dt_window_s=1.0),
        "scorer": ComponentHealthScorer().fit(S, CHS, age_y),
        "rul": RULEstimator(model="ridge", l2=10.0).fit(S, CHS, ttf, age_y),
        "S": S.astype(np.float64),
        "CHS": CHS,
        "ttf": ttf,
        "age_days": np.rint(age_days).astype(np.int64),
    }


def _model(fitted):
    return FixedPointModel.from_models(fitted["agg"], fitted["scorer"], fitted["rul"])


def test_gains_give_the_nominal_time_constants(fitted):
    agg = fitted["agg"]
    gain = np.array(_model(fitted).gain_q32) / 2.0 ** 32
    # the old Q15 alpha rounded every channel to 32767, i.e. tau of about 9 hours
    np.testing.assert_allclose(-agg.dt / np.log1p(-gain), agg.tau, rtol=1e-3)


def test_mantissas_fit_their_range_and_reproduce_the_weights(fitted):
    model = _model(fitted)
    limit = 2 ** 23
    W = fitted["scorer"].W.astype(np.float64) * fitted["scorer"].scale
    for c, (row, e) in enumerate(zip(model.chs_w_q23, model.chs_w_exp)):
        peak = max(abs(m) for m in row)
        assert limit // 2 <= peak <= limit  # the smallest exponent that fits
        back = np.ldexp(np.array(row, dtype=np.float64), e - 23)
        assert np.max(np.abs(back - W[c])) <= 2.0 ** (e - 24) * (1 + 1e-12)
    assert limit // 2 <= max(abs(m) for m in model.rul_coefs_q23) <= limit


def test_integer_chs_and_rul_match_the_float_model(fitted):
    model = _model(fitted)
    scorer, rul = fitted["scorer"], fitted["rul"]
    S, age_days = fitted["S"], fitted["age_days"]
    idx = np.arange(0, len(S), 4)
    chs_f = scorer.score_batch(S[idx], age_days[idx] / 365.0)
    rul_f = rul.predict_days(S[idx], chs_f, age_days[idx] / 365.0)
    worst_chs, worst_rul = 0.0, 0.0
    for r, i in enumerate(idx):
        S_q = model.state_to_q(S[i])
        chs_q = model.chs_q16(S_q, int(age_days[i]))
        worst_chs = max(worst_chs, float(np.max(np.abs(np.array(chs_q) / ONE - chs_f[r]))))
        rul_q = model.rul_days(S_q, chs_q, int(age_days[i]))
        worst_rul = max(worst_rul, abs(rul_q - float(rul_f[r])))
    assert worst_chs < 1e-4
    assert worst_rul < 2.0


def test_update_matches_the_float_recursion():
    agg = PQDNAAggregator(dt_window_s=1.0)
    model = FixedPointModel.from_models(agg, ComponentHealthScorer(), RULEstimator())
    g = np.linspace(0.2, 40.0, N_FEATURES)
    g_q = model.state_to_q(g)
    S_q = [0] * N_FEATURES
    steps = 4000
    for _ in range(steps):
        S_q = model.update(S_q, g_q)
    expected = -np.expm1(-steps * agg.dt / agg.tau) * g
    # dominated by the 0.06% quantisation of the gain of the 60-day channels
    np.testing.assert_allclose(model.state_from_q(S_q), expected, rtol=1e-3)


def test_update_step_is_the_rounded_product():
    model = FixedPointModel.from_models(
        PQDNAAggregator(), ComponentHealthScorer(), RULEstimator()
    )
    rng = np.random.default_rng(3)
    for _ in range(200):
        S_q = [int(x) for x in rng.integers(0, 2 ** 62, N_FEATURES)]
        g_q = [int(x) for x in rng.integers(0, 2 ** 62, N_FEATURES)]
        want = [
            s + ((gain * (g - s) + (1 << 31)) >> 32)
            for gain, s, g in zip(model.gain_q32, S_q, g_q)
        ]
        assert model.update(S_q, g_q) == want


def test_reference_rejects_inputs_outside_its_ranges():
    model = FixedPointModel.from_models(
        PQDNAAggregator(), ComponentHealthScorer(), RULEstimator()
    )
    zeros = [0] * N_FEATURES
    with pytest.raises(ValueError):
        model.update([-1] + zeros[1:], zeros)
    with pytest.raises(ValueError):
        model.update(zeros, [2 ** 63] + zeros[1:])
    with pytest.raises(ValueError):
        model.chs_q16(zeros, 2 ** 32)
    with pytest.raises(ValueError):
        model.rul_days(zeros, [ONE] * N_COMPONENTS, 0)  # no ridge head
    with pytest.raises(ValueError):
        FixedPointModel.state_to_q([-0.5])


def test_rul_with_small_coefficients_is_scaled_correctly():
    """Regression: the old export divided by max(1, max|beta|) but wrote max|beta| as the scale."""
    rul = RULEstimator(model="ridge")
    rul.coefs_ = np.array([0.5] * 12 + [0.25] * 5 + [0.1, 0.3], dtype=np.float32)
    model = FixedPointModel.from_models(PQDNAAggregator(), ComponentHealthScorer(), rul)
    S = np.full(N_FEATURES, 10.0)
    want = float(rul.predict_days(S[None, :], np.ones((1, N_COMPONENTS)), np.array([2.0]))[0])
    got = model.rul_days(model.state_to_q(S), [ONE] * N_COMPONENTS, 2 * 365)
    assert abs(want - 61.75) < 1e-4
    assert abs(got - want) <= 0.5 + 1e-3


def test_unrepresentable_models_raise_instead_of_saturating(fitted):
    agg, scorer, rul = fitted["agg"], fitted["scorer"], fitted["rul"]

    too_fast = PQDNAAggregator(dt_window_s=1000.0, tau_s={n: 1.0 for n in FEATURE_NAMES})
    with pytest.raises(ValueError, match="Q0.32"):
        FixedPointModel.from_models(too_fast, scorer, rul)
    too_slow = PQDNAAggregator(dt_window_s=1.0, tau_s={n: 1e12 for n in FEATURE_NAMES})
    with pytest.raises(ValueError, match="Q0.32"):
        FixedPointModel.from_models(too_slow, scorer, rul)

    nan_W = scorer.W.copy()
    nan_W[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        FixedPointModel.from_models(agg, ComponentHealthScorer(W=nan_W, bias=scorer.bias), rul)

    huge_W = scorer.W * np.float32(1e30)
    with pytest.raises(ValueError, match="exponent"):
        FixedPointModel.from_models(agg, ComponentHealthScorer(W=huge_W, bias=scorer.bias), rul)

    bad_rul = RULEstimator(model="ridge")
    bad_rul.coefs_ = np.ones(7, dtype=np.float32)
    with pytest.raises(ValueError, match="RUL coefficients"):
        FixedPointModel.from_models(agg, scorer, bad_rul)


def _parse_ints(text, name):
    pattern = rf"static const \w+ {name}\[[^\]]*\](?:\[[^\]]*\])? = \{{(.*?)\n\}};"
    return [int(x) for x in re.findall(r"-?\d+", re.search(pattern, text, re.S).group(1))]


def test_header_holds_the_quantised_constants(tmp_path, fitted):
    model = _model(fitted)
    path = export_c_header(
        str(tmp_path / "nested" / "pqdna_model.h"), fitted["agg"], fitted["scorer"], fitted["rul"]
    )
    text = path.read_bytes().decode("utf-8")
    assert "\r" not in text and text.endswith("\n")
    assert text.startswith(
        "/* Copyright (c) Microsoft Corporation.\n * Licensed under the MIT License.\n"
    )
    for symbol in ("pqdna_update", "pqdna_chs", "pqdna_rul_days", "#define PQDNA_HAS_RUL 1"):
        assert symbol in text
    for gone in ("PQDNA_ALPHA_Q15", "PQDNA_CHS_W_Q15", "PQDNA_CHS_SCALE", "PQDNA_RUL_SCALE"):
        assert gone not in text
    assert _parse_ints(text, "PQDNA_GAIN_Q32") == list(model.gain_q32)
    assert _parse_ints(text, "PQDNA_CHS_W_Q23") == [m for row in model.chs_w_q23 for m in row]
    assert _parse_ints(text, "PQDNA_CHS_W_EXP") == list(model.chs_w_exp)
    assert _parse_ints(text, "PQDNA_CHS_BIAS_Q16") == list(model.chs_bias_q16)
    assert _parse_ints(text, "PQDNA_RUL_COEFS_Q23") == list(model.rul_coefs_q23)
    assert f"PQDNA_RUL_EXP = {model.rul_exp};" in text


def test_header_without_a_ridge_head(tmp_path, fitted):
    path = export_c_header(
        str(tmp_path / "pqdna_model.h"), fitted["agg"], fitted["scorer"], RULEstimator(model="gbr")
    )
    text = path.read_text(encoding="utf-8")
    assert "#define PQDNA_HAS_RUL 0" in text
    assert "pqdna_rul_days" not in text and "PQDNA_RUL_COEFS_Q23" not in text


def test_footprint_matches_the_exported_constants(fitted):
    model = _model(fitted)
    expected = {
        "S_state_bytes": 8 * N_FEATURES,
        "gain_bytes": 4 * len(model.gain_q32),
        "age_counter_bytes": 4,
        "chs_W_bytes": 4 * sum(len(row) for row in model.chs_w_q23),
        "chs_W_exp_bytes": len(model.chs_w_exp),
        "chs_bias_bytes": 4 * len(model.chs_bias_q16),
        "rul_coefs_bytes": 4 * len(model.rul_coefs_q23),
        "rul_exp_bytes": 1,
    }
    expected["total_bytes"] = sum(expected.values())
    foot = memory_footprint_bytes(fitted["agg"], fitted["scorer"], fitted["rul"])
    assert foot == expected
    assert foot["total_bytes"] == 730


# ---------------------------------------------------------------------------
# The C reference code in the header against the Python reference, bit for bit.
# Runs when a C compiler is on the PATH or named in the CC environment variable.
# ---------------------------------------------------------------------------

_C_DRIVER = r"""
#include <inttypes.h>
#include <stdio.h>

#include "pqdna_model.h"

int main(void)
{
    int64_t S[PQDNA_N_FEATURES];
    int64_t g[PQDNA_N_FEATURES];
    int32_t chs[PQDNA_N_COMPONENTS];
    uint32_t days;
    int n, k, c;

    if (scanf("%d", &n) != 1) {
        return 2;
    }
    while (n-- > 0) {
        for (k = 0; k < PQDNA_N_FEATURES; k++) {
            if (scanf("%" SCNd64, &S[k]) != 1) {
                return 2;
            }
        }
        for (k = 0; k < PQDNA_N_FEATURES; k++) {
            if (scanf("%" SCNd64, &g[k]) != 1) {
                return 2;
            }
        }
        if (scanf("%" SCNu32, &days) != 1) {
            return 2;
        }
        pqdna_update(S, g);
        pqdna_chs(S, days, chs);
        for (k = 0; k < PQDNA_N_FEATURES; k++) {
            printf("%" PRId64 " ", S[k]);
        }
        for (c = 0; c < PQDNA_N_COMPONENTS; c++) {
            printf("%" PRId32 " ", chs[c]);
        }
#if PQDNA_HAS_RUL
        printf("%" PRId32 "\n", pqdna_rul_days(S, chs, days));
#else
        printf("-1\n");
#endif
    }
    return 0;
}
"""


def _find_c_compiler():
    env = os.environ.get("CC")
    if env:
        return shlex.split(env)
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return [path]
    return None


def _golden_cases(n, seed):
    rng = np.random.default_rng(seed)
    top = 2 ** 63 - 1

    def q48(lo, hi):
        return min(int(10 ** rng.uniform(lo, hi) * 2 ** 48), top)

    cases = []
    for _ in range(n):
        S = [q48(-4, 2.7) for _ in range(N_FEATURES)]
        g = [0 if rng.random() < 0.05 else q48(-4, 2.7) for _ in range(N_FEATURES)]
        days = int(rng.choice([0, 1, 364, 365, 5475, 20000, int(rng.integers(0, 6000))]))
        cases.append((S, g, days))
    zeros = [0] * N_FEATURES
    cases += [
        (zeros, zeros, 0),
        ([top] * N_FEATURES, zeros, 2 ** 32 - 1),
        (zeros, [top] * N_FEATURES, 2 ** 32 - 1),
        ([top] * N_FEATURES, [top] * N_FEATURES, 123),
    ]
    return cases


@pytest.mark.skipif(_find_c_compiler() is None, reason="no C compiler (set CC to enable)")
@pytest.mark.parametrize("variant", ["full", "no_age", "no_rul"])
def test_c_reference_matches_the_python_reference(tmp_path, fitted, variant):
    agg = fitted["agg"]
    if variant == "full":
        scorer, rul = fitted["scorer"], fitted["rul"]
    elif variant == "no_age":
        scorer = ComponentHealthScorer(use_age=False).fit(fitted["S"], fitted["CHS"])
        rul = RULEstimator(model="ridge", l2=10.0).fit(fitted["S"], fitted["CHS"], fitted["ttf"])
    else:
        scorer, rul = fitted["scorer"], RULEstimator(model="gbr")
    model = FixedPointModel.from_models(agg, scorer, rul)
    export_c_header(str(tmp_path / "pqdna_model.h"), agg, scorer, rul)
    (tmp_path / "driver.c").write_text(_C_DRIVER, encoding="utf-8")
    exe = tmp_path / ("driver.exe" if os.name == "nt" else "driver")
    cmd = _find_c_compiler() + [
        "-std=c99", "-Wall", "-Wextra", "-pedantic",
        "-I", str(tmp_path), str(tmp_path / "driver.c"), "-o", str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    cases = _golden_cases(300, seed=5)
    stdin = f"{len(cases)}\n" + "\n".join(
        " ".join(str(v) for v in S + g + [days]) for S, g, days in cases
    ) + "\n"
    run = subprocess.run([str(exe)], input=stdin, check=True, capture_output=True, text=True)
    lines = run.stdout.strip().splitlines()
    assert len(lines) == len(cases)
    for (S_q, g_q, days), line in zip(cases, lines):
        S_new = model.update(S_q, g_q)
        chs = model.chs_q16(S_new, days)
        rul_days = -1 if rul.coefs_ is None else model.rul_days(S_new, chs, days)
        assert [int(x) for x in line.split()] == S_new + chs + [rul_days]
