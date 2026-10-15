"""Fixed-point export of the PQ-DNA model, with a bit-exact integer reference.

`FixedPointModel.from_models` quantises the decay gains, the CHS weights and bias and
the ridge RUL coefficients. `export_c_header` writes them, together with the
fixed-point formats and a small C99 reference implementation (`pqdna_update`,
`pqdna_chs`, `pqdna_rul_days`), to a header that a meter firmware build can include.
`FixedPointModel` evaluates the same integer arithmetic in Python, bit for bit, so its
outputs are golden vectors for a firmware port and its difference from the float model
is the quantisation error.

Formats (Qm.n is a two's-complement integer holding ``x * 2**n``):

    S, g        int64   Q15.48   PQ-DNA state and per-update kernel output, both >= 0
    working     int32   Q15.16   S, CHS and age inside the CHS and RUL stages
    gain        uint32  Q0.32    1 - exp(-dt / tau) of each channel
    W, b, beta  int32   W and beta as a mantissa m with |m| <= 2**23 and one power-of-two
                        exponent E per CHS row and one for the RUL vector (value =
                        m * 2**(E - 23)); b is Q15.16

A 1 s update of a 60-day channel has a gain of 1.9e-7: a 16-bit alpha cannot resolve it
(every alpha saturates at the Q15 maximum) and neither can a 32-bit state, hence the
32-bit gain and the 64-bit state. The fitted weights and coefficients span several
decades, hence the per-block exponents. The scale ``s`` of the CHS design vector is
folded into the weights. Nothing saturates silently: `FixedPointModel.from_models`
raises ValueError for a model that does not fit. The arithmetic itself is specified in
the comment at the top of the generated header and in docs/METHOD.md, Section 3.4.

`memory_footprint_bytes` counts the constants and the persistent state (not the
diagnostic name strings of the header).
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .chs import COMPONENTS, ComponentHealthScorer
from .features import FEATURE_NAMES, N_FEATURES
from .pqdna import PQDNAAggregator
from .rul import RULEstimator

STATE_FRAC_BITS = 48     # S and g: int64, Q15.48
WORK_FRAC_BITS = 16      # S, CHS and age inside the CHS and RUL stages: int32, Q15.16
GAIN_BITS = 32           # decay gains: uint32, Q0.32
MANTISSA_BITS = 23       # coefficient mantissas: |m| <= 2**23, held in an int32
DAYS_PER_YEAR = 365      # age in years = run_days / 365

_INT64_MAX = (1 << 63) - 1
_INT32_MAX = (1 << 31) - 1
_INT32_MIN = -(1 << 31)
_UINT32_MAX = (1 << 32) - 1


def _block_exponent(max_abs: float) -> int:
    """Smallest integer ``e`` with ``max_abs < 2**e`` (0 for an all-zero block)."""
    return 0 if max_abs == 0.0 else math.frexp(max_abs)[1]


def _check_shift(shift: int, what: str) -> None:
    if not 1 <= shift <= 62:
        raise ValueError(f"{what}: the block exponent is outside the supported range")


def _quantise_block(values: np.ndarray, what: str) -> Tuple[Tuple[int, ...], int]:
    """Quantise a block to int mantissas ``m`` with ``|m| <= 2**23`` and one exponent ``E``.

    ``value = m * 2**(E - 23)`` where ``E`` is the smallest integer with
    ``max|value| < 2**E``, so the mantissa rounding never needs to saturate.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{what} contains a non-finite value")
    e = _block_exponent(float(np.max(np.abs(v))) if v.size else 0.0)
    m = np.rint(np.ldexp(v, MANTISSA_BITS - e)).astype(np.int64)
    return tuple(int(x) for x in m), e


def _quantise_gains(aggregator: PQDNAAggregator) -> Tuple[int, ...]:
    """``round(2**32 * (1 - exp(-dt / tau)))`` per channel, as exact uint32 values."""
    q = np.rint(aggregator.gain * 2.0 ** GAIN_BITS)
    if not np.all((q >= 1) & (q <= _UINT32_MAX)):
        raise ValueError(
            "a decay gain does not fit in Q0.32: dt / tau must lie between 1.2e-10 and 22"
        )
    return tuple(int(x) for x in q)


def _sat32(v: int) -> int:
    return max(_INT32_MIN, min(_INT32_MAX, v))


def _round_shift(x: int, n: int) -> int:
    """``x / 2**n`` rounded half up (the C code needs an arithmetic right shift)."""
    return (x + (1 << (n - 1))) >> n


def _i64(v: int) -> int:
    """Return ``v`` if the C code's int64_t holds it, else raise OverflowError."""
    if not -(1 << 63) <= v <= _INT64_MAX:
        raise OverflowError("int64 overflow in the fixed-point arithmetic")
    return v


def _check_range(values: Sequence[int], lo: int, hi: int, what: str) -> None:
    for v in values:
        if not lo <= v <= hi:
            raise ValueError(f"{what} must lie in [{lo}, {hi}], got {v}")


@dataclass(frozen=True)
class FixedPointModel:
    """Quantised model constants and the integer arithmetic of the generated header.

    Build it with `from_models`. All fields are plain Python ints, so the methods
    evaluate exactly what the C code evaluates on a two's-complement machine. They
    raise OverflowError where the C code would overflow an ``int64_t`` and ValueError
    for an input outside the documented ranges.

    Attributes
    ----------
    dt_s : float
        Update interval, in seconds, the gains were computed for.
    gain_q32 : tuple of int
        Per-channel ``round(2**32 * (1 - exp(-dt / tau)))``.
    chs_use_age : bool
        Whether the CHS design vector has the ``age * S`` block (24 inputs, else 12).
    chs_w_q23, chs_w_exp : tuple
        CHS weight mantissas (one row per component, with the scale ``s`` of the float
        model folded in) and the exponent of each row.
    chs_bias_q16 : tuple of int
        CHS bias in Q15.16.
    rul_coefs_q23, rul_use_age, rul_exp
        Ridge RUL coefficient mantissas ordered ``[S, CHS, (age,) intercept]``,
        whether they include the age term, and their exponent. ``rul_coefs_q23`` is
        None when there is no fitted ridge head.
    """

    dt_s: float
    gain_q32: Tuple[int, ...]
    chs_use_age: bool
    chs_w_q23: Tuple[Tuple[int, ...], ...]
    chs_w_exp: Tuple[int, ...]
    chs_bias_q16: Tuple[int, ...]
    rul_coefs_q23: Optional[Tuple[int, ...]] = None
    rul_use_age: bool = False
    rul_exp: int = 0

    @classmethod
    def from_models(
        cls,
        aggregator: PQDNAAggregator,
        chs: ComponentHealthScorer,
        rul: RULEstimator,
    ) -> FixedPointModel:
        """Quantise the gains of ``aggregator``, the weights of ``chs`` and the ridge ``rul``.

        Raises ValueError if a value cannot be represented, instead of saturating.
        """
        n_comp = len(COMPONENTS)
        n_in = N_FEATURES * (2 if chs.use_age else 1)
        W = np.asarray(chs.W, dtype=np.float64)
        bias = np.asarray(chs.bias, dtype=np.float64)
        if W.shape != (n_comp, n_in) or bias.shape != (n_comp,):
            raise ValueError(
                f"CHS weights {W.shape} and bias {bias.shape} do not match "
                f"{n_comp} components x {n_in} inputs"
            )

        # The scale s of phi = s * [S, age * S] is folded into the weights.
        W_eff = W * float(chs.scale)
        w_rows, w_exps = [], []
        for c, name in enumerate(COMPONENTS):
            mantissas, e = _quantise_block(W_eff[c], f"CHS weights of {name}")
            _check_shift(MANTISSA_BITS - e, f"CHS weights of {name}")
            w_rows.append(mantissas)
            w_exps.append(e)

        if not np.all(np.isfinite(bias)):
            raise ValueError("CHS bias contains a non-finite value")
        bias_q = [int(x) for x in np.rint(bias * 2.0 ** WORK_FRAC_BITS)]
        _check_range(bias_q, _INT32_MIN, _INT32_MAX, "the quantised CHS bias")

        rul_q: Optional[Tuple[int, ...]] = None
        rul_age, rul_e = False, 0
        if rul.model == "ridge" and rul.coefs_ is not None:
            coefs = np.asarray(rul.coefs_, dtype=np.float64)
            n_x = N_FEATURES + n_comp
            if len(coefs) not in (n_x + 1, n_x + 2):
                raise ValueError(
                    f"expected {n_x + 1} or {n_x + 2} RUL coefficients, got {len(coefs)}"
                )
            rul_age = len(coefs) == n_x + 2
            rul_q, rul_e = _quantise_block(coefs, "RUL coefficients")
            _check_shift(MANTISSA_BITS + WORK_FRAC_BITS - rul_e, "RUL coefficients")

        return cls(
            dt_s=float(aggregator.dt),
            gain_q32=_quantise_gains(aggregator),
            chs_use_age=bool(chs.use_age),
            chs_w_q23=tuple(w_rows),
            chs_w_exp=tuple(w_exps),
            chs_bias_q16=tuple(bias_q),
            rul_coefs_q23=rul_q,
            rul_use_age=rul_age,
            rul_exp=rul_e,
        )

    @staticmethod
    def state_to_q(values: Sequence[float]) -> List[int]:
        """Round non-negative floats to the int64 Q15.48 state format."""
        q = [int(round(float(v) * (1 << STATE_FRAC_BITS))) for v in values]
        _check_range(q, 0, _INT64_MAX, "the state")
        return q

    @staticmethod
    def state_from_q(S_q: Sequence[int]) -> np.ndarray:
        """Q15.48 state values as float64."""
        return np.array([q / float(1 << STATE_FRAC_BITS) for q in S_q], dtype=np.float64)

    def update(self, S_q: Sequence[int], g_q: Sequence[int]) -> List[int]:
        """One update, ``S += round(gain * (g - S) / 2**32)``, with S and g in Q15.48.

        The product is formed from the two halves of ``g - S`` as the C code does, so
        no 128-bit type is needed.
        """
        if len(S_q) != N_FEATURES or len(g_q) != N_FEATURES:
            raise ValueError(f"S and g must have {N_FEATURES} values")
        _check_range(S_q, 0, _INT64_MAX, "S")
        _check_range(g_q, 0, _INT64_MAX, "g")
        out = []
        for gain, s, g in zip(self.gain_q32, S_q, g_q):
            d = g - s
            hi, lo = d >> 32, d & _UINT32_MAX
            step = _i64(gain * hi) + ((gain * lo + (1 << 31)) >> 32)
            out.append(_i64(s + step))
        return out

    @staticmethod
    def work_state(S_q: Sequence[int]) -> List[int]:
        """Q15.48 state to the Q15.16 working values (rounded half up, saturated)."""
        _check_range(S_q, 0, _INT64_MAX, "S")
        return [_sat32((s >> 32) + ((s >> 31) & 1)) for s in S_q]

    @staticmethod
    def age_q16(run_days: int) -> int:
        """Age in years, Q15.16: ``round(run_days * 2**16 / 365)`` saturated to int32."""
        _check_range([run_days], 0, _UINT32_MAX, "run_days")
        return _sat32(((run_days << WORK_FRAC_BITS) + DAYS_PER_YEAR // 2) // DAYS_PER_YEAR)

    def chs_q16(self, S_q: Sequence[int], run_days: int) -> List[int]:
        """Component health scores in Q15.16 (65536 = 1.0) from the state and the age."""
        if len(S_q) != N_FEATURES:
            raise ValueError(f"S must have {N_FEATURES} values")
        phi = self.work_state(S_q)
        if self.chs_use_age:
            a = self.age_q16(run_days)
            phi = phi + [_sat32(_round_shift(_i64(a * p), WORK_FRAC_BITS)) for p in phi]
        one = 1 << WORK_FRAC_BITS
        scores = []
        for row, e, b in zip(self.chs_w_q23, self.chs_w_exp, self.chs_bias_q16):
            acc = _i64(sum(w * p for w, p in zip(row, phi)))
            damage = _round_shift(acc, MANTISSA_BITS - e) + b
            scores.append(max(0, min(one, one - damage)))
        return scores

    def rul_days(self, S_q: Sequence[int], chs_q16: Sequence[int], run_days: int) -> int:
        """Remaining useful life in days (ridge head), rounded and clamped at zero."""
        if self.rul_coefs_q23 is None:
            raise ValueError("the model has no ridge RUL head")
        if len(chs_q16) != len(COMPONENTS):
            raise ValueError(f"chs_q16 must have {len(COMPONENTS)} values")
        _check_range(chs_q16, _INT32_MIN, _INT32_MAX, "chs_q16")
        x = self.work_state(S_q) + list(chs_q16)
        if self.rul_use_age:
            x.append(self.age_q16(run_days))
        coefs = self.rul_coefs_q23
        # coefs[-1] is the intercept, multiplied by 1.0 in Q15.16.
        acc = _i64(sum(c * v for c, v in zip(coefs, x)) + coefs[-1] * (1 << WORK_FRAC_BITS))
        days = _round_shift(acc, MANTISSA_BITS + WORK_FRAC_BITS - self.rul_exp)
        return _sat32(max(days, 0))


_C_PREAMBLE = """\
/* Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 *
 * Auto-generated by pqdna.edge_export. Do not edit by hand.
 *
 * PQ-DNA fixed-point model: constants and a C99 reference implementation of the
 * arithmetic. pqdna.edge_export.FixedPointModel runs the same arithmetic in Python,
 * bit for bit.
 *
 * Formats (Qm.n is a two's-complement integer holding x * 2^n):
 *   S, g      int64_t   Q15.48  PQ-DNA state and per-update kernel output, both >= 0
 *   working   int32_t   Q15.16  S, CHS and age inside the CHS and RUL stages
 *   gain      uint32_t  Q0.32   1 - exp(-dt / tau_k), dt = PQDNA_UPDATE_INTERVAL_S
 *   W, beta   int32_t   mantissa m, |m| <= 2^23, value = m * 2^(E - 23); one exponent E
 *                       per CHS row (PQDNA_CHS_W_EXP) and one for RUL (PQDNA_RUL_EXP)
 *   bias      int32_t   Q15.16 (CHS damage offset)
 *   run_days  uint32_t  device age in days; the age in years is run_days / 365
 * The scale s of the float CHS model is folded into W.
 *
 * Arithmetic (round(x) = floor(x + 1/2); sat() saturates to int32):
 *   update   S_k  += round(gain_k * (g_k - S_k) / 2^32)         every dt seconds
 *   work S   Sw_k  = sat(round(S_k / 2^32))                     Q15.16
 *   age      a     = sat(round(run_days * 2^16 / 365))          Q15.16, in years
 *   CHS      phi   = [Sw, sat(round(a * Sw / 2^16))]            age block only if used
 *            d_c   = round(sum_j W_cj * phi_j / 2^(23 - E_c)) + bias_c
 *            CHS_c = min(max(2^16 - d_c, 0), 2^16)              Q15.16, 2^16 = 1.0
 *   RUL      x     = [Sw, CHS, a]                               a only if used
 *            days  = max(0, round((sum_j beta_j * x_j + beta_n * 2^16) / 2^(39 - E)))
 *                    with beta_n the intercept, the last coefficient
 *
 * Requirements: C99 <stdint.h> and an arithmetic right shift of negative signed
 * integers (every mainstream compiler). S and g must lie in [0, 2^63).
 */
"""

_C_REFERENCE = """\
/* ---- reference implementation ------------------------------------------------ */

static inline int32_t pqdna_sat32_(int64_t v)
{
    return (int32_t)(v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : v));
}

/* x / 2^n rounded half up, 1 <= n <= 62 */
static inline int64_t pqdna_round_shift_(int64_t x, unsigned n)
{
    return (x + ((int64_t)1 << (n - 1))) >> n;
}

/* round(gain * d / 2^32) without a 128-bit type: d = hi * 2^32 + lo, 0 <= lo < 2^32 */
static inline int64_t pqdna_gain_step_(uint32_t gain, int64_t d)
{
    const int64_t hi = d >> 32;
    const uint64_t lo = (uint64_t)d & 0xFFFFFFFFu;
    return (int64_t)gain * hi + (int64_t)(((uint64_t)gain * lo + 0x80000000u) >> 32);
}

/* Q15.48 -> Q15.16, rounded half up and saturated */
static inline int32_t pqdna_work_(int64_t s)
{
    return pqdna_sat32_((s >> 32) + ((s >> 31) & 1));
}

/* age in years, Q15.16 */
static inline int32_t pqdna_age_(uint32_t run_days)
{
    return pqdna_sat32_((int64_t)((((uint64_t)run_days << 16) + 182u) / 365u));
}

/* One update of the state: S and g are Q15.48, both in [0, 2^63). */
static inline void pqdna_update(int64_t S[PQDNA_N_FEATURES], const int64_t g[PQDNA_N_FEATURES])
{
    int k;
    for (k = 0; k < PQDNA_N_FEATURES; k++) {
        S[k] += pqdna_gain_step_(PQDNA_GAIN_Q32[k], g[k] - S[k]);
    }
}

/* Component health scores, Q15.16 (65536 = 1.0), from the state and the run-day counter. */
static inline void pqdna_chs(const int64_t S[PQDNA_N_FEATURES], uint32_t run_days,
                             int32_t chs_q16[PQDNA_N_COMPONENTS])
{
    int32_t phi[PQDNA_CHS_N_INPUTS];
    int c, j, k;
    for (k = 0; k < PQDNA_N_FEATURES; k++) {
        phi[k] = pqdna_work_(S[k]);
    }
#if PQDNA_CHS_USE_AGE
    {
        const int64_t a = pqdna_age_(run_days);
        for (k = 0; k < PQDNA_N_FEATURES; k++) {
            phi[PQDNA_N_FEATURES + k] = pqdna_sat32_(pqdna_round_shift_(a * phi[k], 16));
        }
    }
#else
    (void)run_days;
#endif
    for (c = 0; c < PQDNA_N_COMPONENTS; c++) {
        int64_t acc = 0;
        int64_t score;
        for (j = 0; j < PQDNA_CHS_N_INPUTS; j++) {
            acc += (int64_t)PQDNA_CHS_W_Q23[c][j] * phi[j];
        }
        score = 65536 - (pqdna_round_shift_(acc, (unsigned)(23 - PQDNA_CHS_W_EXP[c]))
                         + PQDNA_CHS_BIAS_Q16[c]);
        chs_q16[c] = (int32_t)(score < 0 ? 0 : (score > 65536 ? 65536 : score));
    }
}
"""

_C_RUL = """
#if PQDNA_HAS_RUL
/* Remaining useful life in days (ridge head), rounded and clamped at zero. */
static inline int32_t pqdna_rul_days(const int64_t S[PQDNA_N_FEATURES],
                                     const int32_t chs_q16[PQDNA_N_COMPONENTS],
                                     uint32_t run_days)
{
    int64_t acc = 0;
    int j = 0;
    int k;
    for (k = 0; k < PQDNA_N_FEATURES; k++) {
        acc += (int64_t)PQDNA_RUL_COEFS_Q23[j++] * pqdna_work_(S[k]);
    }
    for (k = 0; k < PQDNA_N_COMPONENTS; k++) {
        acc += (int64_t)PQDNA_RUL_COEFS_Q23[j++] * chs_q16[k];
    }
#if PQDNA_RUL_USE_AGE
    acc += (int64_t)PQDNA_RUL_COEFS_Q23[j++] * pqdna_age_(run_days);
#else
    (void)run_days;
#endif
    acc += (int64_t)PQDNA_RUL_COEFS_Q23[j] * 65536;
    acc = pqdna_round_shift_(acc, (unsigned)(39 - PQDNA_RUL_EXP));
    return pqdna_sat32_(acc < 0 ? 0 : acc);
}
#endif /* PQDNA_HAS_RUL */
"""


def _c_values(values: Sequence[int], per_line: int = 8, indent: str = "    ",
              suffix: str = "") -> List[str]:
    """Format integers as C initialiser lines of at most ``per_line`` values."""
    out = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        out.append(indent + ", ".join(f"{v}{suffix}" for v in chunk) + ",")
    return out


def _render_header(model: FixedPointModel) -> str:
    n_comp = len(COMPONENTS)
    n_in = N_FEATURES * (2 if model.chs_use_age else 1)
    has_rul = model.rul_coefs_q23 is not None
    lines: List[str] = [
        _C_PREAMBLE,
        "#ifndef PQDNA_MODEL_H",
        "#define PQDNA_MODEL_H",
        "#include <stdint.h>",
        "",
        f"#define PQDNA_N_FEATURES {N_FEATURES}",
        f"#define PQDNA_N_COMPONENTS {n_comp}",
        f"#define PQDNA_CHS_N_INPUTS {n_in}",
        f"#define PQDNA_CHS_USE_AGE {int(model.chs_use_age)}",
        f"#define PQDNA_UPDATE_INTERVAL_S {model.dt_s!r}",
        f"#define PQDNA_HAS_RUL {int(has_rul)}",
    ]
    if has_rul:
        lines += [
            f"#define PQDNA_RUL_USE_AGE {int(model.rul_use_age)}",
            f"#define PQDNA_RUL_INPUT_LEN {len(model.rul_coefs_q23)}",
        ]

    lines += ["", "/* gain_k = round(2^32 * (1 - exp(-dt / tau_k))) */",
              "static const uint32_t PQDNA_GAIN_Q32[PQDNA_N_FEATURES] = {"]
    lines += _c_values(model.gain_q32, suffix="u")
    lines += ["};", "",
              "/* CHS weights, scale s folded in: value = m * 2^(E_c - 23) */",
              "static const int32_t PQDNA_CHS_W_Q23[PQDNA_N_COMPONENTS][PQDNA_CHS_N_INPUTS] = {"]
    for row in model.chs_w_q23:
        lines.append("    {")
        lines += _c_values(row, indent="        ")
        lines.append("    },")
    lines += ["};",
              "static const int8_t PQDNA_CHS_W_EXP[PQDNA_N_COMPONENTS] = {"]
    lines += _c_values(model.chs_w_exp)
    lines += ["};",
              "/* damage offset in Q15.16 */",
              "static const int32_t PQDNA_CHS_BIAS_Q16[PQDNA_N_COMPONENTS] = {"]
    lines += _c_values(model.chs_bias_q16)
    lines.append("};")

    if has_rul:
        age_term = "age, " if model.rul_use_age else ""
        lines += ["",
                  f"/* ridge RUL: [S (12), CHS ({n_comp}), {age_term}intercept], "
                  "value = m * 2^(E - 23) */",
                  "static const int32_t PQDNA_RUL_COEFS_Q23[PQDNA_RUL_INPUT_LEN] = {"]
        lines += _c_values(model.rul_coefs_q23)
        lines += ["};", f"static const int8_t PQDNA_RUL_EXP = {model.rul_exp};"]

    # Names are for diagnostics and can be stripped from a release build.
    lines += ["", "static const char *PQDNA_FEATURE_NAMES[PQDNA_N_FEATURES] = {"]
    lines += [f'    "{name}",' for name in FEATURE_NAMES]
    lines += ["};", "static const char *PQDNA_COMPONENT_NAMES[PQDNA_N_COMPONENTS] = {"]
    lines += [f'    "{name}",' for name in COMPONENTS]
    lines += ["};", "", _C_REFERENCE.rstrip("\n")]
    if has_rul:
        lines.append(_C_RUL.rstrip("\n"))
    lines += ["", "#endif /* PQDNA_MODEL_H */", ""]
    return "\n".join(lines)


def export_c_header(
    out_path: str,
    aggregator: PQDNAAggregator,
    chs: ComponentHealthScorer,
    rul: RULEstimator,
) -> Path:
    """Write ``pqdna_model.h``: the quantised constants, the formats and the C99 reference code.

    The file is written with ``\\n`` line endings on every platform. A ridge RUL head is
    exported when ``rul`` is a fitted ridge model (``PQDNA_HAS_RUL`` is 0 otherwise).
    Raises ValueError, writing nothing, if a value does not fit its format (see
    `FixedPointModel.from_models`).
    """
    text = _render_header(FixedPointModel.from_models(aggregator, chs, rul))
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8"))
    return p


def memory_footprint_bytes(aggregator: PQDNAAggregator,
                           chs: ComponentHealthScorer,
                           rul: RULEstimator) -> dict:
    """Return the on-device constant + state footprint of the exported model, in bytes.

    Counts the state ``S`` (12 x int64), the decay gains (12 x uint32), the uint32
    run-day age counter (when the CHS or the RUL head uses age), the CHS weights
    (int32) with their per-row exponents (int8) and the bias (int32), and the ridge
    RUL coefficients (int32) with their exponent (int8). The diagnostic name strings
    of the header are excluded. ``aggregator`` is accepted for symmetry with
    `export_c_header`; the footprint does not depend on it.
    """
    n_feat = N_FEATURES
    n_comp = len(COMPONENTS)
    has_rul = rul.model == "ridge" and rul.coefs_ is not None
    rul_uses_age = has_rul and len(rul.coefs_) == n_feat + n_comp + 2
    foot = {
        "S_state_bytes": n_feat * 8,                              # int64 Q15.48
        "gain_bytes": n_feat * 4,                                 # uint32 Q0.32
        "age_counter_bytes": 4 if (chs.use_age or rul_uses_age) else 0,   # uint32 run-days
        "chs_W_bytes": int(chs.W.size) * 4,                       # int32 mantissas
        "chs_W_exp_bytes": n_comp,                                # int8, one per row
        "chs_bias_bytes": n_comp * 4,                             # int32 Q15.16
    }
    if has_rul:
        foot["rul_coefs_bytes"] = int(len(rul.coefs_) * 4)        # int32 mantissas
        foot["rul_exp_bytes"] = 1                                 # int8
    foot["total_bytes"] = int(sum(foot.values()))
    return foot
