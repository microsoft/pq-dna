# pq-dna

**Firmware-resident lifetime power-quality stress signatures with on-device component health scoring.**

`pq-dna` is a reference implementation of a predictive-maintenance technique for power-electronics edge devices — electricity meters, power distribution units, server PSUs, UPS modules — that already digitise voltage and current waveforms for metering or control.

The device continuously condenses its waveform stream into a fixed-size **PQ-DNA state vector** `S` held in non-volatile memory, then maps `S` plus its own elapsed operating age to a **Component Health Score (CHS)** for each internal subsystem and a **remaining-useful-life (RUL)** estimate in days. The entire trained model and its state occupy **730 bytes** in the fixed-point export (see [Footprint](#footprint)). That size is meant to fit in the spare memory of the microcontroller such a device already has, but the export has not been tested on any hardware (see [Limitations](#limitations)). The design needs no new sensors and no raw waveform uplink.

> **Status: research reference implementation.** All results below are produced on synthetic data. The method has **not** been validated against real measurements or real component-failure records. See [Limitations](#limitations) before drawing engineering conclusions.

---

## Statement of need

Condition monitoring for this class of equipment is overwhelmingly either (a) *instantaneous* — threshold alarms on present-value power-quality events — or (b) *cloud-side*, requiring high-rate waveform or feature streams to be shipped off the device for analysis.

Neither serves the large installed base of memory-constrained devices that generate far more waveform data than they can afford to transmit. What is missing is a representation compact enough to live in under a kilobyte of device NVM, updatable with a handful of integer operations per channel per second, and rich enough to support per-subsystem prognosis on the device itself. How it behaves on a real microcontroller has not been measured.

`pq-dna` provides that representation, an open implementation of the aggregation and scoring math, a fixed-point exporter targeting embedded C, and a simulation harness with a deliberately **model-mismatched** ground-truth generator so that estimator accuracy can be measured without the circularity that makes self-consistent synthetic benchmarks meaningless.

Target audience: reliability and prognostics researchers, embedded firmware engineers working on metering or power-conversion equipment, and anyone studying the accuracy that can be recovered from heavily compressed on-device condition state.

---

## Installation

From the repository root:

```bash
python -m pip install -e .
```

Requires Python 3.9+. Runtime dependencies are NumPy and scikit-learn. For development extras:

```bash
python -m pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
from pqdna import PQFeatureExtractor, PQDNAAggregator, PQDNAState, ComponentHealthScorer

# 1. Extract 12 power-quality features from one second of samples.
fs = 7680
t = np.arange(fs) / fs
v = 325.0 * np.sin(2 * np.pi * 60 * t)
i = 42.0 * np.sin(2 * np.pi * 60 * t)

features = PQFeatureExtractor(fs=fs, line_hz=60.0).extract(v, i)

# 2. Fold them into the persistent lifetime state.
state = PQDNAState()
PQDNAAggregator(dt_window_s=1.0).update(state, features)

# 3. Score subsystem health, given the device's age in years.
print(ComponentHealthScorer().score(state, age_years=3.5))
# {'relay': 0.99..., 'smps': 0.99..., 'terminal_block': 0.99..., ...}
```

Reproduce every published number, including the ablations (run from the repository root; it takes a few minutes):

```bash
python -m pqdna.demo
```

Artifacts land in `out/` under the current working directory: `pqdna_model.h` (fixed-point firmware header), `chs_fit_metrics.csv`, `ablation_metrics.csv`, `rul_metrics.csv`, `feature_importance.csv`, `footprint.json`, `fixed_point_check.json`, `sample_meter_trace.csv`.

Run the tests:

```bash
python -m pytest tests -q
```

---

## Results

300 simulated devices, 15-year horizon, meter-disjoint 80/20 train/test split (240 devices to fit, 60 to test), model-mismatched ground truth with 4% multiplicative sensor noise (standard deviation 0.04), and a 13.7% cumulative failure rate (41 of the 300 devices reach CHS < 0.05 in some component within 15 years).

CHS MAE is the mean absolute error of the predicted CHS over the test devices' weekly snapshots. The baseline predicts, for every snapshot, the training-set mean CHS of that component. Skill is baseline MAE divided by CHS MAE, and "overall" averages the five components.

### Component health score

| Component | CHS MAE | Mean-predictor baseline | Skill |
|---|---|---|---|
| relay | 0.0203 | 0.1876 | 9.3x |
| smps | 0.0151 | 0.0860 | 5.7x |
| terminal_block | 0.0192 | 0.1728 | 9.0x |
| mcu | 0.0177 | 0.1247 | 7.0x |
| display | 0.0128 | 0.1008 | 7.9x |
| **overall** | **0.0170** | **0.1344** | **7.9x** |

### Remaining useful life

| Model | Intended target | RMSE | MAE |
|---|---|---|---|
| Ridge (float model; exported in fixed point) | on-device | 145 d | 98 d |
| Gradient boosting | cloud reference | 24 d | 14 d |
| Mean predictor | — | 1595 d | 1386 d |

RMSE and MAE are in days over the test snapshots. The target is the number of days from a snapshot to the first weekly snapshot at which any component's CHS is below 0.05; for a device that never fails it is the number of days to the end of the 15-year horizon (a right-censored value, used as observed). The mean predictor returns the training-set mean target. The CHS input of both regressors is the simulator's ground-truth score, not the output of the scorer (see [Limitations](#limitations)). The fixed-point export of the ridge head reproduces the float head to within 0.63 d (see [Footprint](#footprint)).

### Ablations

This is the most important table in the project.

| Ground-truth generator | Age feature | CHS MAE | Skill vs baseline |
|---|---|---|---|
| mismatched | yes | 0.0170 | 7.9x |
| mismatched | no | 0.0695 | 1.9x |
| matched | yes | 0.0043 | 8.9x |
| matched | no | 0.0044 | 8.6x |

Two findings:

1. **The EWMA state alone is not sufficient.** `S` is an exponentially-weighted moving average, so it estimates the *current stress level* and reaches steady state within a few decay horizons. Accumulated fatigue damage is a monotone integral of that stress. Multiplying `S` by the device's elapsed age supplies the missing dose-time product and improves CHS MAE by **4.09x** (0.0695 → 0.0170). Age costs one `uint32` and is already tracked by every device of this class.

2. **A model-matched synthetic benchmark hides that entirely.** Under the legacy `matched` generator — where ground truth is an exactly linear function of the same `S` the estimator fits — the age feature changes nothing (0.0044 → 0.0043), and the apparent MAE is ~4x better than reality. Any evaluation built on a generator that shares the estimator's functional form measures self-consistency, not accuracy.

### Footprint

| Item | Bytes | Computation |
|---|---|---|
| `S` state (`int64`, Q15.48) | 96 | 12 channels × 8 B |
| decay gains `1 − α` (`uint32`, Q0.32) | 48 | 12 channels × 4 B |
| age counter (`uint32`, run-days) | 4 | 1 × 4 B |
| CHS weights `s·W` (`int32` mantissas) | 480 | 5 components × 24 inputs × 4 B |
| CHS weight exponents (`int8`) | 5 | 5 components × 1 B |
| CHS bias `b` (`int32`, Q15.16) | 20 | 5 components × 4 B |
| RUL ridge coefficients `β` (`int32` mantissas) | 76 | 19 coefficients × 4 B |
| RUL exponent (`int8`) | 1 | 1 × 1 B |
| **total** | **730** | |

Qm.n is a two's-complement integer holding `x · 2^n`. A weight or coefficient is stored as an integer mantissa `m` (at most `2^23` in magnitude) with one power-of-two exponent `E` for each block, value `m · 2^(E−23)`, and the scale `s` is folded into the CHS weights. The state is 64 bits wide because at a 1 s update interval the gain of a 60-day channel is only `1.9e-7`: a 16-bit `α` cannot resolve it (a Q15 `α` saturates at `32767` for every channel, an effective `τ` of about 9 hours) and a 32-bit state would stall. The exporter raises an error instead of saturating a value that does not fit. The total counts model constants plus persistent state; it excludes the diagnostic name strings of the generated header.

The header also holds a C99 reference implementation of the arithmetic (`pqdna_update`, `pqdna_chs`, `pqdna_rul_days`), specified in [docs/METHOD.md](docs/METHOD.md#34-fixed-point-export). The demo compares the bit-exact Python model of that code with the float model on 4339 held-out snapshots (every tenth test row; `out/fixed_point_check.json`): the CHS differs by at most 2.0e-5 and the RUL by at most 0.63 d. The time constants are quantised to within 0.06% (the 60-day channels become 59.96 d).

---

## How it works

```
 v(t), i(t)  ──▶  PQFeatureExtractor  ──▶  x: 12 power-quality features
                                                    │
                                                    ▼
                                          S ← α*S + (1−α)*g(x)       [96 B NVM]
                                                    │
                            ┌───────────────────────┴──────────────┐
                            ▼                                      ▼
              CHS = clip(1 − W@φ − b, 0, 1)                RUL = max(0, β@z)
                            │                                      │
                            └──────────────┬───────────────────────┘
                                           ▼
                              alert on threshold crossing
```

`*` multiplies element-wise and `@` is a dot product, as in NumPy; `[A, B]` concatenates. Symbols:

| Symbol | Meaning |
|---|---|
| `x` | the 12 power-quality features of one update window |
| `g` | per-channel damage kernel, applied channel by channel to `x` |
| `α` | per-channel decay `exp(−Δt/τ)`, with the update interval `Δt` (1 s on a device) and the channel time constant `τ`, both in seconds |
| `S` | PQ-DNA state, 12 values, all zero at manufacture |
| `a` | device age in years: run-days ÷ 365 |
| `s` | fixed scale, `1e-3` |
| `φ` | `s*[S, a*S]`, the 24 inputs of the health score |
| `W`, `b` | health-score weights (24 per component) and bias, one set for each of the 5 components |
| `z` | `[S, CHS, a, 1]`, the 19 inputs of the RUL head (`S`, the 5 component scores, the age and a constant 1), whose coefficients are `β` |

Each of the 12 channels carries its own decay time constant `τ`, chosen to match the horizon of the failure mode that channel drives — 1 day for transients (relay contact erosion) through 60 days for harmonic distortion (SMPS capacitor stress). Full derivation, all 12 damage kernels with their exact constants, and the design of the model-mismatched generator are in [docs/METHOD.md](docs/METHOD.md).

---

## Limitations

Read this before citing or building on the results.

- **No real-world validation.** Every number in this repository comes from simulation. The physics-of-failure weights (`FAILURE_MODES`, the prior for `W`) have not been calibrated against accelerated-life-test data, and the method has never been run on a physical device. Contributions of real measurement data are the most valuable thing this project could receive.
- **The synthetic generator is a model, not reality.** It is deliberately structured to differ from the estimator (different damage exponents, cumulative rather than EWMA accumulation, non-linear health link, cross-component coupling, sensor noise), which removes the circularity in the evaluation but does not make it realistic. Real degradation has regime changes, repair events, and censoring that are not modelled.
- **The fixed-point export has not been run on a device.** The C code in the generated header compiles as C99 and, in the test suite, reproduces the Python integer model bit for bit whenever a C compiler is available; that model matches the float model closely (see [Footprint](#footprint)). It has not been run on target hardware or inside a firmware build. The damage kernels `g` that turn a feature vector into the dose are not part of the export (the firmware supplies `g` to `pqdna_update`), and the update needs 64-bit integer arithmetic.
- **Simulated and on-device state are in different units for the event channels.** The simulator draws one feature vector per day: `inrush_count` is events per day, `sag_dwell` and `swell_dwell` are sag and swell events per day × 0.001, and `transient_pu` is the largest transient of the day. The extractor defines the same channels per 1 s window: crossings of 4 × the rated current in the window, the fraction of the window's cycles in a sag or swell, and the largest step in the window. The trained weights belong to the per-day convention. A device that updated `S` every second from per-window features would hold far smaller values in these channels (about 1/86,400 of the per-day `inrush_count` at the same event rate), so its scores would not mean the same thing without refitting or per-day aggregation.
- **The RUL head is evaluated on ground-truth health scores.** `pqdna.demo` fits and scores the RUL regressors with the simulator's true CHS as an input, whereas a device would feed them the scorer's estimate, so the RUL errors above are not end-to-end device errors. Devices that never fail contribute right-censored targets (days to the end of the horizon) as if they were observed.
- **The 12-channel feature set follows IEEE 1159 terminology but is not a conformant implementation** of IEEE 1159, IEEE 519 or IEC 61000-4-30/-4-15. In particular the flicker channel is a crude proxy, not an IEC flickermeter.
- **CHS is a modelled quantity, not a measurement.** It is only as meaningful as the weight matrix behind it.

---

## Repository layout

| Path | Contents |
|---|---|
| [src/pqdna/features.py](src/pqdna/features.py) | 12-channel power-quality feature extraction from raw samples |
| [src/pqdna/pqdna.py](src/pqdna/pqdna.py) | Persistent state and the per-channel decayed accumulator |
| [src/pqdna/chs.py](src/pqdna/chs.py) | Component health scoring and its calibration |
| [src/pqdna/rul.py](src/pqdna/rul.py) | Remaining-useful-life heads (ridge for device, GBR for cloud) |
| [src/pqdna/synth.py](src/pqdna/synth.py) | Fleet simulator and both ground-truth generators |
| [src/pqdna/edge_export.py](src/pqdna/edge_export.py) | Fixed-point C header export, its bit-exact integer reference model and footprint accounting |
| [src/pqdna/demo.py](src/pqdna/demo.py) | End-to-end pipeline reproducing all published metrics |
| [docs/METHOD.md](docs/METHOD.md) | Full method description, damage kernels, evaluation design |
| [tests/](tests/) | Test suite, including regression guards against evaluation circularity |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). For support, see [SUPPORT.md](SUPPORT.md). For security issues, follow [SECURITY.md](SECURITY.md) — do not open a public issue.

## License

MIT. See [LICENSE](LICENSE). Third-party dependency notices are in [NOTICE.md](NOTICE.md).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
