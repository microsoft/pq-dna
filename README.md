# pq-dna

**Firmware-resident lifetime power-quality stress signatures with on-device component health scoring.**

`pq-dna` is a reference implementation of a predictive-maintenance technique for
power-electronics edge devices — electricity meters, power distribution units,
server PSUs, UPS modules — that already digitise voltage and current waveforms
for metering or control.

The device continuously condenses its waveform stream into a fixed-size
**PQ-DNA state vector** `S` held in non-volatile memory, then maps `S` plus its
own elapsed operating age to a **Component Health Score (CHS)** for each
internal subsystem and a **remaining-useful-life (RUL)** estimate in days. The
entire trained model exports to **378 bytes** of fixed-point C constants, so it
runs on the microcontroller the device already has. No new sensors, no raw
waveform uplink.

> **Status: research reference implementation.** All results below are produced
> on synthetic data. The method has **not** been validated against real
> measurements or real component-failure records. See
> [Limitations](#limitations) before drawing engineering conclusions.

---

## Statement of need

Condition monitoring for this class of equipment is overwhelmingly either
(a) *instantaneous* — threshold alarms on present-value power-quality events —
or (b) *cloud-side*, requiring high-rate waveform or feature streams to be
shipped off the device for analysis.

Neither serves the large installed base of memory-constrained devices that
generate far more waveform data than they can afford to transmit. What is
missing is a representation compact enough to live in a few hundred bytes of
device NVM, cheap enough to update on a Cortex-M0+-class core, and rich enough
to support per-subsystem prognosis on the device itself.

`pq-dna` provides that representation, an open implementation of the
aggregation and scoring math, a fixed-point exporter targeting embedded C, and
a simulation harness with a deliberately **model-mismatched** ground-truth
generator so that estimator accuracy can be measured without the circularity
that makes self-consistent synthetic benchmarks meaningless.

Target audience: reliability and prognostics researchers, embedded firmware
engineers working on metering or power-conversion equipment, and anyone
studying the accuracy that can be recovered from heavily compressed on-device
condition state.

---

## Installation

```bash
python -m pip install -e .
```

Requires Python 3.9+. Runtime dependencies are NumPy and scikit-learn.
For development extras:

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

Reproduce every published number, including the ablations:

```bash
python -m pqdna.demo
```

Artifacts land in `out/`: `pqdna_model.h` (fixed-point firmware header),
`chs_fit_metrics.csv`, `ablation_metrics.csv`, `rul_metrics.csv`,
`feature_importance.csv`, `footprint.json`, `sample_meter_trace.csv`.

Run the tests:

```bash
python -m pytest tests -q
```

---

## Results

300 simulated devices, 15-year horizon, meter-disjoint 80/20 train/test split,
model-mismatched ground truth with 4% sensor noise, 13.7% cumulative failure
rate.

### Component health score

| Component | CHS MAE | Mean-predictor baseline | Skill |
|---|---|---|---|
| relay | 0.0201 | 0.1876 | 9.3x |
| smps | 0.0148 | 0.0860 | 5.8x |
| terminal_block | 0.0192 | 0.1728 | 9.0x |
| mcu | 0.0174 | 0.1247 | 7.2x |
| display | 0.0126 | 0.1008 | 8.0x |
| **overall** | **0.0168** | **0.1344** | **8.0x** |

### Remaining useful life

| Model | Where it runs | RMSE | MAE |
|---|---|---|---|
| Ridge (Q23 fixed-point) | on-device | 142 d | 96 d |
| Gradient boosting | cloud reference | 25 d | 14 d |
| Mean predictor | — | 1595 d | — |

### Ablations

This is the most important table in the project.

| Ground-truth generator | Age feature | CHS MAE | Skill vs baseline |
|---|---|---|---|
| mismatched | yes | 0.0168 | 8.0x |
| mismatched | no | 0.0681 | 2.0x |
| matched | yes | 0.0043 | 8.9x |
| matched | no | 0.0044 | 8.6x |

Two findings:

1. **The EWMA state alone is not sufficient.** `S` is an exponentially-weighted
   moving average, so it estimates the *current stress level* and reaches steady
   state within a few decay horizons. Accumulated fatigue damage is a monotone
   integral of that stress. Multiplying `S` by the device's elapsed age supplies
   the missing dose-time product and improves CHS MAE by **4.1x**
   (0.0681 → 0.0168). Age costs one `uint32` and is already tracked by every
   device of this class.

2. **A model-matched synthetic benchmark hides that entirely.** Under the
   legacy `matched` generator — where ground truth is an exactly linear function
   of the same `S` the estimator fits — the age feature changes nothing
   (0.0044 → 0.0043), and the apparent MAE is ~4x better than reality. Any
   evaluation built on a generator that shares the estimator's functional form
   measures self-consistency, not accuracy.

### Footprint

| Item | Bytes |
|---|---|
| `S` state (Q15) | 24 |
| decay coefficients (Q15) | 24 |
| age counter (uint32) | 4 |
| CHS weight matrix (Q15) | 240 |
| CHS bias (Q15) | 10 |
| RUL ridge coefficients (Q23) | 76 |
| **total** | **378** |

---

## How it works

```
 v(t), i(t)  ──▶  PQFeatureExtractor  ──▶  12 power-quality features
                                                    │
                                                    ▼
                                   S ← α·S + (1−α)·g(features)      [24 B NVM]
                                                    │
                            ┌───────────────────────┴──────────────┐
                            ▼                                      ▼
              CHS = clip(1 − W·[S, S·age] − b)          RUL = ridge([S, CHS, age])
                            │                                      │
                            └──────────────┬───────────────────────┘
                                           ▼
                              alert on threshold crossing
```

Each of the 12 channels carries its own decay time constant `τ`, chosen to match
the horizon of the failure mode that channel drives — 1 day for transients
(relay contact erosion) through 60 days for harmonic distortion (SMPS capacitor
stress). Full derivation, the damage kernels, and the design of the
model-mismatched generator are in [docs/METHOD.md](docs/METHOD.md).

---

## Limitations

Read this before citing or building on the results.

- **No real-world validation.** Every number in this repository comes from
  simulation. The physics-of-failure weight matrix `W` has not been calibrated
  against accelerated-life-test data, and the method has never been run on a
  physical device. Contributions of real measurement data are the most valuable
  thing this project could receive.
- **The synthetic generator is a model, not reality.** It is deliberately
  structured to differ from the estimator (different damage exponents,
  cumulative rather than EWMA accumulation, non-linear health link,
  cross-component coupling, sensor noise), which removes the circularity in the
  evaluation but does not make it realistic. Real degradation has regime
  changes, repair events, and censoring that are not modelled.
- **Fixed-point export is not bit-accurate against a device.** The generated
  header has not been compiled and run against a Q15/Q23 reference
  implementation on hardware, so quantisation error is unmeasured.
- **The 12-channel feature set follows IEEE 1159 terminology but is not a
  conformant implementation** of IEEE 1159, IEEE 519 or IEC 61000-4-30/-4-15.
  In particular the flicker channel is a crude proxy, not an IEC flickermeter.
- **CHS is a modelled quantity, not a measurement.** It is only as meaningful
  as the weight matrix behind it.

---

## Repository layout

| Path | Contents |
|---|---|
| [src/pqdna/features.py](src/pqdna/features.py) | 12-channel power-quality feature extraction from raw samples |
| [src/pqdna/pqdna.py](src/pqdna/pqdna.py) | Persistent state and the per-channel decayed accumulator |
| [src/pqdna/chs.py](src/pqdna/chs.py) | Component health scoring and its calibration |
| [src/pqdna/rul.py](src/pqdna/rul.py) | Remaining-useful-life heads (ridge for device, GBR for cloud) |
| [src/pqdna/synth.py](src/pqdna/synth.py) | Fleet simulator and both ground-truth generators |
| [src/pqdna/edge_export.py](src/pqdna/edge_export.py) | Fixed-point C header export and footprint accounting |
| [src/pqdna/demo.py](src/pqdna/demo.py) | End-to-end pipeline reproducing all published metrics |
| [docs/METHOD.md](docs/METHOD.md) | Full method description, damage kernels, evaluation design |
| [tests/](tests/) | Test suite, including regression guards against evaluation circularity |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). For support, see [SUPPORT.md](SUPPORT.md).
For security issues, follow [SECURITY.md](SECURITY.md) — do not open a public issue.

## License

MIT. See [LICENSE](LICENSE). Third-party dependency notices are in [NOTICE.md](NOTICE.md).

## Trademarks

This project may contain trademarks or logos for projects, products, or services.
Authorized use of Microsoft trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not
cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks
or logos are subject to those third-party's policies.
