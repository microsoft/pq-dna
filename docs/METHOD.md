# PQ-DNA: method and evaluation design

This document describes what `pq-dna` computes, why the evaluation is
constructed the way it is, and where the method's limits are. It is the
technical companion to [../README.md](../README.md).

---

## 1. Problem setting

A large class of power-electronics devices — revenue meters, power distribution
units, server power supplies, UPS and battery-backup controllers, automatic
transfer switches — already digitise voltage and current at kilohertz rates for
metering or closed-loop control. They have spare microcontroller cycles and a
small amount of non-volatile memory. They do **not** have the bandwidth to ship
waveforms upstream, nor the memory to retain them.

These devices fail from cumulative electrical stress: harmonic distortion from
non-linear loads, repetitive inrush, transients, sustained off-nominal voltage.
The failure is gradual; the observation is not. Present-generation condition
monitoring reports discrete events and instantaneous values, which means the
information needed to anticipate wear-out is generated and then discarded,
sample by sample, for the device's entire service life.

The question this project addresses is narrow and empirical: **how much
prognostic signal survives compression into a few hundred bytes of device
state?**

---

## 2. The PQ-DNA state

### 2.1 Feature channels

Each observation window (1 s by default) produces a 12-dimensional feature
vector, computed from one real FFT plus sample-domain statistics. Channel names
follow IEEE 1159 terminology.

| # | Channel | Meaning | Units |
|---|---|---|---|
| 0 | `v_rms` | RMS voltage | per-unit |
| 1 | `i_rms` | RMS current | per-unit |
| 2 | `v_thd` | Voltage total harmonic distortion | fraction |
| 3 | `i_thd` | Current total harmonic distortion | fraction |
| 4 | `i_crest` | Current crest factor | ratio |
| 5 | `transient_pu` | Largest transient excursion in window | per-unit |
| 6 | `sag_dwell` | Fraction of window below the sag threshold | fraction |
| 7 | `swell_dwell` | Fraction of window above the swell threshold | fraction |
| 8 | `flicker_pst` | Short-term flicker proxy | index |
| 9 | `freq_dev` | Line-frequency deviation | per-unit |
| 10 | `unbalance` | Phase unbalance | fraction |
| 11 | `inrush_count` | Inrush / arc events in window | count |

These are *not* a conformant implementation of IEEE 1159, IEEE 519 or
IEC 61000-4-30. `flicker_pst` in particular is a rough proxy and not an
IEC 61000-4-15 flickermeter.

### 2.2 Damage kernel

Raw feature values are mapped to a per-window damage-equivalent dose by a
monotone, per-channel non-linearity `g(.)` chosen to follow the shape of the
dominant physics-of-failure relationship for that channel. Implemented in
`pqdna.vectorised_damage`:

$$
g_0 = 50\,(v_{rms}-1)^2 \qquad
g_1 = i_{rms}^2 \qquad
g_{2,3} = 20\,\mathrm{THD}^{1.5}
$$

$$
g_4 = \max(i_{crest}-\sqrt2,\,0)^2 \qquad
g_5 = 10\,\mathrm{transient}^{2.5}
$$

with the remaining channels linear in their drivers. The exponents follow
standard reliability models — $I^2R$ Joule heating, Coffin–Manson power-law
thermal fatigue, contact-erosion integrals — but the coefficients are
illustrative, **not** fitted to hardware.

### 2.3 Accumulation

The persistent state is a per-channel exponentially-weighted moving average:

$$
S_t = \alpha \odot S_{t-1} + (1-\alpha) \odot g(x_t),
\qquad
\alpha_k = e^{-\Delta t / \tau_k}
$$

The design choice that matters here is that $\tau$ is **per channel**, and each
$\tau_k$ is set to the horizon of the failure mode that channel drives:

| Channel group | $\tau$ | Rationale |
|---|---|---|
| `transient_pu`, `inrush_count` | 1 day | Relay contact erosion, arc events |
| `i_crest`, `sag_dwell`, `swell_dwell` | 7 days | Terminal-block hot-spot cycling |
| `v_rms`, `i_rms`, `flicker_pst`, `freq_dev`, `unbalance` | 30 days | Electrolytic dry-out, Joule heating |
| `v_thd`, `i_thd` | 60 days | SMPS and capacitor stress |

A channel whose damage mechanism is fast gets a short memory; a slow mechanism
gets a long one. This is the part of the construction least likely to be
arrived at by default.

---

## 3. Health and life estimation

### 3.1 Why the state alone is insufficient

An EWMA is a *level* estimator. Given stationary input it converges to the mean
dose in a few multiples of $\tau$ and then stays there. After the first few
months of service, two devices in the same environment have nearly identical
`S` regardless of whether one is three years old and the other fifteen.

Accumulated fatigue damage, by contrast, is a monotone integral (Miner's rule):

$$
L_c(T) = \int_0^{T} \sum_k w_{ck}\, d_k(t)\, \mathrm{d}t
$$

`S` and `L` are different functionals of the same input stream, and no choice of
weights recovers one from the other. `S` carries the *rate*; the missing factor
is exposure *duration*.

The fix is to supply the device's own elapsed operating age — which every device
of this class already tracks — and let the estimator form the dose-time product:

$$
\mathrm{CHS}_c = \mathrm{clip}\!\left(1 - W_c \cdot [\,S,\; S \cdot \mathrm{age}\,] - b_c,\; 0,\; 1\right)
$$

Section 5 shows this is worth a 4.1x reduction in CHS error.

### 3.2 Calibration

`W` and `b` are obtained by closed-form ridge regression against observed health
labels. In deployment those labels come from accelerated-life testing per
hardware revision; in this repository they come from the simulator.

### 3.3 Remaining useful life

RUL regresses days-to-threshold on $[S, \mathrm{CHS}, \mathrm{age}]$. Two heads:
a ridge model small enough to evaluate on the device, and a gradient-boosting
model as a cloud-side reference for the accuracy being given up.

---

## 4. Evaluation design

### 4.1 The circularity problem

The first version of this project evaluated CHS against a synthetic ground truth
defined as

$$
\mathrm{CHS}^{\text{true}}_c = \mathrm{clip}\!\left(1 - \tfrac{\lambda}{r_c} W_c \cdot S\right)
$$

using the *same* weight matrix `W` the estimator was initialised with and then
fitted. The estimator's own form is $\mathrm{clip}(1 - W\cdot S - b)$. Fitting a
linear model to data generated by a linear model with one nuisance scalar is a
self-consistency check, not a measurement. It reported CHS MAE of 0.002–0.011
and a "7–9x improvement over baseline" that was an artefact of the setup.

That generator is retained as `generator="matched"` **solely** so the size of the
artefact can be quantified. It must never be used for headline numbers.

### 4.2 The mismatched generator

The default generator (`generator="mismatched"`, in `pqdna.synth`) differs from
the estimator along four independent axes, each corresponding to a real property
of component degradation:

1. **Different damage exponents.** The latent dose uses
   `LATENT_DOSE_GAIN` and `LATENT_DOSE_EXP` — e.g. $i_{rms}^{2.6}$ against the
   device's $i_{rms}^{2}$, $\mathrm{transient}^{3.1}$ against $^{2.5}$. The
   estimator cannot recover the latent exponents from `S`.
2. **Cumulative, not exponentially-weighted.** Latent damage is
   $L = \sum_t \mathrm{dose}(t)$, monotone in time. This is the mismatch that
   matters most, and the reason the age feature is required.
3. **Non-linear health link.**
   $\mathrm{CHS} = \exp\!\left(-(L/L_{\text{ref}} r)^{m}\right)$ with a
   per-component Weibull-like shape exponent $m \in [1.3, 1.7]$, against the
   estimator's `clip(1 - linear)`.
4. **Cross-component coupling.** A degrading SMPS accelerates MCU and display
   aging; relay and terminal-block wear are mutually reinforcing. The estimator
   scores components independently and structurally cannot represent this.

In addition, multiplicative sensor noise (4% by default) is applied to the
device-visible `S`, so the estimator does not even observe the trajectory that
drove the latent process.

Characteristic lives are anchored to "years of nominal urban exposure to reach
$\mathrm{CHS}=1/e$" and calibrated to a 13.7% cumulative failure rate over 15
years, consistent with the ~0.6%/yr premature-failure rate reported for this
equipment class.

### 4.3 Protocol

- 300 devices, 15-year horizon, weekly snapshots.
- Environment mix: 20% clean grid, 45% urban residential, 20% EV corridor,
  15% industrial.
- **Meter-disjoint** 80/20 split — no device contributes rows to both train and
  test, so temporal autocorrelation within a trajectory cannot leak.
- Baseline is the training-set mean CHS, evaluated on test rows.

### 4.4 Regression guards

`tests/test_pqdna.py` asserts that the latent dose kernel is *not* numerically
close to the device kernel, that ground-truth trajectories are monotone, that
the age feature strictly helps, that the matched generator is strictly easier
than the mismatched one, and that observation noise is actually applied. These
exist to stop the circularity from silently returning.

---

## 5. Results

See [../README.md](../README.md#results) for the full tables. Summary:

- CHS MAE **0.0168** overall against the mismatched generator, 8.0x better than
  a mean predictor.
- Removing the age feature degrades this to **0.0681** — a 4.1x penalty.
- Under the matched generator the age feature is worth nothing
  (0.0044 → 0.0043), demonstrating that the original evaluation design was
  incapable of revealing the flaw.
- RUL ridge RMSE **142 days** against a mean-predictor RMSE of 1595 days. The
  cloud-side GBR reaches 25 days, which bounds what is being traded away for
  on-device execution.
- Total exported model: **378 bytes**.

---

## 6. What this does not establish

The evaluation removes circularity. It does not establish real-world accuracy,
and nothing here should be read as a performance claim about physical hardware.

Specifically unvalidated: the damage kernel exponents, the physics-of-failure
weight matrix, the decay time constants, the fixed-point quantisation error, and
the entire premise that the 12 chosen channels span the failure-relevant
subspace. Each requires either accelerated-life-test data or field data with
confirmed component-level failure attribution.

The most useful contribution anyone could make to this project is a dataset that
pairs power-quality recordings with confirmed component replacements. See
[../CONTRIBUTING.md](../CONTRIBUTING.md).
