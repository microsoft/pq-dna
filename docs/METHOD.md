# PQ-DNA: method and evaluation design

<!-- Math note: GitHub resolves Markdown backslash escapes before typesetting math, so inside math spans a backslash followed by punctuation loses its backslash. Use \thinspace, \negthinspace and \mkern5mu instead of \, \! and \;, \cr instead of \\ as the row separator in aligned blocks, \lbrace and \rbrace instead of \{ and \}, and \Vert instead of \|. Also avoid a closing brace directly followed by an underscore in prose paragraphs (write \tilde D_c, not \tilde{D}_c): Markdown can read that underscore as an emphasis marker and break the formula. -->

This document describes what `pq-dna` computes, why the evaluation is constructed the way it is, and where the method's limits are. It is the technical companion to [../README.md](../README.md).

## Notation

All formulas are written in exact arithmetic. Channels are indexed by $k = 0, \ldots, 11$ in the order of the table in Section 2.1, components by $c$ over the five subsystems `relay`, `smps`, `terminal_block`, `mcu` and `display`, and updates by $t$.

| Symbol | Meaning | Shape and units |
|---|---|---|
| $x_t$ | feature vector of update $t$ (Section 2.1) | 12 values |
| $g_k$ | damage kernel of channel $k$ (Section 2.2) | scalar function |
| $S_t$ | PQ-DNA state, with $S_0 = 0$ (Section 2.3) | 12 values, damage-dose units |
| $\Delta t$ | interval between updates | seconds |
| $\tau_k$ | decay time constant of channel $k$ | seconds |
| $\alpha_k$ | per-channel decay, $e^{-\Delta t / \tau_k}$ | between 0 and 1 |
| $\gamma_k$ | per-channel gain, $1 - \alpha_k$ | between 0 and 1 |
| $a$ | device age, run-days divided by 365 | years |
| $s$ | fixed stress-to-health scale, $10^{-3}$ | dimensionless |
| $\phi(S, a)$ | health-score inputs, $s \cdot \mathrm{concat}(S, aS)$ (Section 3.1) | 24 values |
| $W_c$, $b_c$ | health-score weights and bias of component $c$ | 24 values, 1 value |
| $\mathrm{CHS}_c$ | health score of component $c$ | between 0 (end of life) and 1 (factory-new) |
| $\tilde{z}$, $\beta$ | RUL-head inputs and coefficients (Section 3.3) | 19 values each |

---

## 1. Problem setting

A large class of power-electronics devices — revenue meters, power distribution units, server power supplies, UPS and battery-backup controllers, automatic transfer switches — already digitise voltage and current at kilohertz rates for metering or closed-loop control. They have spare microcontroller cycles and a small amount of non-volatile memory. They do **not** have the bandwidth to ship waveforms upstream, nor the memory to retain them.

These devices fail from cumulative electrical stress: harmonic distortion from non-linear loads, repetitive inrush, transients, sustained off-nominal voltage. The failure is gradual; the observation is not. Present-generation condition monitoring reports discrete events and instantaneous values, which means the information needed to anticipate wear-out is generated and then discarded, sample by sample, for the device's entire service life.

The question this project addresses is narrow and empirical: **how much prognostic signal survives compression into under a kilobyte of device model and state?**

---

## 2. The PQ-DNA state

### 2.1 Feature channels

Each observation window (60 mains cycles, 1 s at 60 Hz) produces a 12-dimensional feature vector, computed from one real FFT plus sample-domain statistics. Channel names follow IEEE 1159 terminology. "Per-unit" means divided by the nominal voltage (230 V) for voltages and by the rated current (100 A) for currents; both are arguments of `PQFeatureExtractor`.

| # | Channel | Meaning | Units |
|---|---|---|---|
| 0 | `v_rms` | RMS voltage over the window | per-unit |
| 1 | `i_rms` | RMS current over the window | per-unit |
| 2 | `v_thd` | Voltage total harmonic distortion: root-sum-square of harmonics 2 to 15 relative to the fundamental | fraction |
| 3 | `i_thd` | Current total harmonic distortion (same definition) | fraction |
| 4 | `i_crest` | Current crest factor: peak divided by RMS | ratio |
| 5 | `transient_pu` | Largest sample-to-sample voltage step in the window | per-unit |
| 6 | `sag_dwell` | Fraction of mains cycles in the window whose RMS voltage is below 0.9 per-unit | fraction |
| 7 | `swell_dwell` | Fraction of mains cycles in the window whose RMS voltage is above 1.1 per-unit | fraction |
| 8 | `flicker_pst` | Short-term flicker proxy | index |
| 9 | `freq_dev` | Line-frequency deviation from nominal, as a fraction of nominal | per-unit |
| 10 | `unbalance` | Phase unbalance (always 0 for a single-phase meter) | fraction |
| 11 | `inrush_count` | Inrush / arc events in window: upward crossings of 4 times the rated current | count |

These are *not* a conformant implementation of IEEE 1159, IEEE 519 or IEC 61000-4-30. `flicker_pst` in particular is a rough proxy and not an IEC 61000-4-15 flickermeter.

### 2.2 Damage kernel

Raw feature values are mapped to a per-window damage-equivalent dose by a per-channel non-linearity $g_k$ that is non-decreasing in the stress driver (for `v_rms` the driver is the deviation from 1 per-unit, in either direction), chosen to follow the shape of the dominant physics-of-failure relationship for that channel. Implemented in `pqdna.vectorised_damage`, with $x$ the value of channel $k$ in the units of the table above:

| $k$ | Channel | Kernel $g_k(x)$ | Basis |
|---|---|---|---|
| 0 | `v_rms` | $50 (x - 1)^2$ | off-nominal voltage, symmetric about 1 per-unit |
| 1 | `i_rms` | $x^2$ | $I^2R$ Joule heating |
| 2 | `v_thd` | $20 x^{1.5}$ | SMPS and capacitor stress |
| 3 | `i_thd` | $20 x^{1.5}$ | SMPS and capacitor stress |
| 4 | `i_crest` | $\max(x - 1.414, 0)^2$ | crest factor above the sinusoidal value, $\sqrt{2} \approx 1.414$ |
| 5 | `transient_pu` | $10 x^{2.5}$ | Coffin–Manson power law on transient amplitude |
| 6 | `sag_dwell` | $10 x$ | linear |
| 7 | `swell_dwell` | $10 x$ | linear |
| 8 | `flicker_pst` | $x$ | linear |
| 9 | `freq_dev` | $1000 x$ | linear |
| 10 | `unbalance` | $100 x$ | linear |
| 11 | `inrush_count` | $x$ | contact erosion grows linearly with arc events |

The exponents follow standard reliability models — $I^2R$ Joule heating, Coffin–Manson power-law thermal fatigue, contact-erosion integrals — but the coefficients are illustrative, **not** fitted to hardware.

### 2.3 Accumulation

The persistent state is a per-channel exponentially-weighted moving average of the damage dose, started from zero:

$$
\begin{aligned}
S_{t,k} &= \alpha_k S_{t-1,k} + (1 - \alpha_k) g_k(x_{t,k}), \qquad S_{0,k} = 0 \cr
\alpha_k &= e^{-\Delta t / \tau_k}
\end{aligned}
$$

for every channel $k$, where $x_{t,k}$ is the value of channel $k$ in update $t$. $\Delta t$ and $\tau_k$ are both in seconds. On a device $\Delta t$ is the 1 s observation window (`dt_window_s=1.0`); the accelerated fleet simulator of Section 4 advances the same recursion once per day on daily feature vectors, with the daily decay described at the end of this section.

The design choice that matters here is that $\tau$ is **per channel**, and each $\tau_k$ is set to the horizon of the failure mode that channel drives:

| Channel group | $\tau$ | Rationale |
|---|---|---|
| `transient_pu`, `inrush_count` | 1 day (86,400 s) | Relay contact erosion, arc events |
| `i_crest`, `sag_dwell`, `swell_dwell` | 7 days (604,800 s) | Terminal-block hot-spot cycling |
| `v_rms`, `i_rms`, `flicker_pst`, `freq_dev`, `unbalance` | 30 days (2,592,000 s) | Electrolytic dry-out, Joule heating |
| `v_thd`, `i_thd` | 60 days (5,184,000 s) | SMPS and capacitor stress |

A channel whose damage mechanism is fast gets a short memory; a slow mechanism gets a long one. This is the part of the construction least likely to be arrived at by default.

The reference implementation evaluates the recursion in float64 as $S_{t,k} = S_{t-1,k} + \gamma_k (g_k(x_{t,k}) - S_{t-1,k})$ with the gain $\gamma_k = 1 - \alpha_k = -\mathrm{expm1}(-\Delta t / \tau_k)$. At $\Delta t = 1$ s the gain is $1.2 \times 10^{-5}$ for the 1-day channels and $1.9 \times 10^{-7}$ for the 60-day channels, so it must not be formed as $1 - \alpha$ or stored in float32, which would change the effective time constants by tens of percent. The fleet simulator advances the state once per day with the decay $\alpha_k^{\mathrm{day}} = e^{-86400 / \tau_k}$, which is exactly what 86,400 one-second updates with a constant input produce, so every channel keeps its nominal time constant of the table above.

---

## 3. Health and life estimation

### 3.1 Why the state alone is insufficient

An EWMA is a *level* estimator. Given stationary input it converges to the mean dose in a few multiples of $\tau$ and then stays there. After the first few months of service, two devices in the same environment have nearly identical `S` regardless of whether one is three years old and the other fifteen.

Accumulated fatigue damage, by contrast, is a monotone integral (Miner's rule). Let $d_k(t)$ be the damage dose of channel $k$ at time $t$ (on the device $d_k = g_k(x_k)$) and $w_{ck} \ge 0$ the weight of channel $k$ in the wear-out of component $c$ (the physics-of-failure weights of `FAILURE_MODES`, which sum to 1 for each component):

$$
L_c(T) = \int_0^{T} \sum_k w_{ck}\thinspace d_k(t)\thinspace \mathrm{d}t
$$

`S` and `L` are different functionals of the same input stream, and no choice of weights recovers one from the other. `S` carries the *rate*; the missing factor is exposure *duration*.

The fix is to supply the device's own elapsed operating age — which every device of this class already tracks — and let the estimator form the dose-time product. With $a$ the device age in years (run-days divided by 365) and $\phi(S, a) = s \cdot \mathrm{concat}(S, aS)$, the 24 inputs made of the 12 values of $S$ followed by the 12 values of $aS$ and scaled by $s = 10^{-3}$, the health score of component $c$ is:

$$
\mathrm{CHS}_c = \mathrm{clip}\left(1 - W_c^{\top} \phi(S, a) - b_c, 0, 1\right)
$$

where $\mathrm{clip}(z, 0, 1) = \min(\max(z, 0), 1)$, $W_c \in \mathbb{R}^{24}$ and $b_c \in \mathbb{R}$ are the weights and bias of component $c$, and a score of 1 is factory-new while 0 is end of life.

Section 5 shows this is worth a 4.09x reduction in CHS error.

### 3.2 Calibration

$W_c$ and $b_c$ are obtained by closed-form ridge regression of the observed damage $1 - \mathrm{CHS}^{\mathrm{obs}}$ on $\phi$; the clip is ignored during the fit and re-applied when scoring:

$$
\min_{W_c, b_c} \sum_n \left(1 - \mathrm{CHS}^{\mathrm{obs}}_{n,c} - W_c^{\top} \phi_n - b_c\right)^2 + \rho \left(\Vert W_c \Vert^2 + b_c^2\right)
$$

where $n$ runs over the training snapshots and the penalty $\rho = 10^{-3}$ applies to the bias as well as the weights. Without fitting, `ComponentHealthScorer()` uses the physics-of-failure prior: the first 12 entries of $W_c$ are the weights $w_{ck}$ of Section 3.1, the last 12 are zero and $b_c = 0$. In deployment the labels come from accelerated-life testing per hardware revision; in this repository they come from the simulator.

### 3.3 Remaining useful life

RUL regresses days-to-failure on $S$, the five component scores $\mathrm{CHS}$ and the age $a$. Two heads: a ridge model small enough to evaluate on the device, and a gradient-boosting model as a cloud-side reference for the accuracy being given up. The ridge head is linear in the 18 inputs plus an intercept, so it has 19 coefficients $\beta$, and its output is clamped at zero:

$$
\begin{aligned}
\widehat{\mathrm{RUL}} &= \max\left(0, \beta^{\top} \tilde{z}\right) \cr
\tilde{z} &= (S, \mathrm{CHS}, a, 1) \in \mathbb{R}^{19}
\end{aligned}
$$

$\beta$ minimises $\sum_n (y_n - \beta^{\top} \tilde z_n)^2 + \rho \Vert \beta \Vert^2$ over the training snapshots, with the intercept penalised and $\rho = 10$ in `pqdna.demo`. The target $y_n$ is the number of days from snapshot $n$ to the first weekly snapshot at which any component's CHS falls below 0.05 (a failure); for a meter that never fails within the horizon it is the number of days to the end of the horizon, a right-censored value used as observed. In `pqdna.demo` the CHS inputs are the simulator's ground-truth scores, for fitting and for evaluation alike; on a device they would be the outputs of the scorer of Section 3.1.

### 3.4 Fixed-point export

`pqdna.edge_export` quantises the trained model to integers and writes a C header holding the constants and a C99 reference implementation of the arithmetic below (`pqdna_update`, `pqdna_chs`, `pqdna_rul_days`). `FixedPointModel` evaluates the same integer arithmetic in Python, bit for bit, and `tests/test_edge_export.py` compiles the header and compares the two on random vectors whenever a C compiler is available. A Q$m$.$n$ value is a two's-complement integer that holds $x \cdot 2^{n}$. All rounding is half up, $\mathrm{rnd}(x) = \lfloor x + 1/2 \rfloor$, and $\mathrm{sat}$ saturates to the `int32` range.

| Quantity | Type | Format |
|---|---|---|
| state $S_k$ and kernel output $g_k$ (both non-negative) | `int64` | Q15.48 |
| decay gain $\gamma_k = 1 - \alpha_k$ | `uint32` | Q0.32 |
| working values: $S$, CHS and the age $a$ | `int32` | Q15.16 |
| CHS weights $sW$ and RUL coefficients $\beta$ | `int32` | mantissa $m$ (magnitude at most $2^{23}$) and one exponent $E$ per block, value $m \cdot 2^{E-23}$ |
| CHS bias $b$ | `int32` | Q15.16 |
| run-day counter $n$ | `uint32` | whole days, $a = n / 365$ |

The exponent of a block is the smallest integer $E$ for which every value of the block is below $2^{E}$ in magnitude, so a mantissa never saturates; there is one exponent for each of the five rows of $sW$ and one for $\beta$. The scale $s$ is folded into the weights, which then act directly on $(S, aS)$. With $\bar b_c = \mathrm{rnd}(2^{16} b_c)$ the quantised bias, one update of the state, with the kernel outputs $g_k$ supplied by the firmware in the same Q15.48 format as $S_k$, is

$$
S_k \leftarrow S_k + \mathrm{rnd}\left(\frac{\gamma_k \thinspace (g_k - S_k)}{2^{32}}\right)
$$

(formed from the two 32-bit halves of $g_k - S_k$, so no 128-bit type is needed). The health scores and the remaining useful life follow from the working values

$$
\begin{aligned}
\bar S_k &= \mathrm{sat}\left(\mathrm{rnd}\left(S_k / 2^{32}\right)\right) \cr
\bar a &= \mathrm{sat}\left(\mathrm{rnd}\left(2^{16} n / 365\right)\right) \cr
\bar\phi_k &= \bar S_k, \qquad \bar\phi_{12+k} = \mathrm{sat}\left(\mathrm{rnd}\left(\bar a \bar S_k / 2^{16}\right)\right)
\end{aligned}
$$

as

$$
\begin{aligned}
\bar d_c &= \mathrm{rnd}\left(\frac{\sum_{j=0}^{23} m_{cj} \bar\phi_j}{2^{23 - E_c}}\right) + \bar b_c \cr
\overline{\mathrm{CHS}}_c &= \min\left(\max\left(2^{16} - \bar d_c, 0\right), 2^{16}\right) \cr
\overline{\mathrm{RUL}} &= \max\left(0, \mathrm{rnd}\left(\frac{\sum_{j=0}^{17} m_j \bar z_j + 2^{16} m_{18}}{2^{39 - E}}\right)\right)
\end{aligned}
$$

where $m_{cj}$ and $E_c$ are the mantissas and the exponent of row $c$ of $sW$, $m_j$ and $E$ those of $\beta$ with $m_{18}$ the intercept, and $\bar z = (\bar S, \overline{\mathrm{CHS}}, \bar a)$. Without the age term the sums have 12 and 17 terms.

The widths follow from the numbers. At $\Delta t = 1$ s the gain of a 60-day channel is $1.9 \times 10^{-7}$, so a state increment can be smaller than one unit in the last place of a 32-bit state, which would then stall; the state therefore has 48 fractional bits. A 16-bit $\alpha$ cannot represent such a decay at all (in Q15 every one of the twelve saturates at 32767, an effective time constant of about 9 hours), so the 32-bit gain $\gamma_k$ is stored instead; it reproduces the time constants to within 0.06% (the 60-day channels become 59.96 days). The fitted coefficients span several decades: the RUL coefficients reach thousands of days, the CHS weights are below 0.13 in magnitude, and the weights of one row differ by up to four orders of magnitude. A single binary point cannot hold that range at useful precision, hence the block exponents. Mantissas are at most $2^{23}$ and working values are 32-bit, so a dot product of at most 24 terms stays below $2^{59}$ and cannot overflow an `int64`. The exporter raises an error instead of saturating a value that does not fit.

On the 4339 held-out snapshots that `pqdna.demo` samples (every tenth test row), the integer CHS differs from the float scorer by at most $2.0 \times 10^{-5}$ and the integer RUL from the float ridge head by at most 0.63 days (`out/fixed_point_check.json`), against test errors of 0.0170 in CHS and 145 days in RUL. The kernels $g_k$ are not part of the export.

---

## 4. Evaluation design

### 4.1 The circularity problem

The first version of this project evaluated CHS against a synthetic ground truth defined as

$$
\mathrm{CHS}^{\text{true}}_c = \mathrm{clip}\left(1 - \frac{\lambda}{r_c} w_c^{\top} S, 0, 1\right)
$$

where $\lambda = 5.5 \times 10^{-3}$, $w_c \in \mathbb{R}^{12}$ is the physics-of-failure weight vector of component $c$ (the $w_{ck}$ of Section 3.1), $S$ is the noise-free device state and $r_c$ is the per-meter robustness multiplier of component $c$ (1 is nominal; drawn from a normal distribution with mean 1 and standard deviation 0.12, clipped to $[0.5, 1.5]$). This uses the *same* weights $w_c$ the estimator is initialised with (the first 12 entries of $W_c$) and then fitted. The estimator's own S-only form is $\mathrm{clip}(1 - s W_c^{\top} S - b_c, 0, 1)$ with $W_c \in \mathbb{R}^{12}$, which differs from the generator only by a rescaling of the weights, a bias and the per-meter factor $1/r_c$. Fitting a linear model to data generated by a linear model with one nuisance scalar is a self-consistency check, not a measurement. It reported CHS MAE of 0.002–0.011 and a "7–9x improvement over baseline" that was an artefact of the setup.

That generator is retained as `generator="matched"` **solely** so the size of the artefact can be quantified. It must never be used for headline numbers.

### 4.2 The mismatched generator

The default generator (`generator="mismatched"`, in `pqdna.synth`) differs from the estimator along four independent axes, each corresponding to a real property of component degradation:

1. **Different damage exponents.** The latent dose of channel $k$ is $\delta_k = G_k \tilde x_k^{p_k}$, with the gains $G_k$ and exponents $p_k$ of `LATENT_DOSE_GAIN` and `LATENT_DOSE_EXP`, where $\tilde x_0 = \vert x_0 - 1 \vert$, $\tilde x_4 = \max(x_4 - 1.414, 0)$ and $\tilde x_k = \max(x_k, 0)$ otherwise — e.g. $p_1 = 2.6$ for `i_rms` against the device's exponent 2, and $p_5 = 3.1$ for `transient_pu` against 2.5. The estimator cannot recover the latent exponents from `S`.
2. **Cumulative, not exponentially-weighted.** Latent damage is the daily sum $L_c = \sum_t \sum_k w_{ck} \delta_k(t)$, monotone in time. This is the mismatch that matters most, and the reason the age feature is required.
3. **Non-linear health link.** $D_c = u_c^{m_c}$ with $u_c = L_c / (L_{\mathrm{ref},c} r_c)$, where $L_{\mathrm{ref},c}$ is the characteristic life (below), $r_c$ the robustness multiplier of Section 4.1 and $m_c \in [1.3, 1.7]$ a per-component Weibull-like shape exponent. Health is $\mathrm{CHS}_c = \exp(-\tilde D_c)$, where $\tilde D_c$ is $D_c$ plus the coupling term of the next item, against the estimator's `clip(1 - linear)`.
4. **Cross-component coupling.** A degrading SMPS accelerates MCU and display aging; relay and terminal-block wear are mutually reinforcing. Formally $\tilde D_c = D_c + \kappa_c D_{\mathrm{src}(c)}$, where $\mathrm{src}(c)$ is the source component of $c$, $\kappa_c$ the coupling gain and $D_{\mathrm{src}(c)}$ the source's uncoupled damage: `mcu` ← `smps` with $\kappa = 0.35$, `display` ← `smps` 0.20, `relay` ← `terminal_block` 0.25 and `terminal_block` ← `relay` 0.15 (`LATENT_COUPLING`). Components without a source have $\tilde D_c = D_c$. The estimator scores components independently and structurally cannot represent this.

In addition, multiplicative sensor noise is applied to every snapshot of the device-visible `S`: each value is multiplied by an independent draw from a normal distribution with mean 1 and standard deviation 0.04 (`obs_noise`, floored at 0), so the estimator does not even observe the trajectory that drove the latent process.

The characteristic life of component $c$ is $L_{\mathrm{ref},c} = 365 Y_c \bar d_c$, where $\bar d_c = \sum_k w_{ck} \delta_k(\bar{x})$ is the daily latent dose at the mean feature vector $\bar{x}$ of the `urban_residential` environment and $Y_c$ (105 to 135 years, `LATENT_LIFE_URBAN_YEARS`) is the anchor, so that a meter with $r_c = 1$ exposed to $\bar{x}$ for $Y_c$ years reaches $u_c = 1$, that is $\mathrm{CHS}_c = 1/e$ before coupling. The anchors are calibrated so that the demo fleet (300 devices, seed 42) has a 13.7% cumulative failure rate over 15 years, an average of about 0.9% of the fleet per year. That rate is an illustrative choice for the demonstration, not a field statistic.

### 4.3 Protocol

- 300 devices, 15-year horizon (5,475 days), weekly snapshots (every 7 days).
- The state is advanced once per day on that day's expected feature vector, with the daily decay $e^{-86400 / \tau_k}$ of Section 2.3; each snapshot records the state, the ground-truth CHS and the age ($a$ = day / 365).
- A device fails at the first weekly snapshot at which any component's CHS is below 0.05; its history is truncated one snapshot later.
- Environment mix: 20% clean grid, 45% urban residential, 20% EV corridor, 15% industrial.
- **Meter-disjoint** 80/20 split (the first 240 devices train, the last 60 test) — no device contributes rows to both train and test, so temporal autocorrelation within a trajectory cannot leak.
- Baseline is the per-component mean CHS of the training rows, evaluated on test rows.
- CHS MAE is the mean absolute error over test rows, per component; "overall" is the mean of the five. Skill is baseline MAE divided by CHS MAE. RUL RMSE and MAE are in days over test rows.

### 4.4 Regression guards

`tests/test_pqdna.py` asserts that the latent dose kernel is *not* numerically close to the device kernel, that ground-truth trajectories are monotone, that the age feature strictly helps, that the matched generator is strictly easier than the mismatched one, and that observation noise is actually applied. These exist to stop the circularity from silently returning. It also asserts that the aggregator's gain and the simulator's daily decay give every channel its nominal time constant. `tests/test_edge_export.py` asserts that the exported constants never saturate, that the integer arithmetic of Section 3.4 matches the float model, and, when a C compiler is available, that the C code in the generated header matches the Python integer model bit for bit.

---

## 5. Results

See [../README.md](../README.md#results) for the full tables. Summary:

- CHS MAE **0.0170** overall against the mismatched generator, 7.9x better than a mean predictor.
- Removing the age feature degrades this to **0.0695** — a 4.09x penalty.
- Under the matched generator the age feature is worth nothing (0.0044 → 0.0043), demonstrating that the original evaluation design was incapable of revealing the flaw.
- RUL ridge RMSE **145 days** against a mean-predictor RMSE of 1595 days, with ground-truth CHS as an input (Section 3.3). The cloud-side GBR reaches 24 days, which bounds what is being traded away for on-device execution.
- Total exported size: **730 bytes** of model constants and state (see the footprint table in the README for the breakdown). The integer model of Section 3.4 agrees with the float model to within $2.0 \times 10^{-5}$ in CHS and 0.63 days in RUL.

---

## 6. What this does not establish

The evaluation removes circularity. It does not establish real-world accuracy, and nothing here should be read as a performance claim about physical hardware.

Specifically unvalidated: the damage kernel exponents, the physics-of-failure weight matrix, the decay time constants, the correspondence between the simulator's per-day feature vectors and the extractor's per-window features (the simulator's `inrush_count`, `sag_dwell`, `swell_dwell` and `transient_pu` are per-day counts, scaled event counts and daily maxima, whereas the extractor defines them per 1 s window), and the entire premise that the 12 chosen channels span the failure-relevant subspace. Each requires either accelerated-life-test data or field data with confirmed component-level failure attribution. The fixed-point export is checked against the float model (Section 3.4) but has not been run on target hardware.

The most useful contribution anyone could make to this project is a dataset that pairs power-quality recordings with confirmed component replacements. See [../CONTRIBUTING.md](../CONTRIBUTING.md).
