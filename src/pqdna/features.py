"""Per-cycle power-quality feature extraction from waveform samples.

These are the features the meter MCU computes once per `WINDOW_CYCLES` mains
cycles (e.g. once per second at 60 cycles @ 60 Hz). They are deliberately
restricted to operations a fixed-point DSP can do in O(N log N) on an
ARM-Cortex-M4-class core: rFFT, abs, RMS, simple thresholding.
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Default measurement parameters (typical revenue-class metrology front-end).
DEFAULT_FS = 7680      # samples / s  (128 samples / cycle @ 60 Hz)
DEFAULT_LINE_HZ = 60.0
WINDOW_CYCLES = 60     # 1-second observation window


@dataclass
class PQFeatures:
    """One window of power-quality features. Order MUST match `FEATURE_NAMES`."""
    v_rms: float          # voltage RMS (per-unit; 1.0 = nominal)
    i_rms: float          # current RMS (per-unit; 1.0 = service-rating)
    v_thd: float          # voltage total harmonic distortion (fraction, e.g. 0.05)
    i_thd: float          # current THD (fraction)
    i_crest: float        # current crest factor = peak / RMS
    transient_pu: float   # peak transient excursion |Δv|/V_nom in the window
    sag_dwell: float      # fraction of window with v_rms < 0.9 pu
    swell_dwell: float    # fraction of window with v_rms > 1.1 pu
    flicker_pst: float    # short-term flicker index (Pst, IEC 61000-4-15-like)
    freq_dev: float       # |f - f_nom| / f_nom
    unbalance: float      # voltage unbalance (single-phase: 0.0)
    inrush_count: float   # # of |i| > 4 pu transitions per window

    def as_vector(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float32)


FEATURE_NAMES = list(PQFeatures.__annotations__.keys())
N_FEATURES = len(FEATURE_NAMES)


class PQFeatureExtractor:
    """Extracts a `PQFeatures` vector from one window of voltage / current samples.

    Designed to be MCU-friendly: 1× rFFT of length `WINDOW_CYCLES * samples_per_cycle`,
    plus O(N) sample-domain stats. No floating-point library calls beyond sqrt.
    """

    def __init__(
        self,
        fs: int = DEFAULT_FS,
        line_hz: float = DEFAULT_LINE_HZ,
        window_cycles: int = WINDOW_CYCLES,
        v_nominal: float = 230.0,
        i_rating: float = 100.0,
    ) -> None:
        self.fs = fs
        self.line_hz = line_hz
        self.window_cycles = window_cycles
        self.samples_per_cycle = int(round(fs / line_hz))
        self.window_samples = self.samples_per_cycle * window_cycles
        self.v_nominal = v_nominal
        self.i_rating = i_rating

    # ---- core feature ops ---------------------------------------------------
    def _thd(self, x: np.ndarray) -> float:
        """THD computed against the fundamental bin of an rFFT."""
        n = len(x)
        X = np.fft.rfft(x * np.hanning(n))
        mag = np.abs(X)
        # Window-aware fundamental index.
        f_axis = np.fft.rfftfreq(n, d=1.0 / self.fs)
        k = int(np.argmin(np.abs(f_axis - self.line_hz)))
        fund = mag[k]
        if fund <= 1e-9:
            return 0.0
        # sum harmonics 2..15.
        h = 0.0
        for m in range(2, 16):
            idx = m * k
            if idx < len(mag):
                h += mag[idx] ** 2
        return float(np.sqrt(h) / fund)

    def _flicker_pst(self, v: np.ndarray) -> float:
        """Lightweight Pst proxy: std of 10-Hz-band envelope, scaled."""
        # bandpass via diff of two moving averages.
        n = len(v)
        win_a = max(1, n // 768)   # ~10 Hz
        win_b = max(1, n // 96)    # ~0.6 Hz
        a = np.convolve(v, np.ones(win_a) / win_a, mode="same")
        b = np.convolve(v, np.ones(win_b) / win_b, mode="same")
        env = np.abs(a - b)
        return float(np.std(env) / self.v_nominal * 100.0)

    # ---- public -------------------------------------------------------------
    def extract(self, v: np.ndarray, i: np.ndarray) -> PQFeatures:
        if len(v) != self.window_samples or len(i) != self.window_samples:
            raise ValueError(
                f"Expected {self.window_samples}-sample window, got v={len(v)}, i={len(i)}"
            )

        v_rms = float(np.sqrt(np.mean(v ** 2)))
        i_rms = float(np.sqrt(np.mean(i ** 2)))

        # per-cycle RMS for sag/swell dwell and transient detection.
        spc = self.samples_per_cycle
        v_cyc = v[: spc * self.window_cycles].reshape(self.window_cycles, spc)
        v_cyc_rms = np.sqrt(np.mean(v_cyc ** 2, axis=1)) / self.v_nominal
        sag_dwell = float(np.mean(v_cyc_rms < 0.9))
        swell_dwell = float(np.mean(v_cyc_rms > 1.1))

        # transient: max sample-to-sample jump normalised by Vnom.
        transient_pu = float(np.max(np.abs(np.diff(v))) / self.v_nominal)

        # crest factor of current.
        i_peak = float(np.max(np.abs(i)))
        i_crest = i_peak / max(i_rms, 1e-9)

        # THD.
        v_thd = self._thd(v)
        i_thd = self._thd(i)

        # frequency estimate via zero-crossings of v.
        zc = np.where(np.diff(np.signbit(v)))[0]
        if len(zc) >= 2:
            period_samples = np.mean(np.diff(zc)) * 2.0
            f_meas = self.fs / max(period_samples, 1.0)
        else:
            f_meas = self.line_hz
        freq_dev = abs(f_meas - self.line_hz) / self.line_hz

        # inrush count: |i| crossings of 4× rating within the window.
        thr = 4.0 * self.i_rating
        inrush = int(np.sum((np.abs(i[:-1]) <= thr) & (np.abs(i[1:]) > thr)))

        pst = self._flicker_pst(v)

        return PQFeatures(
            v_rms=v_rms / self.v_nominal,
            i_rms=i_rms / self.i_rating,
            v_thd=v_thd,
            i_thd=i_thd,
            i_crest=i_crest,
            transient_pu=transient_pu,
            sag_dwell=sag_dwell,
            swell_dwell=swell_dwell,
            flicker_pst=pst,
            freq_dev=freq_dev,
            unbalance=0.0,
            inrush_count=float(inrush),
        )
