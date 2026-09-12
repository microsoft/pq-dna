"""End-to-end PQ-DNA demo.

Run with:
    python -m pqdna.demo

Produces:
  * `out/feature_importance.csv`
  * `out/chs_fit_metrics.csv`
  * `out/rul_metrics.csv`
  * `out/pqdna_model.h`           (fixed-point firmware header)
  * `out/footprint.json`
  * `out/sample_meter_trace.csv`  (a single meter's PQ-DNA + CHS trajectory)
"""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import (
    ComponentHealthScorer,
    PQDNAAggregator,
    PQDNAState,
    PQFeatureExtractor,
    RULEstimator,
    SyntheticMeterFleet,
)
from .chs import COMPONENTS
from .edge_export import export_c_header, memory_footprint_bytes
from .features import FEATURE_NAMES

OUT = Path("out")
OUT.mkdir(exist_ok=True)


def _split_by_meter(traces, frac_train: float = 0.8):
    """Boolean row mask for a *meter-disjoint* train/test split."""
    cut = int(frac_train * len(traces))
    train_ids = {t.meter_id for t in traces[:cut]}
    mask = []
    for t in traces:
        mask.extend([t.meter_id in train_ids] * len(t.times_days))
    return np.array(mask)


def _chs_mae(scorer, S, age_years, CHS_true):
    pred = scorer.score_batch(S, age_years if scorer.use_age else 0.0)
    return np.mean(np.abs(pred - CHS_true), axis=0)


def _evaluate(generator: str, use_age: bool, n_meters=300, years=15.0, seed=42):
    """Fit and score one (generator, feature-set) configuration."""
    fleet = SyntheticMeterFleet(n_meters=n_meters, years=years, seed=seed,
                                generator=generator)
    traces = fleet.simulate()
    S, CHS, ttf, age_days = SyntheticMeterFleet.to_arrays(traces)
    age_y = age_days / 365.0
    tr = _split_by_meter(traces)

    scorer = ComponentHealthScorer(use_age=use_age)
    scorer.fit(S[tr], CHS[tr], age_y[tr] if use_age else None)
    mae = _chs_mae(scorer, S[~tr], age_y[~tr], CHS[~tr])

    # Honest baseline: predict the training-set mean CHS for every snapshot.
    train_mean = CHS[tr].mean(axis=0)
    base = np.mean(np.abs(CHS[~tr] - train_mean[None, :]), axis=0)

    return {
        "traces": traces, "S": S, "CHS": CHS, "ttf": ttf, "age_y": age_y,
        "train": tr, "scorer": scorer, "mae": mae, "baseline": base,
    }


def main() -> None:
    print("[1/7] Simulating fleet (mismatched generator) ...")
    fleet = SyntheticMeterFleet(n_meters=300, years=15.0, seed=42,
                                generator="mismatched")
    traces = fleet.simulate()
    n_failed = sum(1 for t in traces if t.failed_component is not None)
    print(f"      {len(traces)} meters simulated, {n_failed} "
          f"({100 * n_failed / len(traces):.1f}%) failed within 15 y")

    S, CHS, ttf, age_days = SyntheticMeterFleet.to_arrays(traces)
    age_y = age_days / 365.0
    print(f"      training matrix: S={S.shape}, CHS={CHS.shape}, ttf={ttf.shape}")

    train_mask = _split_by_meter(traces)
    print(f"      train rows: {train_mask.sum()}, test rows: {(~train_mask).sum()}"
          f"  (meter-disjoint split)")

    print("[2/7] Fitting Component Health Scorer (S + age) ...")
    scorer = ComponentHealthScorer(use_age=True)
    scorer.fit(S[train_mask], CHS[train_mask], age_y[train_mask])
    mae = _chs_mae(scorer, S[~train_mask], age_y[~train_mask], CHS[~train_mask])
    train_mean = CHS[train_mask].mean(axis=0)
    baseline = np.mean(np.abs(CHS[~train_mask] - train_mean[None, :]), axis=0)
    print("      per-component CHS MAE:  ",
          {c: round(float(v), 4) for c, v in zip(COMPONENTS, mae)})
    print("      mean-predictor baseline:",
          {c: round(float(v), 4) for c, v in zip(COMPONENTS, baseline)})

    with open(OUT / "chs_fit_metrics.csv", "w", encoding="utf-8") as fh:
        fh.write("component,mae,baseline_mean,skill_ratio\n")
        for ci, c in enumerate(COMPONENTS):
            fh.write(f"{c},{mae[ci]:.4f},{baseline[ci]:.4f},"
                     f"{baseline[ci] / max(mae[ci], 1e-9):.2f}\n")

    # Feature importance over the instantaneous-stress block of W.
    importance = np.linalg.norm(scorer.W[:, :len(FEATURE_NAMES)], axis=0)
    with open(OUT / "feature_importance.csv", "w", encoding="utf-8") as fh:
        fh.write("feature,importance\n")
        for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
            fh.write(f"{name},{imp:.4f}\n")

    print("[3/7] Ablations ...")
    ablations = []
    for gen in ("mismatched", "matched"):
        for use_age in (True, False):
            r = _evaluate(gen, use_age)
            overall = float(np.mean(r["mae"]))
            base = float(np.mean(r["baseline"]))
            ablations.append((gen, use_age, overall, base))
            print(f"      generator={gen:<11} age={str(use_age):<5} "
                  f"CHS MAE={overall:.4f}  baseline={base:.4f}  "
                  f"skill={base / max(overall, 1e-9):.2f}x")
    with open(OUT / "ablation_metrics.csv", "w", encoding="utf-8") as fh:
        fh.write("generator,use_age,chs_mae,baseline_mae,skill_ratio\n")
        for gen, ua, ov, bs in ablations:
            fh.write(f"{gen},{ua},{ov:.4f},{bs:.4f},{bs / max(ov, 1e-9):.3f}\n")

    print("[4/7] Fitting Remaining-Useful-Life regressor (ridge head) ...")
    rul = RULEstimator(model="ridge", l2=10.0)
    rul.fit(S[train_mask], CHS[train_mask], ttf[train_mask], age_y[train_mask])
    yhat = rul.predict_days(S[~train_mask], CHS[~train_mask], age_y[~train_mask])
    y_true = ttf[~train_mask]
    rmse = float(np.sqrt(np.mean((yhat - y_true) ** 2)))
    mae_d = float(np.mean(np.abs(yhat - y_true)))
    rmse_base = float(np.sqrt(np.mean((ttf[train_mask].mean() - y_true) ** 2)))
    print(f"      RUL ridge RMSE = {rmse:.1f} d, MAE = {mae_d:.1f} d "
          f"(mean-predictor RMSE = {rmse_base:.1f} d)")

    try:
        rul_gbr = RULEstimator(model="gbr").fit(
            S[train_mask], CHS[train_mask], ttf[train_mask], age_y[train_mask]
        )
        yhat_gbr = rul_gbr.predict_days(S[~train_mask], CHS[~train_mask], age_y[~train_mask])
        rmse_gbr = float(np.sqrt(np.mean((yhat_gbr - y_true) ** 2)))
        mae_gbr = float(np.mean(np.abs(yhat_gbr - y_true)))
        print(f"      RUL GBR   RMSE = {rmse_gbr:.1f} d, MAE = {mae_gbr:.1f} d")
    except Exception as e:                              # pragma: no cover
        rmse_gbr, mae_gbr = None, None
        print(f"      (GBR head skipped: {e})")

    with open(OUT / "rul_metrics.csv", "w", encoding="utf-8") as fh:
        fh.write("model,rmse_days,mae_days\n")
        fh.write(f"ridge,{rmse:.2f},{mae_d:.2f}\n")
        if rmse_gbr is not None:
            fh.write(f"gbr,{rmse_gbr:.2f},{mae_gbr:.2f}\n")
        fh.write(f"mean_predictor,{rmse_base:.2f},"
                 f"{float(np.mean(np.abs(ttf[train_mask].mean() - y_true))):.2f}\n")

    print("[5/7] Exporting fixed-point firmware header ...")
    agg = PQDNAAggregator(dt_window_s=1.0)
    out_h = export_c_header(str(OUT / "pqdna_model.h"), agg, scorer, rul)
    foot = memory_footprint_bytes(agg, scorer, rul)
    with open(OUT / "footprint.json", "w", encoding="utf-8") as fh:
        json.dump(foot, fh, indent=2)
    print(f"      wrote {out_h}, footprint: {foot['total_bytes']} bytes total")

    print("[6/7] Sample meter trajectory ...")
    sample = next((t for t in traces if t.failed_component is not None), traces[0])
    with open(OUT / "sample_meter_trace.csv", "w", encoding="utf-8") as fh:
        cols = ["day"] + [f"S_{n}" for n in FEATURE_NAMES] + [f"chs_{c}" for c in COMPONENTS]
        fh.write(",".join(cols) + "\n")
        for i, day in enumerate(sample.times_days):
            row = [f"{float(day):.0f}"]
            row += [f"{v:.5f}" for v in sample.S_history[i]]
            row += [f"{v:.5f}" for v in sample.chs_history[i]]
            fh.write(",".join(row) + "\n")
    print(f"      sample meter id={sample.meter_id}, "
          f"env={sample.env.env_name}, "
          f"failed={sample.failed_component} on day {sample.failure_day}")

    print("[7/7] One-shot live demo of full pipeline on a 1-second waveform ...")
    fs = 7680
    line_hz = 60.0
    n = fs
    t = np.arange(n) / fs
    # Synthesise a noisy, distorted, transient-rich voltage and current.
    rng = np.random.default_rng(0)
    v = 230.0 * np.sqrt(2) * np.sin(2 * np.pi * line_hz * t)
    v += 0.05 * 230.0 * np.sqrt(2) * np.sin(2 * np.pi * 5 * line_hz * t)
    v += rng.normal(0, 0.5, n)
    # add a transient
    v[300:312] += 80.0
    i = 30.0 * np.sqrt(2) * np.sin(2 * np.pi * line_hz * t - 0.2)
    i += 0.15 * 30.0 * np.sqrt(2) * np.sin(2 * np.pi * 3 * line_hz * t)
    ext = PQFeatureExtractor(fs=fs, line_hz=line_hz)
    feats = ext.extract(v, i)
    print("      extracted features:",
          {k: round(float(val), 4) for k, val in feats.__dict__.items()})
    state = PQDNAState()
    agg.update(state, feats)
    print("      one-update PQ-DNA L2 norm:", float(np.linalg.norm(state.S)))

    print("\nDone. Artifacts in:", OUT.resolve())


if __name__ == "__main__":
    main()
