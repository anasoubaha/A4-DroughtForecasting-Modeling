#!/usr/bin/env python3
"""Backfill per-fold metric rows into an existing `metrics_allLeads.csv`.

Use this when the pipeline was run BEFORE the per-fold-metrics change went
in (existing CSV has only `fold='pooled'` rows) and you want to avoid a
full re-run. Reads:

    results/predictions/{prefix}pooled_allLeads.nc
    configs/cv.yaml                                 (for fold time boundaries)
    configs/metrics.yaml                            (for bootstrap config)
    results/metrics/{prefix}metrics_allLeads.csv    (existing pooled rows)

…then writes back the same CSV with per-fold rows appended (one set per
(model, lead, fold) for each evaluation window). Pooled rows are preserved
unchanged. Idempotent: re-running won't double-count per-fold rows.

Usage:
    python scripts/04_backfill_per_fold_metrics.py [--prefix testOptuna_]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from droughtmodel import cv as dcv
from droughtmodel import evaluation as deval
from droughtmodel.utils import RESULTS_DIR


def _fold_time_mask(time_index: pd.DatetimeIndex, fold_spec) -> np.ndarray:
    """Boolean mask over `time_index` selecting the fold's TEST window."""
    start = pd.Timestamp(fold_spec.test_start)
    end = pd.Timestamp(fold_spec.test_end)
    return (time_index >= start) & (time_index <= end)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="",
                    help="File-name prefix (e.g. 'testOptuna_'). Default '' = production.")
    ap.add_argument("--with-bootstrap", action="store_true",
                    help="Compute block-bootstrap CIs on per-fold rows too. "
                         "Slow (~minutes per fold-lead) — usually unnecessary "
                         "since per-fold rows are for stability plots; the "
                         "pooled rows already carry CIs.")
    args = ap.parse_args()
    prefix = args.prefix
    with_bootstrap = bool(args.with_bootstrap)

    preds_path = RESULTS_DIR / "predictions" / f"{prefix}pooled_allLeads.nc"
    csv_path = RESULTS_DIR / "metrics" / f"{prefix}metrics_allLeads.csv"

    for p in (preds_path, csv_path):
        if not p.exists():
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
            return 1

    existing = pd.read_csv(csv_path)
    if not (existing["fold"] == "pooled").all():
        # Drop any pre-existing non-pooled rows so the backfill is idempotent.
        existing = existing[existing["fold"] == "pooled"].copy()
        print(f"  (dropped existing non-pooled rows so backfill is idempotent)")

    ds = xr.open_dataset(preds_path)
    leads = sorted(int(L) for L in ds["lead"].values)
    time_idx = pd.DatetimeIndex(ds["time"].values)

    cv_cfg = dcv.load_cv_config()
    cv = dcv.RollingOriginCV.from_config(cv_cfg)
    metrics_cfg = deval.load_metrics_config()

    pred_vars = [v for v in ds.data_vars if v.startswith("pred_")]
    models = [v[len("pred_"):] for v in pred_vars]
    if "climatology" not in models or "persistence" not in models:
        print("ERROR: climatology and persistence preds required in NetCDF.", file=sys.stderr)
        return 1

    print(f"Backfilling per-fold metrics:")
    print(f"  predictions: {preds_path.relative_to(ROOT)}")
    print(f"  existing CSV: {csv_path.relative_to(ROOT)}  ({len(existing)} pooled rows)")
    print(f"  leads = {leads}")
    print(f"  folds = {[s.index for s in cv.fold_specs]}")
    print(f"  models = {models}")

    t0 = time.time()
    new_rows: list[pd.DataFrame] = []

    for lead in leads:
        truth_full = ds["truth"].sel(lead=lead).values
        clim_full = ds["pred_climatology"].sel(lead=lead).values
        pers_full = ds["pred_persistence"].sel(lead=lead).values

        for fold_spec in cv.fold_specs:
            mask_t = _fold_time_mask(time_idx, fold_spec)
            if not mask_t.any():
                print(f"  skipping L={lead} fold{fold_spec.index} — no overlapping time steps")
                continue
            fold_truth = truth_full[mask_t]
            fold_time = time_idx[mask_t]
            fold_clim = clim_full[mask_t]
            fold_pers = pers_full[mask_t]

            months = fold_time.month
            winter_mask = np.isin(months, [11, 12, 1, 2])

            for window_name, mask, block_key in [
                ("winter_only", winter_mask, "mean_block_length_winter"),
                ("all_months", np.ones_like(months, dtype=bool), "mean_block_length_all"),
            ]:
                if not mask.any():
                    continue
                reporter = deval.MetricsReporter.from_config(metrics_cfg, evaluation_window=window_name)
                reporter.mean_block_length = metrics_cfg["bootstrap"][block_key]
                if not with_bootstrap:
                    reporter.bootstrap = False           # skip CIs → ~10× faster

                y_true = fold_truth[mask]
                y_clim = fold_clim[mask]
                y_pers = fold_pers[mask]

                for model in models:
                    y_pred = ds[f"pred_{model}"].sel(lead=lead).values[mask_t][mask]
                    res = reporter.evaluate(y_pred, y_true, climatology=y_clim, persistence=y_pers)
                    df = deval.MetricsReporter.to_dataframe(
                        res, model=model, lead=lead, fold=fold_spec.index,
                        evaluation_window=window_name,
                    )
                    new_rows.append(df)

            print(f"  L={lead} fold{fold_spec.index} done  "
                  f"({int(mask_t.sum())} months, {int(winter_mask.sum())} winter)")

    if not new_rows:
        print("WARN: no per-fold rows produced.", file=sys.stderr)
        return 0

    new_df = pd.concat(new_rows, ignore_index=True)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Ensure the `fold` column survives the round-trip with consistent dtype
    # (object: 'pooled' for pooled rows, int for per-fold rows).
    combined["fold"] = combined["fold"].astype(object)

    combined.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path.relative_to(ROOT)}  "
          f"({len(existing)} pooled + {len(new_df)} per-fold = {len(combined)} rows)  "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
