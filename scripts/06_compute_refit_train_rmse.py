#!/usr/bin/env python3
"""Compute refit-train RMSE for every (fold, lead, model) with a saved model,
and write it back as a new `train_rmse` column in `fold_runs.csv`.

The pipeline's Protocol A refits each model on the contiguous train+val slice
*before* evaluating on test. To diagnose memorisation vs generalisation we
need both train RMSE (model evaluated on the data it was actually fit on)
and test RMSE (already in `metrics_allLeads.csv`). The val RMSE that
`fold_runs.csv::best_val_score` already carries is informative but reflects
a different (smaller) model — so train RMSE on the refit slice is a cleaner
overfitting probe.

Constraints:
  - Only works for experiments whose pipeline had `save_models: true`. The
    fast path simply loads each pickled model. There is no slow refit path
    here — refitting all (fold, lead, model) combinations just to recover a
    train RMSE is too expensive to be a routine diagnostic.
  - RMSE is computed on the WINTER target months of the refit slice (to make
    it directly comparable to the test winter pool). For v4 / pruned-winter
    runs the refit slice IS already winter-filtered for ML models, but
    baselines use the unfiltered refit_full slice — we filter in software so
    the column has a single consistent meaning across model families.

Usage:
    python scripts/06_compute_refit_train_rmse.py \\
        --exp-config configs/experiments/exp_winter-training.yaml \\
        [--models climatology,persistence,ar,ols,ridge,lasso,elasticnet,rf,xgboost]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from droughtmodel.pipeline import ExperimentRunner, _BASELINE_MODELS


def _winter_target_rmse(model, slice_ds, lead: int) -> float:
    """RMSE of `model.predict(slice_ds)` against `slice_ds.target`, restricted
    to rows whose (feature_time + lead).month is in {11,12,1,2}."""
    pred = model.predict(slice_ds)
    pred_arr = pred.values if hasattr(pred, "values") else np.asarray(pred)
    truth_arr = slice_ds["target"].values

    time_idx = pd.DatetimeIndex(slice_ds["time"].values)
    target_months = (time_idx + pd.DateOffset(months=int(lead))).month
    winter_mask_t = np.isin(target_months, [11, 12, 1, 2])

    p = pred_arr[winter_mask_t]
    t = truth_arr[winter_mask_t]
    finite = np.isfinite(p) & np.isfinite(t)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((p[finite] - t[finite]) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-config", required=True,
                    help="Experiment YAML used by the original sweep.")
    ap.add_argument("--prefix", default="",
                    help="File-name prefix used by the original sweep "
                         "(matches exp.file_prefix). Default '' = production.")
    ap.add_argument(
        "--models", default="climatology,persistence,ar,ols,ridge,lasso,elasticnet,rf,xgboost",
        help="Comma-separated model names to compute train RMSE for.",
    )
    args = ap.parse_args()
    prefix = args.prefix
    target_models = [m.strip() for m in args.models.split(",") if m.strip()]

    runner = ExperimentRunner(args.exp_config, verbose=False)
    runner.setup_data()

    fold_runs_path = runner.logs_dir / f"{prefix}fold_runs.csv"
    if not fold_runs_path.exists():
        print(f"ERROR: {fold_runs_path} not found", file=sys.stderr)
        return 1

    if not runner.models_dir.exists() or not any(runner.models_dir.glob("*.joblib")):
        print(f"ERROR: no saved models found under {runner.models_dir}.\n"
              f"This script needs `save_models: true` in the experiment YAML "
              f"for the original sweep.", file=sys.stderr)
        return 1

    fold_runs = pd.read_csv(fold_runs_path)
    if "train_rmse" not in fold_runs.columns:
        fold_runs["train_rmse"] = float("nan")

    leads = sorted(fold_runs["lead"].unique())
    print(f"Computing refit-train RMSE — prefix={prefix!r}")
    print(f"  models dir : {runner.models_dir.relative_to(ROOT)}")
    print(f"  fold_runs  : {fold_runs_path.relative_to(ROOT)}")
    print(f"  targets    : {target_models}")
    print(f"  leads      : {leads}")

    t0 = time.time()
    n_filled = 0

    for lead in leads:
        prep_cache: dict[int, object] = {}
        for fold_spec in runner.cv.fold_specs:
            for name in target_models:
                runs_row_mask = (
                    (fold_runs["lead"] == lead)
                    & (fold_runs["fold"] == fold_spec.index)
                    & (fold_runs["model"] == name)
                )
                if not runs_row_mask.any():
                    continue

                model_path = (
                    runner.models_dir
                    / f"{prefix}{name}_lead{lead}_fold{fold_spec.index}.joblib"
                )
                if not model_path.exists():
                    continue

                if fold_spec.index not in prep_cache:
                    prep_cache[fold_spec.index] = runner._prepare_fold(lead, fold_spec)
                prep = prep_cache[fold_spec.index]

                # Slice the model was actually fit on (matches pipeline._fit_predict)
                slice_used = (
                    prep.refit_full if name in _BASELINE_MODELS else prep.refit
                )

                t_step = time.time()
                model = joblib.load(model_path)
                rmse = _winter_target_rmse(model, slice_used, lead)
                fold_runs.loc[runs_row_mask, "train_rmse"] = rmse
                n_filled += 1
                print(f"  L={lead} fold{fold_spec.index} {name:<12} "
                      f"train_rmse={rmse:7.4f}   ({time.time() - t_step:5.1f}s)",
                      flush=True)

    fold_runs.to_csv(fold_runs_path, index=False)
    print(f"\nwrote {fold_runs_path.relative_to(ROOT)}  "
          f"({n_filled} rows updated)  in {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
