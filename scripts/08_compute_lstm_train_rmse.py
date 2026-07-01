#!/usr/bin/env python3
"""Backfill `train_rmse` for the LSTM run.

Mirror of ``scripts/06_compute_refit_train_rmse.py``, but for the LSTM
pipeline's saved models. For every (fold, lead) row in
``results/lstm/logs/fold_runs.csv`` that has a matching pickled model under
``results/lstm/models/``, compute the RMSE of `predict_tensors(refit_seq)`
against the refit-slice winter targets, and write it back as a new
``train_rmse`` column.

Constraints / scope:
  - Only works if the LSTM experiment was launched with ``save_models: true``.
  - RMSE is computed on the WINTER target months of the refit slice (matches
    the winter-pool test RMSE in ``metrics_allLeads.csv``). Off-season months
    are excluded so the train and test columns are directly comparable.
  - LSTM samples that ended up dropped during fitting due to NaN features
    are dropped here too — there is no slow-path refit.

Usage::

    python scripts/08_compute_lstm_train_rmse.py \
        --exp-config configs/experiments/exp_lstm.yaml
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

from droughtmodel.lstm_pipeline import LSTMExperimentRunner   # noqa: E402


def _winter_train_rmse(model, full_ds_std, refit_slice, runner, T) -> float:
    """RMSE of the LSTM on the refit slice's WINTER target months only.

    Builds sequences the same way the pipeline does at training time (winter
    filter applied iff `winter_only_training`) — except that here we drop the
    winter filter and explicitly recompute the mask so the column always has
    the same meaning regardless of v3/v4."""
    # Build sequences over the full refit window (no winter filter), then we
    # post-filter the targets to winter months for the metric.
    X_train, y_train, meta = runner._build_window_tensors(
        full_ds_std, refit_slice, sequence_length=T, winter_filter=False,
    )
    if len(X_train) == 0:
        return float("nan")

    preds = model.predict_tensors(X_train)

    # Winter mask on TARGET time (feature_time + lead).month ∈ {11,12,1,2}.
    full_times = pd.DatetimeIndex(full_ds_std["time"].values)
    lead = int(refit_slice.attrs.get("lead", 0))
    feat_times = full_times[meta.sample_t_idx]
    target_months = (feat_times + pd.DateOffset(months=lead)).month
    winter = np.isin(target_months, [11, 12, 1, 2])

    p = preds[winter]
    t = y_train[winter]
    finite = np.isfinite(p) & np.isfinite(t)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean((p[finite] - t[finite]) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-config", required=True,
                    help="Experiment YAML used by the original LSTM sweep.")
    ap.add_argument("--prefix", default="",
                    help="File prefix (matches exp.file_prefix). Default ''.")
    args = ap.parse_args()
    prefix = args.prefix

    runner = LSTMExperimentRunner(args.exp_config, verbose=False)
    runner.setup_data()

    fold_runs_path = runner.logs_dir / f"{prefix}fold_runs.csv"
    if not fold_runs_path.exists():
        print(f"ERROR: {fold_runs_path} not found", file=sys.stderr)
        return 1

    if not runner.models_dir.exists() or not any(runner.models_dir.glob("*.joblib")):
        print(f"ERROR: no saved LSTM models under {runner.models_dir}.\n"
              f"This script needs `save_models: true` in the experiment YAML.",
              file=sys.stderr)
        return 1

    fold_runs = pd.read_csv(fold_runs_path)
    if "train_rmse" not in fold_runs.columns:
        fold_runs["train_rmse"] = float("nan")

    leads = sorted(fold_runs["lead"].unique())
    print(f"Computing LSTM refit-train RMSE — prefix={prefix!r}")
    print(f"  models dir : {runner.models_dir.relative_to(ROOT)}")
    print(f"  fold_runs  : {fold_runs_path.relative_to(ROOT)}")
    print(f"  leads      : {leads}")

    t0 = time.time()
    n_filled = 0

    for lead in leads:
        # Pre-build the unfiltered feature dataset once per lead.
        full_ds_unstd = runner._build_full_feature_dataset(lead)

        for fold_spec in runner.cv.fold_specs:
            mask = (
                (fold_runs["lead"] == lead)
                & (fold_runs["fold"] == fold_spec.index)
                & (fold_runs["model"] == "lstm")
            )
            if not mask.any():
                continue
            model_path = (
                runner.models_dir
                / f"{prefix}lstm_lead{lead}_fold{fold_spec.index}.joblib"
            )
            if not model_path.exists():
                continue

            model = joblib.load(model_path)
            T = int(model.sequence_length)
            prep = runner._prepare_fold(lead, fold_spec, sequence_length=T)
            full_ds_std = runner._standardize_full_using_train(full_ds_unstd, prep.train_unstd)

            t_step = time.time()
            rmse = _winter_train_rmse(model, full_ds_std, prep.refit, runner, T)
            fold_runs.loc[mask, "train_rmse"] = rmse
            n_filled += 1
            print(f"  L={lead} fold{fold_spec.index} lstm  train_rmse={rmse:7.4f}   "
                  f"(T={T}, {time.time() - t_step:5.1f}s)", flush=True)

    fold_runs.to_csv(fold_runs_path, index=False)
    print(f"\nwrote {fold_runs_path.relative_to(ROOT)}  "
          f"({n_filled} rows updated)  in {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())