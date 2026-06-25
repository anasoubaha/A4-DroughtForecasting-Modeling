#!/usr/bin/env python3
"""Compute post-hoc permutation importance (RF) and TreeSHAP (XGBoost).

Built-in tree importances (Gini for RF, gain for XGBoost) are biased — they
under-credit correlated features and over-credit high-cardinality ones. This
script computes the methodologically clean alternatives on the **out-of-sample
test slice** for each (fold, lead, tree-model):

  - Random Forest    →  permutation importance (model-agnostic, OOS-evaluated)
  - XGBoost          →  TreeSHAP mean(|SHAP|) (theoretical Shapley credit)

Operates in two modes per (fold, lead, model):

  1. FAST PATH — if a pickled model exists at the expected `results/models/`
     path (set `save_models: true` in the experiment YAML for future runs),
     load it directly.

  2. SLOW PATH — otherwise, refit the model on the contiguous refit slice
     using the best HPs recorded in `fold_runs.csv`. ~30s per model on a
     laptop, so a full 5×3×2 sweep takes ~15-30 minutes.

Writes per-feature rows back to `feature_status.csv` with
`kind ∈ {'permutation', 'shap_mean_abs'}` — joining the existing
`kind ∈ {'coef','gini','gain'}` rows. Idempotent (drops any prior post-hoc
rows before appending).

Usage:
    python scripts/05_compute_posthoc_importance.py \\
        --exp-config configs/experiments/exp_default.yaml \\
        [--prefix ""] \\
        [--models rf,xgboost] \\
        [--n-repeats 10] \\
        [--shap-samples 5000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from droughtmodel.models._tabular import _stack_xy
from droughtmodel.models.registry import REGISTRY
from droughtmodel.pipeline import ExperimentRunner
from droughtmodel.selection import (
    permutation_importance_scores,
    tree_shap_importance,
)
from droughtmodel.utils import RESULTS_DIR

POSTHOC_KINDS = {"permutation", "shap_mean_abs"}


def _instantiate_with_best_hps(
    model_name: str,
    best_params: dict,
    best_iteration,
    runner: ExperimentRunner,
):
    """Build a fresh model with the best HPs from `fold_runs.csv`."""
    cls = REGISTRY[model_name]
    defaults = (runner._load_model_defaults(model_name).get("params") or {}).copy()
    # `best_params` overrides defaults; drop the overridden keys to avoid passing
    # the same kwarg twice.
    for k in best_params:
        defaults.pop(k, None)
    # XGBoost: if the search captured a `best_iteration`, lock it as the refit
    # n_estimators and disable further early stopping (matches the original
    # `refit_with_best_iteration=True` logic in `tune_and_refit`).
    if model_name == "xgboost" and pd.notna(best_iteration):
        defaults["n_estimators"] = int(best_iteration) + 1
        defaults["early_stopping_rounds"] = None
    return cls(**{**defaults, **best_params})


def _extract_xy(prep, model) -> tuple[np.ndarray, np.ndarray, list]:
    """Build the OOS test (X, y, feature_names) the model was scored on."""
    feature_names = list(model.feature_names_)
    X, y, _ = _stack_xy(prep.test, feature_names)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], feature_names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-config", default="configs/experiments/exp_default.yaml",
                    help="Path to the experiment YAML used by the original run.")
    ap.add_argument("--prefix", default="",
                    help="File-name prefix used by the original run. Default '' = production.")
    ap.add_argument("--models", default="rf,xgboost",
                    help="Comma-separated model names to process.")
    ap.add_argument("--n-repeats", type=int, default=10,
                    help="Permutation importance n_repeats (RF only).")
    ap.add_argument("--shap-samples", type=int, default=5000,
                    help="Max samples to use for TreeSHAP (XGBoost only).")
    ap.add_argument("--lead", type=int, default=None,
                    help="Only process this single lead (e.g. --lead 3). Default: all leads.")
    ap.add_argument("--fold", type=int, default=None,
                    help="Only process this single fold index (e.g. --fold 1). Default: all folds.")
    args = ap.parse_args()

    prefix = args.prefix
    target_models = [m.strip() for m in args.models.split(",") if m.strip()]

    runner = ExperimentRunner(args.exp_config, verbose=False)
    runner.setup_data()

    # Resolve log paths from the runner (so it follows the experiment YAML's
    # `output.logs_dir` — needed e.g. for winter-training → results/winter-training/logs/).
    fold_runs_path = runner.logs_dir / f"{prefix}fold_runs.csv"
    feat_status_path = runner.logs_dir / f"{prefix}feature_status.csv"
    for p in (fold_runs_path, feat_status_path):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    fold_runs = pd.read_csv(fold_runs_path)
    feat_status = pd.read_csv(feat_status_path)

    leads = sorted(fold_runs["lead"].unique())
    folds = [s.index for s in runner.cv.fold_specs]
    # Apply optional filters for partial / parallel runs
    if args.lead is not None:
        leads = [L for L in leads if L == args.lead]
    if args.fold is not None:
        folds = [f for f in folds if f == args.fold]

    print(f"Post-hoc importance — prefix={prefix!r}", flush=True)
    print(f"  target models : {target_models}", flush=True)
    print(f"  leads × folds : {leads} × {folds}", flush=True)
    print(f"  permutation n_repeats : {args.n_repeats}", flush=True)
    print(f"  shap max_samples       : {args.shap_samples}", flush=True)

    t0 = time.time()
    new_rows: list[dict] = []

    for lead in leads:
        # The per-fold prep is shared by all models within this (lead, fold).
        prep_cache = {}
        for fold_spec in [s for s in runner.cv.fold_specs if s.index in folds]:
            for name in target_models:
                runs_row = fold_runs.query(
                    "lead == @lead and fold == @fold_spec.index and model == @name"
                )
                if runs_row.empty:
                    print(f"  skip L={lead} fold{fold_spec.index} {name}: no fold_runs row")
                    continue
                runs_row = runs_row.iloc[0]

                # Lazily build the per-fold prep — only once per (lead, fold).
                if fold_spec.index not in prep_cache:
                    prep_cache[fold_spec.index] = runner._prepare_fold(lead, fold_spec)
                prep = prep_cache[fold_spec.index]

                # Try to load a saved model; otherwise refit.
                model_path = (
                    runner.models_dir
                    / f"{prefix}{name}_lead{lead}_fold{fold_spec.index}.joblib"
                )
                t_step = time.time()
                if model_path.exists():
                    import joblib
                    model = joblib.load(model_path)
                    src = "loaded"
                else:
                    best_params = json.loads(runs_row["best_params"])
                    best_iter = runs_row.get("best_iteration", float("nan"))
                    model = _instantiate_with_best_hps(name, best_params, best_iter, runner)
                    model.fit(prep.refit)
                    src = "refit"

                X_test, y_test, feature_names = _extract_xy(prep, model)
                if X_test.shape[0] == 0:
                    print(f"  skip L={lead} fold{fold_spec.index} {name}: no finite test samples")
                    continue

                if name == "rf":
                    df = permutation_importance_scores(
                        model.estimator.predict, X_test, y_test, feature_names,
                        n_repeats=args.n_repeats, scoring="neg_mse", random_state=42,
                    )
                    df = df.rename(columns={"importance_mean": "importance"})
                    kind = "permutation"
                elif name == "xgboost":
                    df, _shap = tree_shap_importance(
                        model.estimator, X_test, feature_names,
                        max_samples=args.shap_samples, random_state=42,
                    )
                    df = df.rename(columns={"mean_abs_shap": "importance"})
                    kind = "shap_mean_abs"
                else:
                    print(f"  skip L={lead} fold{fold_spec.index} {name}: no post-hoc method defined")
                    continue

                for _, r in df.iterrows():
                    new_rows.append({
                        "fold": fold_spec.index,
                        "lead": lead,
                        "model": name,
                        "feature": r["feature"],
                        "importance": float(r["importance"]),
                        "retained": True,
                        "kind": kind,
                    })

                print(f"  L={lead} fold{fold_spec.index} {name:<8} ({src})  "
                      f"+{len(df):>3d} rows  ({time.time() - t_step:5.1f}s)", flush=True)

    if not new_rows:
        print("\nWARN: produced 0 post-hoc rows.", file=sys.stderr)
        return 1

    new_df = pd.DataFrame(new_rows)

    # Selective idempotent merge: only drop existing post-hoc rows that match
    # one of the (model, lead, fold, kind) tuples we're about to replace, so
    # incremental --lead / --fold runs don't wipe other slices.
    replacement_keys = set(zip(
        new_df["model"], new_df["lead"], new_df["fold"], new_df["kind"],
    ))
    is_replacement = feat_status.apply(
        lambda r: (r["model"], r["lead"], r["fold"], r["kind"]) in replacement_keys,
        axis=1,
    )
    existing = feat_status[~is_replacement].copy()
    n_dropped = int(is_replacement.sum())
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(feat_status_path, index=False)

    print(f"\nwrote {feat_status_path.relative_to(ROOT)}  "
          f"({len(existing)} kept + {len(new_df)} new = {len(combined)} rows;  "
          f"replaced {n_dropped} matching prior rows)", flush=True)
    print(f"total elapsed: {(time.time() - t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())