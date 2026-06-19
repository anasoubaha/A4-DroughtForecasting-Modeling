"""Pipeline orchestrator — Phase 10 (v3 §6.1 per-fold pipeline).

End-to-end runner: iterates over (lead × fold × model), producing:

  - per-fold predictions stitched into pooled out-of-sample arrays
  - per-(model, lead) NetCDF prediction files
  - per-(model, lead) metrics CSV with block-bootstrap 95% CIs
  - per-(fold, lead, model) fold-runs log
  - per-(fold, lead, model, feature) feature-status log

The runner consumes an experiment YAML (see ``configs/experiments/*.yaml``)
plus the standing data / features / cv / metrics configs. Every per-fold step
follows the §6.1 sequence:

    1. Provisional split
    2. Per-fold PACF + CCF lag selection
    3. K_eff = compute_quarantine_max_lag(selected_lags)  →  gap = L + K_eff + 2
    4. Final indices (train_idx, val_idx, test_idx)
    5. Feature-dataset build
    6. Fit standardizer on train_idx; apply to train/val/test + the contiguous
       refit slice [train_start, test_start − gap − 1]
    7. For each model:
         - tune_and_refit(grid, train, val, refit_dataset=refit_slice)  [if HP grid]
         - or simple .fit(refit_slice) + .predict(test)                  [baselines]
    8. Stitch test predictions into the pooled OOS array
    9. Evaluate metrics (winter pool + all-months) with block-bootstrap CIs

Climatology + Persistence are always run regardless of the experiment's
``models`` list because they're required as references for MSSS metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from droughtmodel import cv as dcv
from droughtmodel import data as ddata
from droughtmodel import evaluation as deval
from droughtmodel import features as dfeat
from droughtmodel.cv import compute_quarantine_max_lag
from droughtmodel.models.registry import get_model
from droughtmodel.tuning import grid_search, optuna_search, tune_and_refit
from droughtmodel.utils import PROJECT_ROOT


# Models that need val passed to fit() (XGBoost for early stopping) and that
# should refit with best_iteration locked from the search.
_XGB_LIKE = {"xgboost"}

# Tolerance below which a Lasso/ElasticNet coefficient is treated as zero
# (feature dropped). For standardized inputs, true L1 zeros are exact.
_ZERO_TOL = 1e-10

# Default Optuna n_trials when search_backends[name] == "optuna" but no
# per-model override is given in exp_config["optuna_n_trials"].
_DEFAULT_OPTUNA_N_TRIALS = 40


def _make_optuna_categorical_space(grid: dict[str, Any]):
    """Convert a `hp_grids[model]` dict of lists into an Optuna search-space callable.

    Each parameter becomes a ``trial.suggest_categorical(name, values)`` so the
    same YAML definition drives both backends — the SEARCH SPACE is identical;
    only the search strategy (exhaustive grid vs TPE sampling) differs.
    """
    items = list(grid.items())

    def space(trial):
        return {k: trial.suggest_categorical(k, list(v)) for k, v in items}

    return space


# ---------------------------------------------------------------------------
# Log row schemas
# ---------------------------------------------------------------------------

@dataclass
class FoldRunLog:
    fold: int
    lead: int
    model: str
    best_params: str               # JSON-encoded dict
    best_val_score: float | None
    best_iteration: int | None
    n_features_total: int
    n_features_retained: int
    K_eff: int
    boundary_gap: int
    search_duration_s: float
    fit_duration_s: float
    n_trials: int


@dataclass
class FeatureStatusLog:
    fold: int
    lead: int
    model: str
    feature: str
    importance: float
    retained: bool
    kind: str                      # "coef" | "gini" | "gain" | "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    with open(p) as f:
        return yaml.safe_load(f)


def _importance_kind(model) -> str:
    est = getattr(model, "estimator", None)
    if est is None:
        return "none"
    if hasattr(est, "coef_"):
        return "coef"
    # Tree importance: RF uses Gini, XGBoost gain (default in modern xgboost).
    if hasattr(est, "feature_importances_"):
        from sklearn.ensemble import RandomForestRegressor
        return "gini" if isinstance(est, RandomForestRegressor) else "gain"
    return "none"


def _feature_status_rows(model, fold: int, lead: int, name: str) -> list[FeatureStatusLog]:
    fi = model.feature_importance() if hasattr(model, "feature_importance") else None
    if not fi:
        return []
    kind = _importance_kind(model)
    rows: list[FeatureStatusLog] = []
    for feat, val in fi.items():
        # Linear with L1 → "retained" means non-zero coefficient.
        # Other models → all features are retained (no embedded selection).
        if kind == "coef":
            retained = abs(val) > _ZERO_TOL
        else:
            retained = True
        rows.append(FeatureStatusLog(
            fold=fold, lead=lead, model=name, feature=feat,
            importance=float(val), retained=retained, kind=kind,
        ))
    return rows


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

@dataclass
class _PreparedFold:
    fold_index: int
    lead: int
    selected_lags: dict[str, list[int]]
    K_eff: int
    boundary_gap: int
    train: xr.Dataset
    val: xr.Dataset
    test: xr.Dataset
    refit: xr.Dataset
    time_test: np.ndarray
    target_test: np.ndarray


class ExperimentRunner:
    """Runs one experiment defined by a YAML config."""

    def __init__(
        self,
        exp_config: dict[str, Any] | str | Path,
        *,
        data_cfg: dict | None = None,
        feat_cfg: dict | None = None,
        cv_cfg: dict | None = None,
        metrics_cfg: dict | None = None,
        verbose: bool = True,
    ):
        if isinstance(exp_config, (str, Path)):
            exp_config = _load_yaml(exp_config)
        self.exp = exp_config
        self.verbose = verbose

        self.data_cfg = data_cfg or ddata.load_config()
        self.feat_cfg = feat_cfg or dfeat.load_features_config()
        self.cv_cfg = cv_cfg or dcv.load_cv_config()
        self.metrics_cfg = metrics_cfg or deval.load_metrics_config()

        out = self.exp.get("output", {})
        self.preds_dir = self._resolve(out.get("predictions_dir", "results/predictions"))
        self.metrics_dir = self._resolve(out.get("metrics_dir", "results/metrics"))
        self.logs_dir = self._resolve(out.get("logs_dir", "results/logs"))
        self.models_dir = self._resolve(out.get("models_dir", "results/models"))
        # Optional prefix prepended to every output filename — lets smoke and
        # full-sweep runs coexist in the same directories without collision.
        self.file_prefix: str = str(self.exp.get("file_prefix", "") or "")
        # Opt-in: pickle each fitted model so post-hoc importance (permutation,
        # SHAP) can run later without refitting. Adds ~1-3 GB to a full sweep.
        self.save_models: bool = bool(self.exp.get("save_models", False))
        for d in (self.preds_dir, self.metrics_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.save_models:
            self.models_dir.mkdir(parents=True, exist_ok=True)

        # State filled in by .run()
        self._datasets: dict[str, xr.Dataset] | None = None
        self._template: xr.DataArray | None = None
        self._morocco_mask: xr.DataArray | None = None
        self._cv: dcv.RollingOriginCV | None = None
        self._fold_runs: list[FoldRunLog] = []
        self._feature_status: list[FeatureStatusLog] = []
        # Progress / ETA tracking
        self._fits_done: int = 0
        self._total_fits: int = 0
        self._run_t0: float = 0.0

    @staticmethod
    def _resolve(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -----------------------------------------------------------------------
    # Per-fold preparation
    # -----------------------------------------------------------------------
    def setup_data(self) -> None:
        """Eagerly load datasets + Morocco mask + CV object.

        Normally called from inside `run()`. External scripts (e.g.
        `scripts/05_compute_posthoc_importance.py`) call it directly to use
        `_prepare_fold` without running the full pipeline.
        """
        if self._datasets is not None:
            return
        self._datasets = ddata.load_all(self.data_cfg)
        self._template = dfeat.gather_predictor("spei3", self._datasets)
        self._morocco_mask = dfeat.load_region_mask(
            self.feat_cfg["region_mask"]["path"], self._template,
            name=self.feat_cfg["region_mask"]["name"],
        )
        self._cv = dcv.RollingOriginCV.from_config(self.cv_cfg)

    @property
    def cv(self) -> dcv.RollingOriginCV:
        """The configured RollingOriginCV (after ``setup_data()``)."""
        if self._cv is None:
            raise RuntimeError("ExperimentRunner.setup_data() must be called first.")
        return self._cv

    def prepare_fold(self, lead: int, fold_spec: dcv.FoldSpec) -> _PreparedFold:
        """Public wrapper around `_prepare_fold` that ensures data is loaded."""
        self.setup_data()
        return self._prepare_fold(lead, fold_spec)

    def _prepare_fold(self, lead: int, fold_spec: dcv.FoldSpec) -> _PreparedFold:
        """All per-fold prep through §6.1 step 6 (standardized slices ready to fit)."""
        # 2. Lag selection on planned train (pre-quarantine).
        train_end = fold_spec.val_start - pd.Timedelta(days=1)
        train_subset = {
            k: ds.sel(time=slice(fold_spec.train_start, train_end))
            for k, ds in self._datasets.items()
        }
        selected, _ = dfeat.select_lags_from_training(
            train_subset,
            long_memory_vars=self.feat_cfg["long_memory_vars"],
            fast_response_vars=self.feat_cfg["fast_response_vars"],
            pacf_threshold=self.feat_cfg["pacf"]["threshold"],
            pacf_n_lags=self.feat_cfg["pacf"]["n_lags"],
            ccf_target=self.feat_cfg["ccf"]["target"],
            ccf_threshold=self.feat_cfg["ccf"]["threshold"],
            ccf_max_lag=self.feat_cfg["ccf"]["max_lag"],
            region_mask=self._morocco_mask,
            winter_only_ccf=True,
            aggregation_mode=self.feat_cfg["aggregation_mode"],
        )

        # 3. Strict quarantine: K_eff over precip-touching variables only.
        K_eff = compute_quarantine_max_lag(selected)

        # 4. Final fold indices.
        fi = self._cv.get_fold_indices(self._template["time"], fold_spec,
                                       max_lag=K_eff, lead=lead)

        # 5. Build the lead-shifted feature dataset for this fold.
        ds_feat = dfeat.build_dataset(
            self._datasets, lead=lead,
            contemporary=self.feat_cfg["contemporary_predictors"],
            lags=selected,
            include_seasonal=self.feat_cfg["include_seasonal_encoding"],
            include_spatial=self.feat_cfg["include_spatial_encoding"],
        )

        train = ds_feat.isel(time=fi.train_idx)
        val = ds_feat.isel(time=fi.val_idx)
        test = ds_feat.isel(time=fi.test_idx)

        # Contiguous refit slice [train_start, test_start − gap − 1]
        # train_idx[0] = train_start_index; test_idx[0] = test_start_index;
        # last refit index = test_start_index − gap − 1 (inclusive).
        refit_start = int(fi.train_idx[0])
        refit_stop = int(fi.test_idx[0]) - fi.boundary_gap   # exclusive (Python slice)
        refit = ds_feat.isel(time=slice(refit_start, refit_stop))

        # 6. Standardize: fit on train_idx, apply to all four slices.
        std = dcv.FoldStandardizer.from_config(self.cv_cfg, region_mask=self._morocco_mask).fit(train)
        train_n = std.transform(train).where(self._morocco_mask)
        val_n = std.transform(val).where(self._morocco_mask)
        test_n = std.transform(test).where(self._morocco_mask)
        refit_n = std.transform(refit).where(self._morocco_mask)

        return _PreparedFold(
            fold_index=fi.index,
            lead=lead,
            selected_lags=selected,
            K_eff=K_eff,
            boundary_gap=fi.boundary_gap,
            train=train_n,
            val=val_n,
            test=test_n,
            refit=refit_n,
            time_test=test_n["time"].values,
            target_test=test_n["target"].values,
        )

    # -----------------------------------------------------------------------
    # Per-model fit/predict
    # -----------------------------------------------------------------------
    def _fit_predict(self, name: str, prep: _PreparedFold) -> np.ndarray:
        """Fit a single model on the prepared fold and return test predictions."""
        hp_grid = self.exp.get("hp_grids", {}).get(name)
        model_defaults = self._load_model_defaults(name)
        fixed = {**(model_defaults.get("params") or {})}

        t_fit_start = time.time()

        if hp_grid:
            # Drop any default params that the grid will set, to avoid duplicates.
            grid_keys = set(hp_grid.keys()) if isinstance(hp_grid, dict) else set()
            for k in grid_keys:
                fixed.pop(k, None)

            is_xgb = name in _XGB_LIKE
            search_backend = self.exp.get("search_backends", {}).get(name, "grid")

            if search_backend == "optuna":
                n_trials = (self.exp.get("optuna_n_trials") or {}).get(
                    name, _DEFAULT_OPTUNA_N_TRIALS
                )
                space = _make_optuna_categorical_space(hp_grid)
                final_model, search_result = tune_and_refit(
                    model_class=get_model_class(name),
                    grid={},                       # ignored by optuna backend
                    train=prep.train,
                    val=prep.val,
                    refit_dataset=prep.refit,
                    fixed_params=fixed,
                    pass_val_to_fit=is_xgb,
                    refit_with_best_iteration=is_xgb,
                    search_fn=optuna_search,
                    search_kwargs={
                        "search_space": space,
                        "n_trials": int(n_trials),
                        "sampler_seed": 42,
                    },
                )
            else:
                final_model, search_result = tune_and_refit(
                    model_class=get_model_class(name),
                    grid=hp_grid,
                    train=prep.train,
                    val=prep.val,
                    refit_dataset=prep.refit,
                    fixed_params=fixed,
                    pass_val_to_fit=is_xgb,
                    refit_with_best_iteration=is_xgb,
                    search_fn=grid_search,
                )

            best_params = dict(search_result.best_params)
            best_score = float(search_result.best_score)
            n_trials = int(search_result.n_trials)
            search_duration = float(search_result.duration_s)
            best_iter = getattr(search_result.best_model, "best_iteration", None)
        else:
            # No HP tuning — direct fit on the contiguous refit slice.
            final_model = get_model(name, **fixed).fit(prep.refit)
            best_params = dict(fixed)
            best_score = None
            n_trials = 0
            search_duration = 0.0
            best_iter = getattr(final_model, "best_iteration", None)

        fit_duration = time.time() - t_fit_start

        # Optional model serialization for downstream post-hoc importance
        # (permutation, SHAP). Only trees actually benefit — but we save all
        # models for consistency if opted in. Joblib + compress=3 trims size.
        if self.save_models:
            try:
                import joblib
                model_path = self.models_dir / (
                    f"{self.file_prefix}{name}_lead{prep.lead}_fold{prep.fold_index}.joblib"
                )
                joblib.dump(final_model, model_path, compress=3)
            except Exception as e:
                self._log(f"    [warn] could not save {name} model: {e}")

        # Predict on test (NaN-safe via TabularBaseModel.predict).
        pred = final_model.predict(prep.test).values

        # Log row
        fi_dict = final_model.feature_importance() if hasattr(final_model, "feature_importance") else None
        n_total = len(fi_dict) if fi_dict else 0
        if fi_dict and _importance_kind(final_model) == "coef":
            n_retained = sum(1 for v in fi_dict.values() if abs(v) > _ZERO_TOL)
        else:
            n_retained = n_total

        self._fold_runs.append(FoldRunLog(
            fold=prep.fold_index, lead=prep.lead, model=name,
            best_params=json.dumps(best_params, default=float),
            best_val_score=best_score,
            best_iteration=int(best_iter) if best_iter is not None else None,
            n_features_total=n_total,
            n_features_retained=n_retained,
            K_eff=prep.K_eff, boundary_gap=prep.boundary_gap,
            search_duration_s=search_duration,
            fit_duration_s=fit_duration,
            n_trials=n_trials,
        ))
        self._feature_status.extend(_feature_status_rows(final_model, prep.fold_index, prep.lead, name))

        # Real-time progress line
        self._fits_done += 1
        pct = 100.0 * self._fits_done / max(self._total_fits, 1)
        elapsed = time.time() - self._run_t0
        avg = elapsed / self._fits_done
        remaining = avg * (self._total_fits - self._fits_done)
        extras = f", trials={n_trials}" if n_trials else ""
        extras += f", retained={n_retained}/{n_total}" if n_total else ""
        if best_iter is not None:
            extras += f", best_iter={best_iter}"
        self._log(
            f"    [{self._fits_done:>3}/{self._total_fits}  {pct:5.1f}%] "
            f"L={prep.lead} fold{prep.fold_index} {name:<12} "
            f"fit={fit_duration:6.2f}s{extras}  | elapsed={elapsed:6.1f}s  ETA={remaining:6.1f}s"
        )
        return pred

    def _load_model_defaults(self, name: str) -> dict[str, Any]:
        path = PROJECT_ROOT / "configs" / "models" / f"{name}.yaml"
        if not path.exists():
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    # -----------------------------------------------------------------------
    # Save helpers — single combined-across-leads files
    # -----------------------------------------------------------------------
    def _save_predictions_all_leads(
        self,
        preds_per_lead: dict[int, dict[str, np.ndarray]],
        truth_per_lead: dict[int, np.ndarray],
        time_per_lead: dict[int, np.ndarray],
    ) -> None:
        """Write ONE NetCDF combining all leads as a `lead` dimension.

        Each pred / truth array becomes ``(lead, time, lat, lon)``. The time
        coord is identical across leads (the stitched 2000-01 → 2024-12 OOS
        window depends only on the fold layout, not on the lead).
        """
        leads = sorted(preds_per_lead.keys())
        lat = self._template["lat"].values
        lon = self._template["lon"].values

        ref_time = time_per_lead[leads[0]]
        for L in leads[1:]:
            if not np.array_equal(time_per_lead[L], ref_time):
                raise ValueError(
                    f"Time coord mismatch between leads {leads[0]} and {L}; cannot stack."
                )
        time_coord = pd.DatetimeIndex(ref_time)

        model_names = list(preds_per_lead[leads[0]].keys())
        data_vars: dict[str, tuple] = {
            f"pred_{name}": (
                ("lead", "time", "lat", "lon"),
                np.stack([preds_per_lead[L][name] for L in leads], axis=0),
            )
            for name in model_names
        }
        data_vars["truth"] = (
            ("lead", "time", "lat", "lon"),
            np.stack([truth_per_lead[L] for L in leads], axis=0),
        )

        ds = xr.Dataset(
            data_vars,
            coords={"lead": leads, "time": time_coord, "lat": lat, "lon": lon},
            attrs={"experiment": self.exp.get("name", "default")},
        )
        out = self.preds_dir / f"{self.file_prefix}pooled_allLeads.nc"
        ds.to_netcdf(out)
        self._log(f"wrote predictions → {out.relative_to(PROJECT_ROOT)}  "
                  f"({len(leads)} leads × {len(time_coord)} months)")

    def _evaluate_slice(
        self,
        preds: dict[str, np.ndarray],
        truth: np.ndarray,
        time_arr: np.ndarray,
        lead: int,
        fold_label,
    ) -> pd.DataFrame:
        """Evaluate every model on ONE (preds, truth, time) slice and return tidy rows.

        Computes winter-only + all-months with block-bootstrap CIs. The ``fold_label``
        is written into the ``fold`` column verbatim — pass ``'pooled'`` for the
        across-folds slice or the fold index (int) for a single-fold slice.
        """
        clim = preds.get("climatology")
        pers = preds.get("persistence")
        if clim is None or pers is None:
            raise RuntimeError("climatology and persistence preds required for skill scores.")

        months = pd.DatetimeIndex(time_arr).month
        winter_mask = np.isin(months, [11, 12, 1, 2])

        rows: list[pd.DataFrame] = []
        for window_name, mask, block in [
            ("winter_only", winter_mask, self.metrics_cfg["bootstrap"]["mean_block_length_winter"]),
            ("all_months", np.ones_like(months, dtype=bool),
             self.metrics_cfg["bootstrap"]["mean_block_length_all"]),
        ]:
            # If a slice has no rows for this window (e.g. a single fold's winter
            # subset might be empty under unusual configs), skip gracefully.
            if not mask.any():
                continue
            reporter = deval.MetricsReporter.from_config(self.metrics_cfg, evaluation_window=window_name)
            reporter.mean_block_length = block

            y_true = truth[mask]
            y_clim = clim[mask]
            y_pers = pers[mask]

            for name, arr in preds.items():
                y_pred = arr[mask]
                res = reporter.evaluate(y_pred, y_true, climatology=y_clim, persistence=y_pers)
                df = deval.MetricsReporter.to_dataframe(
                    res, model=name, lead=lead, fold=fold_label, evaluation_window=window_name,
                )
                rows.append(df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _compute_metrics_for_lead(
        self,
        preds_per_fold: dict[str, list[np.ndarray]],
        truth_per_fold: list[np.ndarray],
        time_per_fold: list[np.ndarray],
        lead: int,
    ) -> pd.DataFrame:
        """Compute headline metrics on the pooled slice AND each individual fold.

        The pooled rows (``fold='pooled'``) are the paper headline. The per-fold
        rows (``fold=1, 2, …``) enable stability diagnostics — see notebook 06
        and the related stability bar chart.

        Returns the concatenated tidy DataFrame; caller is responsible for the
        across-leads concatenation and final CSV write.
        """
        n_folds = len(truth_per_fold)
        all_rows: list[pd.DataFrame] = []

        # Pooled slice — concatenate the per-fold lists along the time axis.
        pooled_preds = {n: np.concatenate(arrs, axis=0) for n, arrs in preds_per_fold.items()}
        pooled_truth = np.concatenate(truth_per_fold, axis=0)
        pooled_time = np.concatenate(time_per_fold, axis=0)
        all_rows.append(self._evaluate_slice(pooled_preds, pooled_truth, pooled_time, lead, "pooled"))

        # Per-fold slices.
        for i in range(n_folds):
            fold_preds = {n: arrs[i] for n, arrs in preds_per_fold.items()}
            all_rows.append(self._evaluate_slice(
                fold_preds, truth_per_fold[i], time_per_fold[i], lead, i + 1,
            ))

        return pd.concat([df for df in all_rows if not df.empty], ignore_index=True)

    def _save_metrics_all_leads(self, dfs_per_lead: list[pd.DataFrame]) -> pd.DataFrame:
        """Concat per-lead metrics DataFrames and write one combined CSV."""
        combined = pd.concat(dfs_per_lead, ignore_index=True)
        out = self.metrics_dir / f"{self.file_prefix}metrics_allLeads.csv"
        combined.to_csv(out, index=False)
        self._log(f"wrote metrics → {out.relative_to(PROJECT_ROOT)}  ({len(combined)} rows)")
        return combined

    def _save_logs(self) -> None:
        if self._fold_runs:
            df = pd.DataFrame([asdict(r) for r in self._fold_runs])
            out = self.logs_dir / f"{self.file_prefix}fold_runs.csv"
            df.to_csv(out, index=False)
            self._log(f"wrote fold-runs log → {out.relative_to(PROJECT_ROOT)}  ({len(df)} rows)")
        if self._feature_status:
            df = pd.DataFrame([asdict(r) for r in self._feature_status])
            out = self.logs_dir / f"{self.file_prefix}feature_status.csv"
            df.to_csv(out, index=False)
            self._log(f"wrote feature-status log → {out.relative_to(PROJECT_ROOT)}  ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # run()
    # -----------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Execute the full experiment. Returns the combined metrics DataFrame
        (pooled + per-fold rows across all leads)."""
        t0 = time.time()
        self._log(f"=== ExperimentRunner: '{self.exp.get('name', 'default')}' ===")

        # Load data once
        self.setup_data()
        self._log(f"loaded data ({self._template.sizes}) + Morocco mask "
                  f"({int(self._morocco_mask.sum())} cells)")

        # Models to run for THIS experiment (climatology + persistence always added for refs)
        configured = list(self.exp["models"])
        models_to_run = list(configured)
        for required in ("climatology", "persistence"):
            if required not in models_to_run:
                models_to_run.insert(0, required)

        # Accumulators across all leads — written once at the end as combined files.
        preds_per_lead: dict[int, dict[str, np.ndarray]] = {}
        truth_per_lead: dict[int, np.ndarray] = {}
        time_per_lead: dict[int, np.ndarray] = {}
        metrics_dfs: list[pd.DataFrame] = []

        # Progress counters — used by _fit_predict for ETA computation.
        self._total_fits = len(self.exp["leads"]) * len(self._cv.fold_specs) * len(models_to_run)
        self._fits_done = 0
        self._run_t0 = time.time()
        self._log(f"will perform {self._total_fits} fits "
                  f"({len(self.exp['leads'])} leads × {len(self._cv.fold_specs)} folds × "
                  f"{len(models_to_run)} models)")

        for lead in self.exp["leads"]:
            lead_t0 = time.time()
            self._log(f"\n--- lead L = {lead} ---")
            pooled_preds: dict[str, list[np.ndarray]] = {n: [] for n in models_to_run}
            truth_pool: list[np.ndarray] = []
            time_pool: list[np.ndarray] = []

            for fold_spec in self._cv.fold_specs:
                fold_t0 = time.time()
                self._log(f"  fold {fold_spec.index} …")
                prep = self._prepare_fold(lead, fold_spec)
                self._log(f"    K_eff={prep.K_eff}, gap={prep.boundary_gap}, "
                          f"n_train={prep.train.sizes['time']}, n_val={prep.val.sizes['time']}, "
                          f"n_test={prep.test.sizes['time']}, n_refit={prep.refit.sizes['time']}")
                for name in models_to_run:
                    pred = self._fit_predict(name, prep)
                    pooled_preds[name].append(pred)
                truth_pool.append(prep.target_test)
                time_pool.append(prep.time_test)
                self._log(f"  fold {fold_spec.index} done in {time.time() - fold_t0:.1f}s")

            # Compute pooled + per-fold metrics from the per-fold lists.
            metrics_dfs.append(self._compute_metrics_for_lead(
                pooled_preds, truth_pool, time_pool, lead,
            ))
            # Stitch the per-lead pools (used by _save_predictions_all_leads below).
            preds_per_lead[lead] = {n: np.concatenate(p, axis=0) for n, p in pooled_preds.items()}
            truth_per_lead[lead] = np.concatenate(truth_pool, axis=0)
            time_per_lead[lead] = np.concatenate(time_pool, axis=0)
            self._log(f"--- lead L={lead} done in {time.time() - lead_t0:.1f}s ---")

        # Combined writes (one predictions NetCDF + one metrics CSV across all leads).
        self._save_predictions_all_leads(preds_per_lead, truth_per_lead, time_per_lead)
        combined_metrics = self._save_metrics_all_leads(metrics_dfs)
        self._save_logs()
        self._log(f"\n=== done in {time.time() - t0:.1f}s "
                  f"({self._fits_done}/{self._total_fits} fits completed) ===")
        return combined_metrics


# ---------------------------------------------------------------------------
# Small lookup that avoids importing model classes upfront
# ---------------------------------------------------------------------------

def get_model_class(name: str):
    """Look up the model CLASS (not an instance) from the registry."""
    from droughtmodel.models.registry import REGISTRY
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
