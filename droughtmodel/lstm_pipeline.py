"""Phase 12 — LSTM-specific experiment runner.

Sits beside `droughtmodel.pipeline.ExperimentRunner` and reuses the same
data-loading, FoldStandardizer, CV-fold logic, metrics, and prediction-stitching
machinery — but diverges in three places:

  1. **No PACF/CCF lag selection.** The feature dataset is built with
     ``lags={}``; the LSTM ingests the raw history of contemporary predictors
     (+ seasonal & spatial encodings) and learns the lag structure end-to-end.

  2. **Updated boundary gap** ``gap = lead + T + 1`` (vs the tabular
     ``L + K_eff + 2``), implemented by passing ``max_lag = T - 1`` to
     ``RollingOriginCV.get_fold_indices`` so the existing per-fold quarantine
     plumbing is reused exactly.

  3. **Manual 8-combo grid search** run on Fold 1 only. The best combo is
     locked and deployed verbatim to Folds 2-5 (matching the v12 spec's
     "Representative Tuning" directive).

Climatology + Persistence are run alongside the LSTM as MSSS references —
they use the same v3-equivalent unfiltered baseline slices as the tabular
pipeline so the skill scores are directly comparable across experiments.
"""

from __future__ import annotations

import itertools
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
from droughtmodel.models.lstm import LSTMModel
from droughtmodel.models.registry import get_model
from droughtmodel.pipeline import (
    FoldRunLog,
    _BASELINE_MODELS,
    _filter_to_winter_targets,
    _winter_target_mask,
)
from droughtmodel.sequence import (
    SequenceMeta,
    build_sequences,
    lstm_boundary_gap,
    predict_to_grid,
)
from droughtmodel.utils import PROJECT_ROOT


__all__ = ["LSTMExperimentRunner"]


# ---------------------------------------------------------------------------
# Per-fold preparation result
# ---------------------------------------------------------------------------

@dataclass
class _PreparedLSTMFold:
    fold_index: int
    lead: int
    sequence_length: int
    boundary_gap: int
    # 2-D xarray slices — these have been standardized via FoldStandardizer.
    # ML-LSTM slices may be winter-filtered (winter_only_training=True);
    # *_full slices are always unfiltered for baseline use.
    train: xr.Dataset
    val: xr.Dataset
    test: xr.Dataset
    refit: xr.Dataset
    train_full: xr.Dataset
    val_full: xr.Dataset
    refit_full: xr.Dataset
    # UNSTANDARDIZED train slice — kept so the full-timeline standardizer can
    # refit on raw inputs. Re-fitting on the already-standardized `train` slice
    # was a no-op transform (mu≈0, sigma≈1) that left full_ds in raw units,
    # which is what caused the LSTM divergence on 2026-06-30 — see incident
    # notes in droughtmodel/models/lstm.py.
    train_unstd: xr.Dataset
    time_test: np.ndarray
    target_test: np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    with open(p) as f:
        return yaml.safe_load(f)


def _make_grid_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the {param: [values...]} grid, preserving key order."""
    keys = list(grid.keys())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def _train_window_end_indices(
    ds: xr.Dataset,
    slice_start_t: pd.Timestamp,
    slice_end_t: pd.Timestamp,
    sequence_length: int,
    *,
    winter_filter: bool,
) -> np.ndarray:
    """Feature-time indices into `ds.time` that mark the END of each sample's window.

    Returned indices satisfy:
      - The full T-window `[t - T + 1, t]` is within the dataset bounds.
      - The feature time `ds.time[t]` falls in the inclusive `[slice_start_t, slice_end_t]` window.
      - (If `winter_filter`) the lead-shifted target month is in {Nov, Dec, Jan, Feb}.

    Lets the LSTM look BACKWARD across slice boundaries — which is legitimate
    because the boundary gap quarantines target leakage, not raw input
    visibility. (The contaminating side is the FUTURE target at train_end + L,
    not the past inputs at slice_start - T + 1.)
    """
    time_arr = pd.DatetimeIndex(ds["time"].values)
    start_i = int(np.searchsorted(time_arr, slice_start_t, side="left"))
    end_i_excl = int(np.searchsorted(time_arr, slice_end_t, side="right"))

    # Need at least T-1 timesteps of history available BEFORE the slice start.
    start_i = max(start_i, sequence_length - 1)
    indices = np.arange(start_i, end_i_excl, dtype=np.int64)

    if winter_filter and indices.size > 0:
        lead = int(ds.attrs.get("lead", 0))
        target_months = (
            time_arr[indices] + pd.DateOffset(months=lead)
        ).month
        keep = np.isin(target_months, [11, 12, 1, 2])
        indices = indices[keep]
    return indices


# ---------------------------------------------------------------------------
# LSTMExperimentRunner
# ---------------------------------------------------------------------------

class LSTMExperimentRunner:
    """Runs the Phase 12 LSTM experiment end-to-end.

    Parameters
    ----------
    exp_config
        Path to an experiment YAML (or a pre-loaded dict). See
        ``configs/experiments/exp_lstm.yaml`` for the expected schema. The
        ``lstm`` block carries:
          - ``grid``                       — dict of lists for the manual grid
          - ``representative_tuning_fold`` — 1 or null. When set, only that fold
                                              runs the grid; other folds reuse
                                              its locked best HPs.
          - ``max_sequence_length``        — int T used to size the gap during
                                              the grid SEARCH (so the same data
                                              window is used for every combo).
    """

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
        self.preds_dir = self._resolve(out.get("predictions_dir", "results/lstm/predictions"))
        self.metrics_dir = self._resolve(out.get("metrics_dir", "results/lstm/metrics"))
        self.logs_dir = self._resolve(out.get("logs_dir", "results/lstm/logs"))
        self.models_dir = self._resolve(out.get("models_dir", "results/lstm/models"))
        self.file_prefix: str = str(self.exp.get("file_prefix", "") or "")
        self.save_models: bool = bool(self.exp.get("save_models", False))
        self.winter_only_training: bool = bool(self.exp.get("winter_only_training", False))
        self.feature_overrides: dict[str, Any] = dict(self.exp.get("feature_overrides") or {})

        lstm_cfg = dict(self.exp.get("lstm") or {})
        self.lstm_grid: dict[str, list[Any]] = dict(lstm_cfg.get("grid") or {})
        self.representative_tuning_fold = lstm_cfg.get("representative_tuning_fold")
        max_T_grid = max(self.lstm_grid.get("sequence_length", [12]) or [12])
        self.max_sequence_length: int = int(lstm_cfg.get("max_sequence_length", max_T_grid))

        for d in (self.preds_dir, self.metrics_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.save_models:
            self.models_dir.mkdir(parents=True, exist_ok=True)

        # Default LSTM params from configs/models/lstm.yaml
        self._lstm_defaults = self._load_lstm_defaults()

        # State filled in by .run()
        self._datasets: dict[str, xr.Dataset] | None = None
        self._template: xr.DataArray | None = None
        self._morocco_mask: xr.DataArray | None = None
        self._cv: dcv.RollingOriginCV | None = None
        self._fold_runs: list[FoldRunLog] = []
        # Per-lead locked best HP combo (filled by the Fold-1 grid search)
        self._locked_combo_per_lead: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _resolve(p: str | Path) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _load_lstm_defaults(self) -> dict[str, Any]:
        path = PROJECT_ROOT / "configs" / "models" / "lstm.yaml"
        if not path.exists():
            return {}
        with open(path) as f:
            return (yaml.safe_load(f) or {}).get("params", {}) or {}

    # -----------------------------------------------------------------------
    # Data loading (mirrors ExperimentRunner.setup_data)
    # -----------------------------------------------------------------------
    def setup_data(self) -> None:
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
        if self._cv is None:
            raise RuntimeError("LSTMExperimentRunner.setup_data() must be called first.")
        return self._cv

    # -----------------------------------------------------------------------
    # Per-fold preparation
    # -----------------------------------------------------------------------
    def _prepare_fold(
        self,
        lead: int,
        fold_spec: dcv.FoldSpec,
        *,
        sequence_length: int,
    ) -> _PreparedLSTMFold:
        """Build standardized (train, val, test, refit) slices for one (lead, fold).

        The boundary gap is ``lead + sequence_length + 1`` months. To stay on
        the existing CV plumbing without duplicating it, we pass
        ``max_lag = sequence_length − 1`` to ``RollingOriginCV.get_fold_indices``
        — the internal formula `L + max_lag + 2` then evaluates to
        `L + (T-1) + 2 = L + T + 1`. Exact match.
        """
        max_lag_for_gap = sequence_length - 1

        # 1. Build a single feature dataset for the entire timeline.
        include_seasonal = bool(self.feat_cfg["include_seasonal_encoding"])
        if self.feature_overrides.get("drop_seasonal_encoding"):
            include_seasonal = False
        # The LSTM does its own learning of the lag structure — no PACF/CCF.
        ds_feat = dfeat.build_dataset(
            self._datasets, lead=lead,
            contemporary=self.feat_cfg["contemporary_predictors"],
            lags={},
            include_seasonal=include_seasonal,
            include_spatial=self.feat_cfg["include_spatial_encoding"],
        )

        # 2. Fold indices with the LSTM gap.
        fi = self.cv.get_fold_indices(
            ds_feat["time"], fold_spec, max_lag=max_lag_for_gap, lead=lead,
        )

        train_full_2d = ds_feat.isel(time=fi.train_idx)
        val_full_2d = ds_feat.isel(time=fi.val_idx)
        test_2d = ds_feat.isel(time=fi.test_idx)

        refit_start = int(fi.train_idx[0])
        refit_stop = int(fi.test_idx[0]) - fi.boundary_gap   # exclusive
        refit_full_2d = ds_feat.isel(time=slice(refit_start, refit_stop))

        # 3. Optional v4 winter-only training filter.
        if self.winter_only_training:
            train_2d = _filter_to_winter_targets(train_full_2d)
            val_2d = _filter_to_winter_targets(val_full_2d)
            refit_2d = _filter_to_winter_targets(refit_full_2d)
        else:
            train_2d, val_2d, refit_2d = train_full_2d, val_full_2d, refit_full_2d

        # 4. Fold-wise standardization. Fit on the (possibly winter-filtered)
        # train slice. Apply the SAME stats to every slice the LSTM sees and
        # to the baseline-only `*_full` slices.
        std = dcv.FoldStandardizer.from_config(
            self.cv_cfg, region_mask=self._morocco_mask
        ).fit(train_2d)
        train_n = std.transform(train_2d).where(self._morocco_mask)
        val_n = std.transform(val_2d).where(self._morocco_mask)
        test_n = std.transform(test_2d).where(self._morocco_mask)
        refit_n = std.transform(refit_2d).where(self._morocco_mask)
        train_full_n = std.transform(train_full_2d).where(self._morocco_mask)
        val_full_n = std.transform(val_full_2d).where(self._morocco_mask)
        refit_full_n = std.transform(refit_full_2d).where(self._morocco_mask)

        return _PreparedLSTMFold(
            fold_index=fi.index,
            lead=lead,
            sequence_length=sequence_length,
            boundary_gap=fi.boundary_gap,
            train=train_n, val=val_n, test=test_n, refit=refit_n,
            train_full=train_full_n, val_full=val_full_n, refit_full=refit_full_n,
            train_unstd=train_2d,
            time_test=test_n["time"].values,
            target_test=test_n["target"].values,
        )

    # -----------------------------------------------------------------------
    # Sliding-window tensor assembly tied to a per-fold slice
    # -----------------------------------------------------------------------
    def _build_window_tensors(
        self,
        full_ds: xr.Dataset,
        slice_ds: xr.Dataset,
        sequence_length: int,
        *,
        winter_filter: bool,
    ) -> tuple[np.ndarray, np.ndarray, SequenceMeta]:
        """Build (X, y, meta) tensors whose END indices fall inside `slice_ds`.

        `full_ds` is the unfiltered, standardized, full-timeline dataset (so the
        LSTM can read backward into pre-slice months — legitimate, see
        ``_train_window_end_indices``). `slice_ds.time[0]` and ``[-1]`` mark
        which timestamps are eligible to be sample CENTERS.
        """
        slice_times = pd.DatetimeIndex(slice_ds["time"].values)
        if slice_times.size == 0:
            return (
                np.zeros((0, sequence_length, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                SequenceMeta(
                    feature_names=[],
                    sample_t_idx=np.zeros((0,), dtype=np.int64),
                    sample_lat_idx=np.zeros((0,), dtype=np.int64),
                    sample_lon_idx=np.zeros((0,), dtype=np.int64),
                    sequence_length=sequence_length,
                    template_shape=(0, 0, 0),
                ),
            )

        end_indices = _train_window_end_indices(
            full_ds,
            slice_times[0],
            slice_times[-1],
            sequence_length,
            winter_filter=winter_filter,
        )
        return build_sequences(
            full_ds,
            sequence_length=sequence_length,
            end_indices=end_indices,
            cell_mask=self._morocco_mask,
            target_filter=None,         # already filtered via end_indices
            drop_nan_samples=True,
        )

    # -----------------------------------------------------------------------
    # Fit / predict — LSTM
    # -----------------------------------------------------------------------
    def _fit_lstm_combo(
        self,
        full_ds: xr.Dataset,
        train_slice: xr.Dataset,
        val_slice: xr.Dataset | None,
        combo: dict[str, Any],
    ) -> tuple[LSTMModel, dict[str, Any]]:
        """Fit one LSTMModel under the given hyperparameter combo. Returns
        ``(model, info)`` where info carries val loss + training duration."""
        T = int(combo["sequence_length"])
        X_train, y_train, meta = self._build_window_tensors(
            full_ds, train_slice, T, winter_filter=self.winter_only_training,
        )
        X_val = y_val = None
        if val_slice is not None and val_slice.sizes.get("time", 0) > 0:
            X_val, y_val, _ = self._build_window_tensors(
                full_ds, val_slice, T, winter_filter=self.winter_only_training,
            )

        params = {**self._lstm_defaults, **combo}
        model = LSTMModel(**params)
        t0 = time.time()
        model.fit_tensors(
            X_train, y_train,
            X_val if X_val is not None and len(X_val) > 0 else None,
            y_val if y_val is not None and len(y_val) > 0 else None,
            feature_names=meta.feature_names,
        )
        info = {
            "fit_duration_s": time.time() - t0,
            "n_train_samples": int(len(X_train)),
            "n_val_samples": int(len(X_val) if X_val is not None else 0),
            "best_val_loss": float(model.fit_state_.best_val_loss) if model.fit_state_ else float("nan"),
            "best_epoch": int(model.fit_state_.best_epoch) if model.fit_state_ else 0,
            "epochs_run": int(model.fit_state_.epochs_run) if model.fit_state_ else 0,
        }
        return model, info

    def _predict_lstm_on_test(
        self,
        full_ds: xr.Dataset,
        test_slice: xr.Dataset,
        model: LSTMModel,
        sequence_length: int,
    ) -> np.ndarray:
        """Predict on the test slice and reshape back to the test grid.

        The test slice is NEVER winter-filtered — we score against all months.
        Returns a `(n_test_time, n_lat, n_lon)` array with NaN where no
        sample existed (off-mask or NaN features).
        """
        X_test, _, meta = self._build_window_tensors(
            full_ds, test_slice, sequence_length, winter_filter=False,
        )
        if len(X_test) == 0:
            return np.full(
                (
                    test_slice.sizes["time"],
                    test_slice.sizes["lat"],
                    test_slice.sizes["lon"],
                ),
                np.nan,
                dtype=np.float32,
            )
        preds_flat = model.predict_tensors(X_test)

        # The `meta` is anchored at `full_ds`'s template shape. We need to scatter
        # ONLY into the test slice's grid, with correct local time indexing.
        test_time_arr = pd.DatetimeIndex(test_slice["time"].values)
        full_time_arr = pd.DatetimeIndex(full_ds["time"].values)
        # Map global feature-time index → local test-time index.
        # `meta.sample_t_idx` are global indices into `full_ds.time`.
        # We need them as local indices into `test_slice.time`.
        global_to_local = {
            int(np.searchsorted(full_time_arr, ts)): i
            for i, ts in enumerate(test_time_arr)
        }
        local_t = np.array(
            [global_to_local.get(int(g), -1) for g in meta.sample_t_idx], dtype=np.int64
        )
        valid = local_t >= 0

        n_t = test_slice.sizes["time"]
        n_lat = test_slice.sizes["lat"]
        n_lon = test_slice.sizes["lon"]
        out = np.full((n_t, n_lat, n_lon), np.nan, dtype=np.float32)
        out[local_t[valid], meta.sample_lat_idx[valid], meta.sample_lon_idx[valid]] = (
            preds_flat[valid]
        )
        return out

    # -----------------------------------------------------------------------
    # Baselines (climatology, persistence) — reused 1:1 from the tabular pipeline
    # -----------------------------------------------------------------------
    def _fit_predict_baseline(self, name: str, prep: _PreparedLSTMFold) -> np.ndarray:
        """Fit a baseline on the v3-equivalent unfiltered refit slice and
        predict on the test slice. Returns a `(n_test_time, n_lat, n_lon)`
        ndarray."""
        if name not in _BASELINE_MODELS:
            raise ValueError(f"_fit_predict_baseline: expected a baseline, got {name!r}")
        model = get_model(name).fit(prep.refit_full)
        return model.predict(prep.test).values

    # -----------------------------------------------------------------------
    # Grid search (Fold 1)
    # -----------------------------------------------------------------------
    def _fold1_grid_search(self, lead: int) -> dict[str, Any]:
        """Run the 8-combo grid on Fold 1; return the best combo (lowest val MSE)."""
        fold_spec = next(f for f in self.cv.fold_specs if f.index == 1)
        # Use the MAX sequence length to size the gap, so the same (train, val)
        # split is reused for every combo in the search.
        prep_search = self._prepare_fold(
            lead, fold_spec, sequence_length=self.max_sequence_length,
        )

        # Full-timeline standardized dataset for backward-looking history.
        # Fit the standardizer on the UNSTANDARDIZED train slice — passing
        # prep_search.train (already-standardized) was the bug that left
        # full_ds in raw physical units on the 2026-06-30 run.
        full_ds_unstd = self._build_full_feature_dataset(lead)
        full_ds_std = self._standardize_full_using_train(full_ds_unstd, prep_search.train_unstd)

        combos = _make_grid_combos(self.lstm_grid)
        self._log(f"  Fold-1 LSTM grid: {len(combos)} combos at L={lead}")
        results: list[dict[str, Any]] = []
        best = None
        for i, combo in enumerate(combos, 1):
            T = int(combo["sequence_length"])
            t0 = time.time()
            model, info = self._fit_lstm_combo(
                full_ds_std, prep_search.train, prep_search.val, combo,
            )
            score = info["best_val_loss"]
            entry = {
                **combo,
                "val_loss": score,
                "fit_duration_s": info["fit_duration_s"],
                "n_train_samples": info["n_train_samples"],
                "n_val_samples": info["n_val_samples"],
                "epochs_run": info["epochs_run"],
                "best_epoch": info["best_epoch"],
            }
            results.append(entry)
            elapsed = time.time() - t0
            self._log(
                f"    [{i}/{len(combos)}] h={combo['hidden_units']:>2} "
                f"d={combo['dropout']:<3} T={T:<2} lr={combo['learning_rate']:<6}  "
                f"val_mse={score:7.4f}  ({elapsed:5.1f}s)"
            )
            if best is None or score < best["val_loss"]:
                best = entry
            # Free GPU/CPU memory between combos
            del model
        # Persist a search log so the run is reproducible.
        search_log_path = self.logs_dir / f"{self.file_prefix}lstm_grid_search_L{lead}.csv"
        pd.DataFrame(results).to_csv(search_log_path, index=False)
        self._log(
            f"  Fold-1 BEST L={lead}: "
            f"h={best['hidden_units']}, d={best['dropout']}, "
            f"T={best['sequence_length']}, lr={best['learning_rate']}  "
            f"(val_mse={best['val_loss']:.4f})  →  log saved to "
            f"{search_log_path.relative_to(PROJECT_ROOT)}"
        )
        return {
            k: best[k]
            for k in ("hidden_units", "dropout", "sequence_length", "learning_rate")
        }

    def _build_full_feature_dataset(self, lead: int) -> xr.Dataset:
        include_seasonal = bool(self.feat_cfg["include_seasonal_encoding"])
        if self.feature_overrides.get("drop_seasonal_encoding"):
            include_seasonal = False
        return dfeat.build_dataset(
            self._datasets, lead=lead,
            contemporary=self.feat_cfg["contemporary_predictors"],
            lags={},
            include_seasonal=include_seasonal,
            include_spatial=self.feat_cfg["include_spatial_encoding"],
        )

    def _standardize_full_using_train(
        self, full_ds: xr.Dataset, train_unstd: xr.Dataset
    ) -> xr.Dataset:
        """Fit a fresh FoldStandardizer on the **unstandardized** train slice and
        apply it to the full-timeline dataset.

        ``train_unstd`` MUST be in raw physical units. Passing an already-
        standardized slice here is the bug that caused the 2026-06-30 LSTM
        divergence: stats came out as (mu≈0, sigma≈1) so the transform was
        a no-op and the LSTM ingested raw precip/temperature values.
        """
        std = dcv.FoldStandardizer.from_config(
            self.cv_cfg, region_mask=self._morocco_mask
        ).fit(train_unstd)
        return std.transform(full_ds).where(self._morocco_mask)

    # -----------------------------------------------------------------------
    # Save helpers — mirror ExperimentRunner outputs
    # -----------------------------------------------------------------------
    def _save_predictions_all_leads(
        self,
        preds_per_lead: dict[int, dict[str, np.ndarray]],
        truth_per_lead: dict[int, np.ndarray],
        time_per_lead: dict[int, np.ndarray],
    ) -> None:
        leads = sorted(preds_per_lead.keys())
        lat = self._template["lat"].values
        lon = self._template["lon"].values

        ref_time = time_per_lead[leads[0]]
        for L in leads[1:]:
            if not np.array_equal(time_per_lead[L], ref_time):
                raise ValueError(f"Time coord mismatch between leads {leads[0]} and {L}.")
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
            attrs={"experiment": self.exp.get("name", "lstm")},
        )
        out = self.preds_dir / f"{self.file_prefix}pooled_allLeads.nc"
        ds.to_netcdf(out)
        self._log(f"wrote predictions → {out.relative_to(PROJECT_ROOT)}")

    def _evaluate_slice(
        self,
        preds: dict[str, np.ndarray],
        truth: np.ndarray,
        time_arr: np.ndarray,
        lead: int,
        fold_label,
    ) -> pd.DataFrame:
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
        n_folds = len(truth_per_fold)
        all_rows: list[pd.DataFrame] = []
        pooled_preds = {n: np.concatenate(arrs, axis=0) for n, arrs in preds_per_fold.items()}
        pooled_truth = np.concatenate(truth_per_fold, axis=0)
        pooled_time = np.concatenate(time_per_fold, axis=0)
        all_rows.append(self._evaluate_slice(pooled_preds, pooled_truth, pooled_time, lead, "pooled"))
        for i in range(n_folds):
            fold_preds = {n: arrs[i] for n, arrs in preds_per_fold.items()}
            all_rows.append(self._evaluate_slice(
                fold_preds, truth_per_fold[i], time_per_fold[i], lead, i + 1,
            ))
        return pd.concat([df for df in all_rows if not df.empty], ignore_index=True)

    def _save_metrics_all_leads(self, dfs_per_lead: list[pd.DataFrame]) -> pd.DataFrame:
        combined = pd.concat(dfs_per_lead, ignore_index=True)
        out = self.metrics_dir / f"{self.file_prefix}metrics_allLeads.csv"
        combined.to_csv(out, index=False)
        self._log(f"wrote metrics → {out.relative_to(PROJECT_ROOT)}  ({len(combined)} rows)")
        return combined

    def _save_fold_runs(self) -> None:
        if self._fold_runs:
            df = pd.DataFrame([asdict(r) for r in self._fold_runs])
            out = self.logs_dir / f"{self.file_prefix}fold_runs.csv"
            df.to_csv(out, index=False)
            self._log(f"wrote fold-runs log → {out.relative_to(PROJECT_ROOT)}  ({len(df)} rows)")

    def _save_lstm_model(self, model: LSTMModel, lead: int, fold_index: int) -> None:
        """Persist a fitted LSTMModel via joblib. Moves the torch module to CPU
        before pickling so the dump is portable across devices.

        On failure, deletes the partial file so a downstream load can't pick up
        a corrupt header-only stream (this bit us once when ``VariationalLSTM``
        was nested in a factory function and pickle silently produced 10-byte
        files — fixed 2026-06-30).
        """
        if not self.save_models:
            return
        import joblib
        path = self.models_dir / (
            f"{self.file_prefix}lstm_lead{lead}_fold{fold_index}.joblib"
        )
        try:
            if model.module_ is not None:
                model.module_ = model.module_.to("cpu")
            joblib.dump(model, path, compress=3)
            # Sanity: a successful compressed LSTM dump is several KB; anything
            # tiny is a header-only corrupt write.
            if path.stat().st_size < 64:
                raise RuntimeError(
                    f"joblib.dump produced a {path.stat().st_size}-byte file "
                    f"(suspiciously small — likely a pickle failure)"
                )
        except Exception as e:                                   # noqa: BLE001
            if path.exists():
                path.unlink()
            self._log(f"    [warn] could not save LSTM model {path.name}: {e}")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        t0 = time.time()
        self._log(f"=== LSTMExperimentRunner: '{self.exp.get('name', 'lstm')}' ===")

        self.setup_data()
        self._log(
            f"loaded data ({dict(self._template.sizes)}) + Morocco mask "
            f"({int(self._morocco_mask.sum())} cells)"
        )

        models_to_run = ["climatology", "persistence", "lstm"]

        preds_per_lead: dict[int, dict[str, np.ndarray]] = {}
        truth_per_lead: dict[int, np.ndarray] = {}
        time_per_lead: dict[int, np.ndarray] = {}
        metrics_dfs: list[pd.DataFrame] = []

        for lead in self.exp["leads"]:
            lead_t0 = time.time()
            self._log(f"\n--- lead L = {lead} ---")

            # 1. Fold-1 grid search → locked combo for this lead.
            if self.lstm_grid:
                locked = self._fold1_grid_search(lead)
            else:
                locked = {
                    k: self._lstm_defaults[k]
                    for k in ("hidden_units", "dropout", "sequence_length", "learning_rate")
                }
            self._locked_combo_per_lead[lead] = locked
            T_star = int(locked["sequence_length"])

            pooled_preds: dict[str, list[np.ndarray]] = {n: [] for n in models_to_run}
            truth_pool: list[np.ndarray] = []
            time_pool: list[np.ndarray] = []

            # Standardized full-timeline dataset, fit once per (fold, lead) below.
            full_ds_unstd = self._build_full_feature_dataset(lead)

            for fold_spec in self.cv.fold_specs:
                fold_t0 = time.time()
                self._log(f"  fold {fold_spec.index}  (locked: "
                          f"h={locked['hidden_units']}, d={locked['dropout']}, "
                          f"T={T_star}, lr={locked['learning_rate']})")

                # Prepare slices with the locked T's gap.
                prep = self._prepare_fold(lead, fold_spec, sequence_length=T_star)

                # Per-fold standardized FULL dataset (for backward-looking history).
                # Fit on UNSTANDARDIZED train so the transform actually rescales.
                full_ds_std = self._standardize_full_using_train(full_ds_unstd, prep.train_unstd)

                # ---- LSTM: refit on refit_slice, predict on test ----
                lstm_combo = {**locked}
                t_fit = time.time()
                lstm_model, info = self._fit_lstm_combo(
                    full_ds_std, prep.refit, val_slice=None, combo=lstm_combo,
                )
                lstm_fit_dur = time.time() - t_fit
                lstm_pred = self._predict_lstm_on_test(
                    full_ds_std, prep.test, lstm_model, T_star,
                )
                pooled_preds["lstm"].append(lstm_pred)
                self._save_lstm_model(lstm_model, lead, fold_spec.index)

                # ---- Baselines on the v3-equivalent unfiltered slices ----
                for bname in ("climatology", "persistence"):
                    bpred = self._fit_predict_baseline(bname, prep)
                    pooled_preds[bname].append(bpred)

                truth_pool.append(prep.target_test)
                time_pool.append(prep.time_test)

                # Log row
                self._fold_runs.append(FoldRunLog(
                    fold=prep.fold_index, lead=lead, model="lstm",
                    best_params=json.dumps(lstm_combo, default=float),
                    best_val_score=None,
                    best_iteration=info["best_epoch"],
                    n_features_total=len(lstm_model.feature_names_ or []),
                    n_features_retained=len(lstm_model.feature_names_ or []),
                    K_eff=T_star - 1,
                    boundary_gap=prep.boundary_gap,
                    search_duration_s=0.0,
                    fit_duration_s=lstm_fit_dur,
                    n_trials=0,
                ))
                self._log(
                    f"    LSTM fit={lstm_fit_dur:6.1f}s  epochs={info['epochs_run']}  "
                    f"n_train={info['n_train_samples']}  "
                    f"| fold done in {time.time() - fold_t0:.1f}s"
                )

            metrics_dfs.append(self._compute_metrics_for_lead(
                pooled_preds, truth_pool, time_pool, lead,
            ))
            preds_per_lead[lead] = {n: np.concatenate(p, axis=0) for n, p in pooled_preds.items()}
            truth_per_lead[lead] = np.concatenate(truth_pool, axis=0)
            time_per_lead[lead] = np.concatenate(time_pool, axis=0)
            self._log(f"--- lead L={lead} done in {time.time() - lead_t0:.1f}s ---")

        self._save_predictions_all_leads(preds_per_lead, truth_per_lead, time_per_lead)
        combined_metrics = self._save_metrics_all_leads(metrics_dfs)
        self._save_fold_runs()
        # Save the locked HPs per lead so the run is reproducible.
        with open(self.logs_dir / f"{self.file_prefix}lstm_locked_combos.json", "w") as f:
            json.dump(
                {str(L): combo for L, combo in self._locked_combo_per_lead.items()},
                f, indent=2, default=float,
            )
        self._log(f"\n=== done in {time.time() - t0:.1f}s ===")
        return combined_metrics