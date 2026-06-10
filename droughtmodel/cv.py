"""Cross-validation and fold-wise standardization (v3 §4–5).

Provides:
- `RollingOriginCV` — 5-fold rolling-origin with continuous test windows and
  an adaptive boundary-gap quarantine (see v3 §4.1)
- `FoldStandardizer` — fold-wise z-score standardization with a pre-standardized
  exception list (ENSO, NAO, MO, SPEI3, target — and their lagged variants)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from droughtmodel.utils import PROJECT_ROOT


def load_cv_config(path: str | Path = "configs/cv.yaml") -> dict[str, Any]:
    """Load the CV YAML config (paths anchored at PROJECT_ROOT if relative)."""
    p = Path(path)
    p = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FoldSpec:
    """Planned fold boundaries (before quarantine is applied)."""
    index: int
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class FoldIndices:
    """Final integer indices for a fold, with quarantine applied to train/val."""
    index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    boundary_gap: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_val(self) -> int:
        return int(len(self.val_idx))

    @property
    def n_test(self) -> int:
        return int(len(self.test_idx))

    def winter_mask(self, time_coord) -> np.ndarray:
        """Boolean array (size n_test) — True where the test target month is in [Nov, Feb]."""
        time_arr = pd.DatetimeIndex(time_coord.values if hasattr(time_coord, "values") else time_coord)
        months = time_arr[self.test_idx].month.values
        return np.isin(months, [11, 12, 1, 2])

    @property
    def n_test_winter(self) -> int | None:
        return None  # use .winter_mask(time_coord).sum() with the actual time coord


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------

class RollingOriginCV:
    """5-fold rolling-origin CV with continuous test windows and adaptive quarantine.

    Iteration is index-based: callers pass a `time_coord` (xarray time coord or
    pandas DatetimeIndex) and an integer `max_lag` (typically the per-fold max
    selected lag from PACF/CCF). The quarantine subtracts `max_lag` months from
    the END of train and the END of val. Test windows are never shrunk, so the
    per-fold test predictions stitch into a single continuous out-of-sample
    array exactly matching the planned test windows.
    """

    def __init__(self, folds: list[dict[str, str]], boundary_gap_months: int | None = None):
        self.fold_specs: list[FoldSpec] = [
            FoldSpec(
                index=i + 1,
                train_start=pd.Timestamp(f["train_start"]),
                val_start=pd.Timestamp(f["val_start"]),
                test_start=pd.Timestamp(f["test_start"]),
                test_end=pd.Timestamp(f["test_end"]),
            )
            for i, f in enumerate(folds)
        ]
        self.fixed_gap = boundary_gap_months  # None ⇒ adaptive

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "RollingOriginCV":
        config = config if config is not None else load_cv_config()
        return cls(folds=config["folds"], boundary_gap_months=config.get("boundary_gap_months"))

    @property
    def n_folds(self) -> int:
        return len(self.fold_specs)

    def _resolve_gap(self, max_lag: int) -> int:
        return int(self.fixed_gap) if self.fixed_gap is not None else int(max_lag)

    @staticmethod
    def _as_time_index(time_coord) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(time_coord.values if hasattr(time_coord, "values") else time_coord)

    def get_fold_indices(
        self,
        time_coord,
        fold: FoldSpec,
        max_lag: int = 0,
    ) -> FoldIndices:
        """Compute integer indices for one fold with the quarantine applied.

        Parameters
        ----------
        time_coord
            Monthly time coordinate (xarray time coord or pandas DatetimeIndex).
        fold
            Planned fold boundaries (`FoldSpec`).
        max_lag
            Maximum selected lag for this fold (used as boundary gap unless
            `boundary_gap_months` is fixed in config).
        """
        time_arr = self._as_time_index(time_coord)
        gap = self._resolve_gap(max_lag)

        train_start_i = int(np.searchsorted(time_arr, fold.train_start, side="left"))
        val_start_i = int(np.searchsorted(time_arr, fold.val_start, side="left"))
        test_start_i = int(np.searchsorted(time_arr, fold.test_start, side="left"))
        # test_end is INCLUSIVE; searchsorted with 'right' gives one past.
        test_end_i_excl = int(np.searchsorted(time_arr, fold.test_end, side="right"))

        # Quarantine: trim end of train, end of val.
        train_end_i = max(val_start_i - gap, train_start_i)
        val_end_i = max(test_start_i - gap, val_start_i)

        train_idx = np.arange(train_start_i, train_end_i)
        val_idx = np.arange(val_start_i, val_end_i)
        test_idx = np.arange(test_start_i, test_end_i_excl)

        # Safe lookups for boundary timestamps
        def _get(i: int, fallback: pd.Timestamp) -> pd.Timestamp:
            return time_arr[i] if 0 <= i < len(time_arr) else fallback

        return FoldIndices(
            index=fold.index,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            boundary_gap=gap,
            train_start=_get(train_start_i, fold.train_start),
            train_end=_get(train_end_i - 1, fold.train_start),
            val_start=_get(val_start_i, fold.val_start),
            val_end=_get(val_end_i - 1, fold.val_start),
            test_start=_get(test_start_i, fold.test_start),
            test_end=_get(test_end_i_excl - 1, fold.test_end),
        )

    def folds(self, time_coord, max_lag: int = 0) -> Iterator[FoldIndices]:
        """Yield `FoldIndices` for each fold using `max_lag` as the boundary gap.

        Pass a single `max_lag` for all folds (e.g. a fixed worst-case 12), OR
        call `get_fold_indices` per fold with a per-fold max_lag computed from
        per-fold PACF/CCF (the recommended pipeline path).
        """
        for spec in self.fold_specs:
            yield self.get_fold_indices(time_coord, spec, max_lag=max_lag)


# ---------------------------------------------------------------------------
# Fold-wise standardizer
# ---------------------------------------------------------------------------

def _apply_region_mask(data: xr.Dataset, region_mask: xr.DataArray | None) -> xr.Dataset:
    """Return a copy of `data` with non-region cells set to NaN on spatial variables.

    Non-spatial variables (1-D time series, scalar coords) are returned unchanged.
    """
    if region_mask is None:
        return data
    masked = data.copy()
    for var in masked.data_vars:
        da = masked[var]
        if "lat" in da.dims and "lon" in da.dims:
            masked[var] = da.where(region_mask)
    return masked


class FoldStandardizer:
    """Fold-wise z-score standardization (pooled across all cells).

    Per v3 §5 + Morocco-only modeling scope: variables listed in `exceptions` are
    NOT re-standardized (they are already on a standardized scale by construction
    or by provider). Lagged variants matching `<base>_lag{k}` for any `base` in
    `exceptions` are also excluded automatically.

    Parameters
    ----------
    exceptions
        Variables to leave untouched (and their `_lag{k}` variants).
    region_mask
        Optional 2-D boolean `xr.DataArray` (dims `lat`, `lon`). When provided,
        statistics are computed using **only the masked-in cells** (e.g. Morocco
        only). Transform applies those same statistics to whatever data is passed;
        cells outside the mask will receive z-scores based on the masked-in
        distribution (the downstream pipeline ignores those cells anyway). This
        is the recommended setup for our Morocco-only modeling scope.

    Workflow:
        std = FoldStandardizer(
            exceptions=['enso', 'nao', 'mo', 'spei3', 'target'],
            region_mask=morocco_mask,
        )
        std.fit(train_dataset)               # stats from Morocco cells only
        train_n = std.transform(train_dataset)
        val_n   = std.transform(val_dataset)
        test_n  = std.transform(test_dataset)
    """

    def __init__(
        self,
        exceptions: list[str] | None = None,
        region_mask: xr.DataArray | None = None,
    ):
        self.exceptions: set[str] = set(exceptions or [])
        self.region_mask = region_mask
        self.stats: dict[str, tuple[float, float]] = {}
        self._fitted = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None = None,
        region_mask: xr.DataArray | None = None,
    ) -> "FoldStandardizer":
        config = config if config is not None else load_cv_config()
        return cls(
            exceptions=config.get("standardization_exceptions", []),
            region_mask=region_mask,
        )

    def is_excepted(self, var_name: str) -> bool:
        """Match either the exact name or `<base>_lag{k}` where base is excepted."""
        if var_name in self.exceptions:
            return True
        if "_lag" in var_name:
            base = var_name.rsplit("_lag", 1)[0]
            if base in self.exceptions:
                return True
        return False

    def fit(self, data: xr.Dataset) -> "FoldStandardizer":
        """Compute per-variable mean and std on `data`, optionally restricted to the region mask."""
        self.stats = {}
        masked = _apply_region_mask(data, self.region_mask)
        for var in masked.data_vars:
            if self.is_excepted(var):
                continue
            arr = np.asarray(masked[var].values)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            mu = float(finite.mean())
            sigma = float(finite.std())
            self.stats[var] = (mu, sigma if sigma > 0 else 1.0)
        self._fitted = True
        return self

    def transform(self, data: xr.Dataset) -> xr.Dataset:
        """Apply standardization to `data`. Excepted variables pass through unchanged.

        Note: this transforms the WHOLE grid using the (Morocco-derived) stats.
        Downstream the pipeline restricts modeling to Morocco cells anyway, so
        the standardized values at non-Morocco cells are harmless.
        """
        if not self._fitted:
            raise RuntimeError("FoldStandardizer must be `.fit()` before `.transform()`.")
        out = data.copy()
        for var, (mu, sigma) in self.stats.items():
            if var in out.data_vars:
                out[var] = (out[var] - mu) / sigma
        return out

    def fit_transform(self, data: xr.Dataset) -> xr.Dataset:
        return self.fit(data).transform(data)

    @property
    def standardized_vars(self) -> list[str]:
        return sorted(self.stats.keys())

    @property
    def excepted_vars_seen(self) -> list[str]:
        """Exception list as-configured (may include names not present in data)."""
        return sorted(self.exceptions)


class PerCellStandardizer:
    """Per-cell z-score standardization, scoped to a region mask.

    For each (lat, lon) cell of a gridded variable, computes its own (μ, σ) along
    the time axis and standardizes that cell's time series independently. This is
    the natural choice for the **per-cell sensitivity test** (v3 §7.2).

    Parameters
    ----------
    exceptions
        Variables to leave untouched (and their `_lag{k}` variants).
    region_mask
        Optional 2-D boolean `xr.DataArray`. When provided:
          - Stats are computed **only for masked-in cells**
          - Cells outside the mask receive μ=0, σ=1 (identity transform — preserves
            original values harmlessly, since the downstream pipeline ignores them)
        Recommended for our Morocco-only modeling scope (~164 cells vs 4096).

    Behavior per dim signature:
      - (time, lat, lon) gridded vars  →  per-cell (μ, σ), broadcast over time
      - (time,) variables              →  pooled scalar (μ, σ) (mask doesn't apply)
      - (lat, lon) variables           →  pooled scalar (μ, σ) — but typically these
        (e.g. `lat_feat`, `lon_feat`) are dropped from per-cell experiments anyway
        (zero variance within a cell makes them useless features)
      - excepted vars                  →  pass-through, unchanged

    Defensive against degenerate cells: σ ≤ 0 or NaN → replaced with 1.0;
    NaN μ → replaced with 0.0. These cells then transform to (x − 0) / 1 = x.
    """

    def __init__(
        self,
        exceptions: list[str] | None = None,
        region_mask: xr.DataArray | None = None,
    ):
        self.exceptions: set[str] = set(exceptions or [])
        self.region_mask = region_mask
        # stats[var] = (mu, sigma); each may be a scalar OR a 2-D (lat, lon) DataArray
        self.stats: dict[str, tuple[Any, Any]] = {}
        self._fitted = False

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None = None,
        region_mask: xr.DataArray | None = None,
    ) -> "PerCellStandardizer":
        config = config if config is not None else load_cv_config()
        return cls(
            exceptions=config.get("standardization_exceptions", []),
            region_mask=region_mask,
        )

    def is_excepted(self, var_name: str) -> bool:
        """Same matching logic as `FoldStandardizer.is_excepted`."""
        if var_name in self.exceptions:
            return True
        if "_lag" in var_name:
            base = var_name.rsplit("_lag", 1)[0]
            if base in self.exceptions:
                return True
        return False

    def fit(self, data: xr.Dataset) -> "PerCellStandardizer":
        """Compute (μ, σ) per variable: per-cell for (time, lat, lon) vars; pooled scalar otherwise.

        When `region_mask` is set, per-cell stats are only computed for masked-in cells;
        cells outside the mask receive μ=0, σ=1 (so transform is identity for them).
        """
        self.stats = {}
        masked = _apply_region_mask(data, self.region_mask)
        for var in masked.data_vars:
            if self.is_excepted(var):
                continue
            da = masked[var]
            dims = set(da.dims)
            if {"time", "lat", "lon"}.issubset(dims):
                mu = da.mean(dim="time", skipna=True)
                sigma = da.std(dim="time", skipna=True)
                # Guard against zero/NaN std (degenerate cells, or cells outside mask)
                sigma = sigma.where(sigma > 0).fillna(1.0)
                mu = mu.fillna(0.0)
                self.stats[var] = (mu, sigma)
            else:
                arr = np.asarray(da.values)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                mu_s = float(finite.mean())
                sigma_s = float(finite.std())
                self.stats[var] = (mu_s, sigma_s if sigma_s > 0 else 1.0)
        self._fitted = True
        return self

    def transform(self, data: xr.Dataset) -> xr.Dataset:
        """Apply standardization. Excepted variables pass through unchanged.

        Non-mask cells receive (μ=0, σ=1) and so pass through with their original values.
        """
        if not self._fitted:
            raise RuntimeError("PerCellStandardizer must be `.fit()` before `.transform()`.")
        out = data.copy()
        for var, (mu, sigma) in self.stats.items():
            if var in out.data_vars:
                out[var] = (out[var] - mu) / sigma
        return out

    def fit_transform(self, data: xr.Dataset) -> xr.Dataset:
        return self.fit(data).transform(data)

    @property
    def standardized_vars(self) -> list[str]:
        return sorted(self.stats.keys())

    @property
    def excepted_vars_seen(self) -> list[str]:
        return sorted(self.exceptions)
