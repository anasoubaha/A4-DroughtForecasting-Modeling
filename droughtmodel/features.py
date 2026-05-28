"""Feature engineering for SPEI3 forecasting (v2 scheme Section 3).

Builds Dataset_L1, Dataset_L3, Dataset_L6 from the canonical inputs (returned by
`droughtmodel.data.load_all`). Provides:

- PACF / CCF lag-selection primitives (run per-fold in the pipeline)
- `build_lag_features`, `add_seasonal_encoding`, `add_spatial_encoding`
- `build_target` (SPEI3 shifted by L months) and `apply_winter_mask`
- `build_dataset` (assembles full per-lead feature+target xarray Dataset)
- `select_lags_from_training` (PACF + CCF on spatially-averaged training data)

Variables in the assembled dataset keep their natural dimensions:
    gridded vars   → (time, lat, lon)
    climate indices → (time,)
    seasonal encoding → (time,)
    spatial encoding → (lat, lon)
    target         → (time, lat, lon)
    winter_mask    → (time,)

Downstream code (CV / pipeline) broadcasts these into a tabular per-sample form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from statsmodels.tsa.stattools import pacf

from droughtmodel.utils import PROJECT_ROOT


def load_features_config(path: str | Path = "configs/features.yaml") -> dict[str, Any]:
    """Load the feature-engineering YAML config."""
    p = Path(path)
    p = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# PACF / CCF primitives
# ---------------------------------------------------------------------------

def compute_pacf(series: np.ndarray, n_lags: int = 12) -> np.ndarray:
    """PACF for a 1-D series; returns an array of length `n_lags` (excludes lag 0).

    NaN-safe (NaNs are dropped before fitting). Returns all-NaN if the series is
    too short to fit reliably (< max(3 * n_lags, 30) finite samples).
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < max(3 * n_lags, 30):
        return np.full(n_lags, np.nan)
    vals = pacf(s, nlags=n_lags, method="ols")
    return vals[1:]


def compute_ccf(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 12,
    target_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-correlation: ccf[k-1] = corr(x[t-k], y[t]) for k = 1..max_lag.

    Standardization is computed per-lag on the subset of valid (x[t-k], y[t]) pairs,
    so each returned value is a proper Pearson correlation on the points actually used.

    Parameters
    ----------
    target_mask
        Optional boolean array (same length as `y`). When provided, only pairs where
        `target_mask[t]` is True are used — e.g. pass a winter-month mask to compute
        the CCF on Nov-Feb target months only.

    Returns
    -------
    Array of shape (max_lag,). All-NaN if any lag has too few finite paired samples.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length (got {len(x)} vs {len(y)})")
    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool)
        if len(target_mask) != len(y):
            raise ValueError(f"target_mask must match y length (got {len(target_mask)} vs {len(y)})")

    results = np.full(max_lag, np.nan)
    for k in range(1, max_lag + 1):
        xs = x[: len(x) - k]
        ys = y[k:]
        if target_mask is not None:
            tm = target_mask[k:]
            xs = xs[tm]
            ys = ys[tm]
        valid = ~(np.isnan(xs) | np.isnan(ys))
        n = int(valid.sum())
        if n < max(3 * max_lag, 30):
            continue
        xv = xs[valid]
        yv = ys[valid]
        if xv.std() == 0 or yv.std() == 0:
            continue
        results[k - 1] = float(np.corrcoef(xv, yv)[0, 1])
    return results


def select_lags_pacf(series: np.ndarray, threshold: float = 0.20, n_lags: int = 12) -> list[int]:
    """Return lags (1-indexed) where |PACF| > threshold."""
    vals = compute_pacf(series, n_lags=n_lags)
    return [i + 1 for i, p in enumerate(vals) if not np.isnan(p) and abs(p) > threshold]


def select_lags_ccf(
    x: np.ndarray, y: np.ndarray, max_lag: int = 12, threshold: float = 0.20
) -> list[int]:
    """Return lags (1-indexed) where |CCF| > threshold."""
    vals = compute_ccf(x, y, max_lag=max_lag)
    return [i + 1 for i, c in enumerate(vals) if not np.isnan(c) and abs(c) > threshold]


def spatial_mean(da: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """Mean over (lat, lon) for gridded vars, optionally restricted to a region mask.

    Parameters
    ----------
    da
        Gridded or 1-D DataArray.
    mask
        Optional 2-D boolean DataArray with dims (lat, lon). If provided, only
        cells where mask is True contribute to the mean. No-op for 1-D series.
    """
    dims = [d for d in ("lat", "lon") if d in da.dims]
    if not dims:
        return da
    if mask is not None:
        da = da.where(mask)
    return da.mean(dim=dims, skipna=True)


def load_region_mask(
    shapefile_path: str | Path,
    template: xr.DataArray,
    name: str = "region",
) -> xr.DataArray:
    """Load a shapefile and return a 2-D boolean mask on `template`'s (lat, lon) grid.

    True where the cell center is inside (any) polygon in the shapefile.
    Requires `geopandas` and `regionmask`.
    """
    import geopandas as gpd
    import regionmask

    path = Path(shapefile_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    gdf = gpd.read_file(path)
    region = regionmask.from_geopandas(gdf, name=name)
    mask = region.mask(template["lon"].values, template["lat"].values)
    return xr.DataArray(
        ~np.isnan(mask.values),
        dims=("lat", "lon"),
        coords={"lat": template["lat"], "lon": template["lon"]},
        name=f"{name}_mask",
    )


def _winter_mask_from_time(time_coord) -> np.ndarray:
    """Numpy boolean array, True where month ∈ {11, 12, 1, 2}."""
    return np.isin(pd.DatetimeIndex(time_coord).month.values, [11, 12, 1, 2])


# ---------------------------------------------------------------------------
# Feature builders (operate on one variable / one tensor)
# ---------------------------------------------------------------------------

def gather_predictor(name: str, datasets: dict[str, xr.Dataset]) -> xr.DataArray:
    """Look up a canonical variable name in the dataset collection."""
    for d in datasets.values():
        if isinstance(d, xr.Dataset) and name in d.data_vars:
            return d[name]
    raise KeyError(f"Predictor '{name}' not found in any of: {list(datasets)}")


def build_lag_features(da: xr.DataArray, lags: list[int]) -> dict[str, xr.DataArray]:
    """For each k in `lags`, return `{name}_lag{k}` = da.shift(time=k)."""
    name = da.name
    return {f"{name}_lag{k}": da.shift(time=k) for k in lags}


def add_seasonal_encoding(time_coord) -> dict[str, xr.DataArray]:
    """Sin/cos of month-of-year, indexed by `time_coord`."""
    months = pd.DatetimeIndex(time_coord).month.values
    sin_m = np.sin(2 * np.pi * months / 12)
    cos_m = np.cos(2 * np.pi * months / 12)
    return {
        "sin_month": xr.DataArray(sin_m, dims="time", coords={"time": time_coord}),
        "cos_month": xr.DataArray(cos_m, dims="time", coords={"time": time_coord}),
    }


def add_spatial_encoding(lat: xr.DataArray, lon: xr.DataArray) -> dict[str, xr.DataArray]:
    """Broadcast lat and lon to (lat, lon) for use as features."""
    lat2d, lon2d = xr.broadcast(lat, lon)
    return {"lat_feat": lat2d, "lon_feat": lon2d}


def build_target(spei3: xr.DataArray, lead: int) -> xr.DataArray:
    """y_t = SPEI3(t + L). Shift backward by `lead` so target[t] = spei3[t+L]."""
    out = spei3.shift(time=-lead)
    out.name = "target"
    out.attrs["lead"] = lead
    return out


def apply_winter_mask(time_coord) -> xr.DataArray:
    """Bool array along `time`: True where month ∈ {11, 12, 1, 2}."""
    months = pd.DatetimeIndex(time_coord).month.values
    return xr.DataArray(
        np.isin(months, [11, 12, 1, 2]),
        dims="time",
        coords={"time": time_coord},
        name="winter_mask",
    )


# ---------------------------------------------------------------------------
# Lag selection driver (run on training-fold data only)
# ---------------------------------------------------------------------------

def _lag_spectrum_for_var(
    var_name: str,
    datasets: dict[str, xr.Dataset],
    ccf_target_name: str,
    n_lags: int,
    ccf_max_lag: int,
    region_mask: xr.DataArray | None,
    winter_mask: np.ndarray | None,
    aggregation_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (PACF, CCF) for one variable using the requested aggregation.

    Returns
    -------
    (pacf_vals, ccf_vals) — arrays of shape (n_lags,) and (ccf_max_lag,).

    Aggregation modes
    -----------------
    spatial_mean (approach A)
        Apply `region_mask` (if any), spatial-mean the variable → one 1-D series →
        PACF and CCF computed once.
    per_cell_then_mean (approach B)
        For each cell inside `region_mask` (or every cell if None), compute PACF
        on the cell's own series and CCF against the *same cell's* SPEI3 series,
        then average those arrays across cells.

    For 1-D variables (climate indices), both modes are equivalent and use the
    1-D series directly.
    """
    da = gather_predictor(var_name, datasets)
    is_gridded = "lat" in da.dims and "lon" in da.dims

    if not is_gridded:
        target = spatial_mean(gather_predictor(ccf_target_name, datasets), mask=region_mask).values
        series = da.values
        pacf_vals = compute_pacf(series, n_lags=n_lags)
        ccf_vals = compute_ccf(series, target, max_lag=ccf_max_lag, target_mask=winter_mask)
        return pacf_vals, ccf_vals

    if aggregation_mode == "spatial_mean":
        target = spatial_mean(gather_predictor(ccf_target_name, datasets), mask=region_mask).values
        series = spatial_mean(da, mask=region_mask).values
        pacf_vals = compute_pacf(series, n_lags=n_lags)
        ccf_vals = compute_ccf(series, target, max_lag=ccf_max_lag, target_mask=winter_mask)
        return pacf_vals, ccf_vals

    if aggregation_mode == "per_cell_then_mean":
        target_da = gather_predictor(ccf_target_name, datasets)
        # Determine which cells participate
        if region_mask is not None:
            lat_idx, lon_idx = np.where(region_mask.values)
        else:
            sh = da.isel(time=0).shape
            lat_idx, lon_idx = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), indexing="ij")
            lat_idx = lat_idx.flatten()
            lon_idx = lon_idx.flatten()

        pacf_stack = np.full((len(lat_idx), n_lags), np.nan)
        ccf_stack = np.full((len(lat_idx), ccf_max_lag), np.nan)
        da_vals = da.values  # (time, lat, lon)
        tgt_vals = target_da.values  # (time, lat, lon)
        for n, (i, j) in enumerate(zip(lat_idx, lon_idx)):
            x_series = da_vals[:, i, j]
            y_series = tgt_vals[:, i, j]
            pacf_stack[n] = compute_pacf(x_series, n_lags=n_lags)
            ccf_stack[n] = compute_ccf(x_series, y_series, max_lag=ccf_max_lag, target_mask=winter_mask)
        pacf_vals = np.nanmean(pacf_stack, axis=0)
        ccf_vals = np.nanmean(ccf_stack, axis=0)
        return pacf_vals, ccf_vals

    raise ValueError(
        f"aggregation_mode must be 'spatial_mean' or 'per_cell_then_mean', got: {aggregation_mode!r}"
    )


def select_lags_from_training(
    datasets_train: dict[str, xr.Dataset],
    long_memory_vars: list[str],
    fast_response_vars: list[str],
    pacf_threshold: float = 0.20,
    pacf_n_lags: int = 12,
    ccf_target: str = "spei3",
    ccf_threshold: float = 0.20,
    ccf_max_lag: int = 12,
    region_mask: xr.DataArray | None = None,
    winter_only_ccf: bool = False,
    aggregation_mode: str = "spatial_mean",
) -> tuple[dict[str, list[int]], dict[str, dict]]:
    """PACF ∪ CCF lag selection on training-period data.

    Long-memory variables: tested up to `pacf_n_lags` lags.
    Fast-response variables: tested at lag 1 only.

    Parameters
    ----------
    region_mask
        Optional 2-D boolean mask (built by `load_region_mask`). If provided,
        spatial aggregation is restricted to cells inside the mask.
    winter_only_ccf
        If True, CCF is computed only over target months t ∈ {Nov, Dec, Jan, Feb}.
        PACF is unaffected (within-variable autocorrelation is season-independent
        for our purposes).
    aggregation_mode
        ``"spatial_mean"`` (approach A) or ``"per_cell_then_mean"`` (approach B).
        See `_lag_spectrum_for_var` for details.

    Returns
    -------
    lags : dict
        {var_name: [list of selected lags]}
    diagnostics : dict
        {'pacf': {var: array}, 'ccf': {var: array}, 'config': {...}}
    """
    pacf_diag: dict[str, np.ndarray] = {}
    ccf_diag: dict[str, np.ndarray] = {}
    selected_lags: dict[str, list[int]] = {}

    # Build winter mask once if requested
    winter_mask = None
    if winter_only_ccf:
        time_coord = gather_predictor(ccf_target, datasets_train)["time"].values
        winter_mask = _winter_mask_from_time(time_coord)

    for var in long_memory_vars:
        pacf_vals, ccf_vals = _lag_spectrum_for_var(
            var, datasets_train, ccf_target, pacf_n_lags, ccf_max_lag,
            region_mask, winter_mask, aggregation_mode,
        )
        pacf_diag[var] = pacf_vals
        ccf_diag[var] = ccf_vals
        pacf_set = {i + 1 for i, p in enumerate(pacf_vals) if not np.isnan(p) and abs(p) > pacf_threshold}
        ccf_set = {i + 1 for i, c in enumerate(ccf_vals) if not np.isnan(c) and abs(c) > ccf_threshold}
        selected_lags[var] = sorted(pacf_set | ccf_set)

    for var in fast_response_vars:
        pacf_vals, ccf_vals = _lag_spectrum_for_var(
            var, datasets_train, ccf_target, 1, 1,
            region_mask, winter_mask, aggregation_mode,
        )
        pacf_diag[var] = pacf_vals
        ccf_diag[var] = ccf_vals
        keep = (not np.isnan(pacf_vals[0]) and abs(pacf_vals[0]) > pacf_threshold) or (
            not np.isnan(ccf_vals[0]) and abs(ccf_vals[0]) > ccf_threshold
        )
        selected_lags[var] = [1] if keep else []

    diagnostics = {
        "pacf": pacf_diag,
        "ccf": ccf_diag,
        "config": {
            "pacf_threshold": pacf_threshold,
            "ccf_threshold": ccf_threshold,
            "winter_only_ccf": winter_only_ccf,
            "aggregation_mode": aggregation_mode,
            "region_masked": region_mask is not None,
        },
    }
    return selected_lags, diagnostics


# ---------------------------------------------------------------------------
# Full feature dataset assembly
# ---------------------------------------------------------------------------

def build_dataset(
    datasets: dict[str, xr.Dataset],
    lead: int,
    contemporary: list[str],
    lags: dict[str, list[int]] | None = None,
    include_seasonal: bool = True,
    include_spatial: bool = True,
) -> xr.Dataset:
    """Assemble a per-lead feature+target Dataset.

    Variables keep their natural dimensions:
      - gridded predictors: (time, lat, lon)
      - climate indices: (time,)
      - seasonal encoding: (time,)
      - spatial encoding: (lat, lon)
      - target: (time, lat, lon)
      - winter_mask: (time,) — included as a coord on `time` for convenience

    Parameters
    ----------
    datasets
        Dict returned by `droughtmodel.data.load_all`.
    lead
        Lead time in months (1, 3, or 6).
    contemporary
        List of canonical variable names to include as time-t features.
    lags
        Mapping {var_name: [lag_1, lag_2, ...]} of lagged features to include.
    include_seasonal, include_spatial
        Toggle seasonal (sin/cos month) and spatial (lat, lon) encodings.
    """
    lags = lags or {}
    feature_vars: dict[str, xr.DataArray] = {}

    spei3 = gather_predictor("spei3", datasets)
    time_coord = spei3["time"]
    lat = spei3["lat"]
    lon = spei3["lon"]

    for name in contemporary:
        feature_vars[name] = gather_predictor(name, datasets)

    for name, lag_list in lags.items():
        da = gather_predictor(name, datasets)
        feature_vars.update(build_lag_features(da, lag_list))

    if include_seasonal:
        feature_vars.update(add_seasonal_encoding(time_coord))

    if include_spatial:
        feature_vars.update(add_spatial_encoding(lat, lon))

    feature_vars["target"] = build_target(spei3, lead)

    ds = xr.Dataset(feature_vars)
    ds = ds.assign_coords(winter_mask=("time", apply_winter_mask(time_coord).values))
    ds.attrs["lead"] = lead
    ds.attrs["contemporary"] = list(contemporary)
    ds.attrs["lags"] = {k: list(v) for k, v in lags.items()}
    return ds


def cache_dataset_zarr(ds: xr.Dataset, path: str | Path) -> Path:
    """Write the assembled dataset to zarr with sensible chunking."""
    path = Path(path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Chunk along time so per-fold slices load efficiently.
    chunks = {}
    if "time" in ds.dims:
        chunks["time"] = min(120, ds.sizes["time"])
    if "lat" in ds.dims:
        chunks["lat"] = ds.sizes["lat"]
    if "lon" in ds.dims:
        chunks["lon"] = ds.sizes["lon"]
    ds = ds.chunk(chunks)
    # Drop the JSON-incompatible attrs before writing
    safe_attrs = {k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float, bool, list))}
    ds.attrs = safe_attrs
    ds.to_zarr(path, mode="w", consolidated=True)
    return path


def load_cached_dataset(path: str | Path) -> xr.Dataset:
    """Load a previously cached zarr feature dataset."""
    path = Path(path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return xr.open_zarr(path, consolidated=True)
