"""Data loading and preprocessing for the SPEI3 forecasting study.

Loads CRU, ERA5, climate indices, and SPEI3 NetCDFs from `inputs/`, renames
variables to canonical names, aligns coordinates, handles missing values, and
provides stationarity diagnostics for SPEI3.

All gridded datasets are returned with:
  - `lat` ascending
  - `time` monthly, between `time_period.start` and `time_period.end`
  - canonical variable names (see configs/data.yaml)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Project root is the parent of the `droughtmodel/` package directory.
# Used to anchor relative paths from YAML configs so loaders work regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(p: str | Path) -> Path:
    """Resolve a path against PROJECT_ROOT if it's relative; return absolute Path."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_config(path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    """Load the data-layer YAML config and resolve all paths against the project root.

    The config can be loaded from any CWD; relative paths inside it (under `paths:`)
    are rewritten to absolute paths anchored at the project root so that downstream
    loaders (e.g., xarray.open_dataset) succeed regardless of where Python is run.
    """
    config_path = _resolve_path(path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "paths" in config:
        config["paths"] = {k: str(_resolve_path(v)) for k, v in config["paths"].items()}

    return config


def _rename_and_sort(ds: xr.Dataset, renames: dict[str, str], sort_lat_ascending: bool) -> xr.Dataset:
    """Apply variable renames and (optionally) sort `lat` ascending."""
    ds = ds.rename({k: v for k, v in renames.items() if k in ds.variables})
    if sort_lat_ascending and "lat" in ds.coords and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds


def load_cru(path: str | Path, renames: dict[str, str], sort_lat_ascending: bool = True) -> xr.Dataset:
    """Load CRU dataset (precip, Tmin, Tmax, PET)."""
    ds = xr.open_dataset(path)
    return _rename_and_sort(ds, renames, sort_lat_ascending)


def load_era5(path: str | Path, renames: dict[str, str], sort_lat_ascending: bool = True) -> xr.Dataset:
    """Load ERA5 dataset (solar, wind, VPD, TCWV, RZSM)."""
    ds = xr.open_dataset(path)
    return _rename_and_sort(ds, renames, sort_lat_ascending)


def load_climate_indices(path: str | Path, renames: dict[str, str]) -> xr.Dataset:
    """Load large-scale climate indices (ENSO, NAO, MO) as a time-only xarray Dataset."""
    ds = xr.open_dataset(path)
    return ds.rename({k: v for k, v in renames.items() if k in ds.variables})


def load_spei3(path: str | Path, renames: dict[str, str], sort_lat_ascending: bool = True) -> xr.DataArray:
    """Load the SPEI3 target."""
    ds = xr.open_dataset(path)
    ds = _rename_and_sort(ds, renames, sort_lat_ascending)
    return ds["spei3"]


def align_time_period(
    datasets: dict[str, xr.Dataset | xr.DataArray],
    start: str = "1950-01",
    end: str = "2024-12",
) -> dict[str, xr.Dataset | xr.DataArray]:
    """Clip every dataset to the study time period."""
    return {k: ds.sel(time=slice(start, end)) for k, ds in datasets.items()}


def handle_missing_values(
    ds: xr.Dataset | xr.DataArray,
    strategy: str = "interpolate",
    max_gap_months: int = 2,
) -> xr.Dataset | xr.DataArray:
    """Handle missing values along the time dimension.

    Strategies:
        - 'interpolate': linear interpolation along time, up to `max_gap_months` consecutive NaNs
        - 'ffill': forward-fill
        - 'drop': drop time steps where any value is NaN (use cautiously on gridded data)
        - 'none': return unchanged
    """
    if strategy == "none":
        return ds
    if strategy == "interpolate":
        return ds.interpolate_na(dim="time", method="linear", max_gap=pd.Timedelta(days=31 * max_gap_months))
    if strategy == "ffill":
        return ds.ffill(dim="time")
    if strategy == "drop":
        return ds.dropna(dim="time", how="any")
    raise ValueError(f"Unknown missing-value strategy: {strategy}")


def cell_missing_fraction(da: xr.DataArray) -> xr.DataArray:
    """Return per-cell fraction of NaN values along time."""
    return da.isnull().mean(dim="time")


def filter_cells_by_missingness(
    da: xr.DataArray, threshold: float = 0.05
) -> xr.DataArray:
    """Mask cells exceeding the per-cell missing-value fraction threshold.

    Returns the same DataArray with cells set to NaN if their missing fraction exceeds the threshold.
    Used per-fold on the training data.
    """
    mask = cell_missing_fraction(da) <= threshold
    return da.where(mask)


def load_all(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Orchestrate loading and alignment of all inputs.

    Returns a dict with keys: cru, era5, indices, spei3 — each clipped to the
    study time period and with canonical variable names.
    """
    if config is None:
        config = load_config()

    paths = config["paths"]
    renames = config["variable_renames"]
    sort_asc = config.get("sort_lat_ascending", True)

    raw = {
        "cru": load_cru(paths["cru"], renames["cru"], sort_asc),
        "era5": load_era5(paths["era5"], renames["era5"], sort_asc),
        "indices": load_climate_indices(paths["climate_indices"], renames["climate_indices"]),
        "spei3": load_spei3(paths["spei3"], renames["spei3"], sort_asc).to_dataset(name="spei3"),
    }

    tp = config["time_period"]
    aligned = align_time_period(raw, start=tp["start"], end=tp["end"])

    mv = config.get("missing_values", {})
    strategy = mv.get("strategy", "interpolate")
    max_gap = mv.get("max_gap_months", 2)
    if strategy != "none":
        for k, ds in aligned.items():
            aligned[k] = handle_missing_values(ds, strategy=strategy, max_gap_months=max_gap)

    return aligned


# ---------------------------------------------------------------------------
# Stationarity diagnostics (SPEI3-focused, called from the EDA notebook)
# ---------------------------------------------------------------------------

def mann_kendall_pvalue(series: np.ndarray) -> float:
    """Two-sided Mann-Kendall trend test on a 1-D series; returns p-value.

    Uses scipy's Kendall tau on (rank, time). NaN-safe via mask.
    """
    from scipy.stats import kendalltau

    mask = ~np.isnan(series)
    if mask.sum() < 10:
        return np.nan
    t = np.arange(len(series))[mask]
    y = series[mask]
    _, p = kendalltau(t, y, nan_policy="omit")
    return float(p)


def mann_kendall_pvalue_map(spei3: xr.DataArray) -> xr.DataArray:
    """Apply Mann-Kendall to annual-mean SPEI3 per cell. Returns a 2-D p-value map."""
    annual = spei3.resample(time="YE").mean()
    pmap = xr.apply_ufunc(
        mann_kendall_pvalue,
        annual,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    pmap.name = "mk_pvalue"
    pmap.attrs["long_name"] = "Mann-Kendall trend test p-value (annual SPEI3)"
    return pmap


def ks_test_pvalue(series: np.ndarray, split_idx: int) -> float:
    """Two-sample Kolmogorov-Smirnov p-value comparing pre- vs post-split distributions."""
    from scipy.stats import ks_2samp

    pre = series[:split_idx]
    post = series[split_idx:]
    pre = pre[~np.isnan(pre)]
    post = post[~np.isnan(post)]
    if len(pre) < 30 or len(post) < 30:
        return np.nan
    _, p = ks_2samp(pre, post)
    return float(p)


def ks_pvalue_map(spei3: xr.DataArray, split_year: int = 1990) -> xr.DataArray:
    """Apply KS test per cell comparing distributions before and after `split_year`."""
    times = pd.DatetimeIndex(spei3["time"].values)
    split_idx = int((times.year < split_year).sum())

    pmap = xr.apply_ufunc(
        lambda s: ks_test_pvalue(s, split_idx),
        spei3,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    pmap.name = "ks_pvalue"
    pmap.attrs["long_name"] = f"KS-test p-value (pre-{split_year} vs post-{split_year} SPEI3)"
    return pmap
