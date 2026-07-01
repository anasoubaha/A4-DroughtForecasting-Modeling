"""Sliding-window reshaping for the LSTM pipeline (Phase 12).

This module sits between the existing 2-D-tabular pipeline (`droughtmodel.models._tabular._stack_xy`)
and the PyTorch LSTM (`droughtmodel.models.lstm`).

Responsibilities:

  1. `lstm_boundary_gap(lead, sequence_length)`
     Returns the boundary gap that must be enforced between train / val / test
     for the LSTM. Per the Phase 12 spec: `gap = L + T + 1`. Larger than the
     tabular `L + K_eff + 2` for typical T ≥ 6 because the LSTM looks back T
     steps from feature-time t, so train → test contamination requires
     `test_start − train_end > T + L`.

  2. `build_sequences(ds, sequence_length, end_indices=None) → (X, y, meta)`
     Turns a standardized `(time, lat, lon)` xarray Dataset into 3-D float32
     numpy arrays of shape `(n_samples, T, n_features)` and labels of shape
     `(n_samples,)`. One sample per (lat, lon, t_feature) cell where
     `t_feature ∈ end_indices` (defaults to all valid t with full T-window
     available within the dataset).

     - The same `FoldStandardizer` is applied to the 2-D dataset BEFORE this
       function is called. We just rearrange standardized values into windows.
     - Static features (`lat_feat`, `lon_feat`, and any other (lat, lon)-only
       or no-time variables) are broadcast across all T timesteps within a
       sample's sequence (so the model sees lat/lon at every step).
     - Time-only series (sin_month, cos_month, ENSO, NAO, MO) are tiled across
       cells.
     - Samples that contain any NaN in X or y are filtered out.

     `meta` is a dict with the alignment info needed to scatter scalar
     predictions back into a (time, lat, lon) grid: `sample_t_idx`,
     `sample_lat_idx`, `sample_lon_idx`, `feature_names`.

  3. `predict_to_grid(preds_flat, meta, target_template)`
     Scatters scalar predictions back to the template's `(time, lat, lon)`
     grid (NaN where no sample was emitted for a given cell-time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import xarray as xr


__all__ = [
    "lstm_boundary_gap",
    "build_sequences",
    "predict_to_grid",
    "SequenceMeta",
]


# ---------------------------------------------------------------------------
# Boundary-gap formula
# ---------------------------------------------------------------------------

def lstm_boundary_gap(lead: int, sequence_length: int) -> int:
    """Boundary gap for the LSTM at the train→val / val→test seam.

    Each sample at feature-time t consumes features from [t-T+1, t] and predicts
    target SPEI3(t+L). For the test sample whose features START at time `t_start`,
    the inputs reach back to `t_start - T + 1`. To prevent any train target
    SPEI3(train_end + L) from appearing in the test inputs we need

        test_start - T + 1 > train_end + L   ⇒   gap = test_start - train_end ≥ T + L

    The +1 in the returned gap is a one-month safety margin (matches the +2
    margin used by the tabular `L + K_eff + 2` formula scaled to one month).
    """
    return int(lead) + int(sequence_length) + 1


# ---------------------------------------------------------------------------
# Reshape helpers
# ---------------------------------------------------------------------------

@dataclass
class SequenceMeta:
    """Provenance of the rows in a (n_samples, T, n_features) tensor."""
    feature_names: list[str]
    sample_t_idx: np.ndarray       # (n_samples,) — feature-time index in the dataset
    sample_lat_idx: np.ndarray     # (n_samples,)
    sample_lon_idx: np.ndarray     # (n_samples,)
    sequence_length: int
    template_shape: tuple[int, int, int]   # (n_time, n_lat, n_lon)


def _feature_columns(ds: xr.Dataset) -> list[str]:
    """All data_vars except `target`, sorted for deterministic ordering."""
    return sorted(v for v in ds.data_vars if v != "target")


def _broadcast_to_template(ds: xr.Dataset, feature_names: list[str]) -> np.ndarray:
    """Stack each feature broadcast to the dataset's `(time, lat, lon)` shape.

    Mirrors the broadcast logic in `droughtmodel.models._tabular._stack_xy`:
      - 3-D (time, lat, lon) vars are used as-is
      - 1-D (time,) climate indices and seasonal encodings are tiled across cells
      - 2-D (lat, lon) spatial encodings (lat_feat / lon_feat) are tiled across time

    Returns a (n_features, n_time, n_lat, n_lon) float32 array — features-leading
    so per-cell slicing is contiguous in memory along the time axis.
    """
    if "target" in ds.data_vars:
        template = ds["target"]
    else:
        full_dims = ("time", "lat", "lon")
        candidates = [n for n in feature_names if set(full_dims).issubset(ds[n].dims)]
        if not candidates:
            raise ValueError(
                "build_sequences needs at least one feature with full (time, lat, lon) "
                "dims (or a `target` variable) to define the template."
            )
        template = ds[candidates[0]]

    arrays = []
    for name in feature_names:
        da = ds[name].broadcast_like(template).transpose(*template.dims)
        arrays.append(np.asarray(da.values, dtype=np.float32))
    return np.stack(arrays, axis=0)   # (F, T, H, W)


def build_sequences(
    ds: xr.Dataset,
    sequence_length: int,
    *,
    end_indices: Iterable[int] | None = None,
    cell_mask: xr.DataArray | None = None,
    target_filter: np.ndarray | None = None,
    drop_nan_samples: bool = True,
) -> tuple[np.ndarray, np.ndarray, SequenceMeta]:
    """Build `(n_samples, T, n_features)` tensors from a standardized 2-D dataset.

    Parameters
    ----------
    ds
        xarray Dataset with dims `(time, lat, lon)` on `target` and on at least one
        feature. Features may also be 1-D or 2-D (broadcast automatically).
    sequence_length
        T — number of months of history per sample. The first valid feature-time
        index in the dataset is `T - 1` (zero-indexed); earlier indices are skipped.
    end_indices
        Optional iterable of feature-time indices (into `ds.time`) to use as the
        END of each sample's window. Defaults to all valid indices `[T-1, n_time)`.
    cell_mask
        Optional 2-D boolean DataArray with dims `(lat, lon)`. Only cells where
        the mask is True become sample centers (matches the Morocco-only modeling
        scope).
    target_filter
        Optional boolean array of length `len(end_indices)` (or `n_time` if
        `end_indices` is None). When True at index i, samples whose feature-time
        is `end_indices[i]` are KEPT; False rows are dropped before the NaN
        filter. Used for the v4 winter-only training filter on TRAIN/VAL/REFIT
        slices — pass None to keep all months (e.g. for TEST).
    drop_nan_samples
        If True (default), drops samples whose `X` window or `y` contains any
        NaN. The downstream LSTM cannot ingest NaNs.

    Returns
    -------
    X : float32 `(n_samples, sequence_length, n_features)` ndarray
    y : float32 `(n_samples,)` ndarray
    meta : SequenceMeta
        Alignment info for `predict_to_grid`.
    """
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be ≥ 1, got {sequence_length}")
    if "target" not in ds.data_vars:
        raise ValueError("build_sequences needs a `target` variable in `ds`.")

    feature_names = _feature_columns(ds)
    if not feature_names:
        raise ValueError("build_sequences: no feature columns in dataset.")

    feat_cube = _broadcast_to_template(ds, feature_names)             # (F, T, H, W)
    target_arr = np.asarray(ds["target"].values, dtype=np.float32)    # (T, H, W)
    n_time, n_lat, n_lon = target_arr.shape
    if feat_cube.shape[1:] != target_arr.shape:
        raise ValueError(
            f"feature cube shape {feat_cube.shape[1:]} does not match target shape "
            f"{target_arr.shape} (after broadcast)."
        )

    if end_indices is None:
        end_indices = np.arange(sequence_length - 1, n_time, dtype=np.int64)
    else:
        end_indices = np.asarray(list(end_indices), dtype=np.int64)
        if (end_indices < sequence_length - 1).any():
            raise ValueError(
                f"All end_indices must be ≥ sequence_length-1 ({sequence_length-1}); "
                f"got minimum {int(end_indices.min())}."
            )
        if (end_indices >= n_time).any():
            raise ValueError(
                f"All end_indices must be < n_time ({n_time}); got maximum "
                f"{int(end_indices.max())}."
            )

    # Apply the winter-target filter on END_INDICES if requested.
    if target_filter is not None:
        target_filter = np.asarray(target_filter, dtype=bool)
        if target_filter.shape != end_indices.shape:
            raise ValueError(
                f"target_filter shape {target_filter.shape} must match "
                f"end_indices shape {end_indices.shape}."
            )
        end_indices = end_indices[target_filter]

    # Which cells become sample centers?
    if cell_mask is not None:
        mask2d = np.asarray(cell_mask.values, dtype=bool)
        if mask2d.shape != (n_lat, n_lon):
            raise ValueError(
                f"cell_mask shape {mask2d.shape} must match (lat, lon) = "
                f"({n_lat}, {n_lon})."
            )
        cell_ii, cell_jj = np.where(mask2d)
    else:
        cell_ii, cell_jj = np.meshgrid(
            np.arange(n_lat), np.arange(n_lon), indexing="ij"
        )
        cell_ii = cell_ii.ravel()
        cell_jj = cell_jj.ravel()

    n_per_time = len(cell_ii)
    n_times_sel = len(end_indices)
    n_samples_max = n_per_time * n_times_sel

    if n_samples_max == 0:
        empty_X = np.zeros((0, sequence_length, len(feature_names)), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.float32)
        meta = SequenceMeta(
            feature_names=feature_names,
            sample_t_idx=np.zeros((0,), dtype=np.int64),
            sample_lat_idx=np.zeros((0,), dtype=np.int64),
            sample_lon_idx=np.zeros((0,), dtype=np.int64),
            sequence_length=sequence_length,
            template_shape=(n_time, n_lat, n_lon),
        )
        return empty_X, empty_y, meta

    F = len(feature_names)
    X = np.empty((n_samples_max, sequence_length, F), dtype=np.float32)
    y = np.empty((n_samples_max,), dtype=np.float32)
    s_t = np.empty((n_samples_max,), dtype=np.int64)
    s_i = np.empty((n_samples_max,), dtype=np.int64)
    s_j = np.empty((n_samples_max,), dtype=np.int64)

    k = 0
    for t_end in end_indices:
        t0 = int(t_end) - sequence_length + 1
        # window slice over the time axis is contiguous; lat/lon indexed by mask
        window = feat_cube[:, t0:t_end + 1, :, :]                                # (F, T, H, W)
        # Reorder to (cells, T, F): pick (i, j) from H,W and move axes
        window_cells = window[:, :, cell_ii, cell_jj]                            # (F, T, n_cells)
        window_cells = np.transpose(window_cells, (2, 1, 0))                     # (n_cells, T, F)
        y_cells = target_arr[t_end, cell_ii, cell_jj]                            # (n_cells,)

        X[k:k + n_per_time] = window_cells
        y[k:k + n_per_time] = y_cells
        s_t[k:k + n_per_time] = t_end
        s_i[k:k + n_per_time] = cell_ii
        s_j[k:k + n_per_time] = cell_jj
        k += n_per_time

    if drop_nan_samples:
        finite_X = np.isfinite(X).reshape(X.shape[0], -1).all(axis=1)
        finite_y = np.isfinite(y)
        keep = finite_X & finite_y
        X = X[keep]
        y = y[keep]
        s_t = s_t[keep]
        s_i = s_i[keep]
        s_j = s_j[keep]

    meta = SequenceMeta(
        feature_names=feature_names,
        sample_t_idx=s_t,
        sample_lat_idx=s_i,
        sample_lon_idx=s_j,
        sequence_length=sequence_length,
        template_shape=(n_time, n_lat, n_lon),
    )
    return X, y, meta


def predict_to_grid(
    preds_flat: np.ndarray,
    meta: SequenceMeta,
    target_template: xr.DataArray,
) -> xr.DataArray:
    """Scatter scalar predictions back into a `(time, lat, lon)` DataArray.

    Cells / time-points without a corresponding sample (e.g. dropped by the NaN
    filter, or off-mask) receive NaN. The output DataArray inherits the
    template's coords and dims.

    Parameters
    ----------
    preds_flat
        1-D float array of length `len(meta.sample_t_idx)` — one prediction per
        sample produced by `build_sequences`.
    meta
        The `SequenceMeta` returned by `build_sequences`.
    target_template
        A `(time, lat, lon)` DataArray to inherit coords/dims from. Typically
        the test-slice `target`. The TIME COORDS of the scattered output match
        this template's time coords on a position-by-position basis — the
        caller is responsible for using a template that matches the dataset
        that `build_sequences` was called on.
    """
    preds_flat = np.asarray(preds_flat, dtype=np.float32).ravel()
    if preds_flat.shape != meta.sample_t_idx.shape:
        raise ValueError(
            f"preds_flat length {len(preds_flat)} does not match the number of "
            f"samples in meta ({len(meta.sample_t_idx)})."
        )

    out = np.full(meta.template_shape, np.nan, dtype=np.float32)
    out[meta.sample_t_idx, meta.sample_lat_idx, meta.sample_lon_idx] = preds_flat
    return xr.DataArray(
        out,
        dims=target_template.dims,
        coords=target_template.coords,
        name="pred_lstm",
    )