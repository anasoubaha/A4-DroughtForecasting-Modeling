"""Shared infrastructure for tabular models (linear + tree).

Internal module — exposes helpers for stacking `(time, lat, lon)` xarray
Datasets into `(n_samples, n_features)` numpy matrices, and a
`TabularBaseModel` that handles fit/predict/feature_importance for any
sklearn-style estimator that takes `(X, y)` and produces `.predict(X)`.

Subclasses in `linear.py` and `tree.py` configure the concrete estimator
and (optionally) override `_fit_estimator` for early-stopping behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from droughtmodel.models.base import BaseModel


def _feature_columns(dataset: xr.Dataset) -> list[str]:
    """All data_vars except the target, sorted for deterministic ordering."""
    return sorted(v for v in dataset.data_vars if v != "target")


def _stack_xy(
    dataset: xr.Dataset, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray | None, xr.DataArray]:
    """Stack `(time, lat, lon, n_features) → (n_samples, n_features)`.

    Features may have heterogeneous dimensions:
      - 3-D gridded predictors with dims ``(time, lat, lon)``,
      - 1-D climate indices with dims ``(time,)`` (NAO, ENSO, MO),
      - 1-D seasonal encodings ``sin_m``, ``cos_m``,
      - 2-D spatial encodings ``(lat, lon)``.

    Each feature is broadcast to the template's full ``(time, lat, lon)`` shape
    before stacking — 1-D climate indices are tiled across the spatial axes
    (the index value is identical for every Morocco cell at a given month),
    spatial encodings are tiled across the time axis.

    Returns
    -------
    X        : ``(n_samples, n_features)`` ndarray
    y        : ``(n_samples,)`` ndarray, or ``None`` if `target` is absent
    template : ``DataArray`` of shape ``(time, lat, lon)`` used for reshaping
               predictions back to the input grid.
    """
    # The template must be a 3-D DataArray so we have an authoritative shape
    # to broadcast against. Prefer the target; fall back to the first feature
    # that has the full (time, lat, lon) dims.
    if "target" in dataset.data_vars:
        template = dataset["target"]
    else:
        full_dims = ("time", "lat", "lon")
        candidates = [n for n in feature_names if set(full_dims).issubset(dataset[n].dims)]
        if not candidates:
            raise ValueError(
                "_stack_xy needs at least one feature with full (time, lat, lon) dims "
                "to define the template; got only lower-dimensional features."
            )
        template = dataset[candidates[0]]

    arrays = [dataset[c].broadcast_like(template).transpose(*template.dims).values
              for c in feature_names]
    X = np.stack(arrays, axis=-1)
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = dataset["target"].values.ravel() if "target" in dataset.data_vars else None
    return X_flat, y_flat, template


class TabularBaseModel(BaseModel):
    """Shared base for sklearn-style models on `(time, lat, lon)` datasets.

    Subclasses pass a fitted-or-not sklearn estimator to ``__init__`` and
    inherit the xarray <-> numpy plumbing. Override ``_fit_estimator`` to
    customise the fit call (e.g.\\ XGBoost early-stopping with ``eval_set``).
    """

    name = "tabular"

    def __init__(self, estimator: Any):
        self.estimator: Any = estimator
        self.feature_names_: list[str] | None = None

    # ----- override point for early-stopping models -----
    def _fit_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        self.estimator.fit(X, y)

    # ----- public API (BaseModel) -----
    def fit(self, train: xr.Dataset, val: xr.Dataset | None = None) -> "TabularBaseModel":
        if "target" not in train.data_vars:
            raise ValueError(f"{self.name}: training dataset must contain a 'target' variable.")
        feature_names = _feature_columns(train)
        if not feature_names:
            raise ValueError(f"{self.name}: no feature columns found in dataset.")

        X, y, _ = _stack_xy(train, feature_names)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if not mask.any():
            raise ValueError(f"{self.name}: no finite training samples after NaN filtering.")
        X, y = X[mask], y[mask]

        X_val: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if val is not None:
            missing = [c for c in feature_names if c not in val.data_vars]
            if missing:
                raise ValueError(f"{self.name}: features missing from val dataset: {missing}")
            Xv, yv, _ = _stack_xy(val, feature_names)
            if yv is not None:
                mv = np.isfinite(yv) & np.all(np.isfinite(Xv), axis=1)
                if mv.any():
                    X_val, y_val = Xv[mv], yv[mv]

        self._fit_estimator(X, y, X_val, y_val)
        self.feature_names_ = feature_names
        return self

    def predict(self, dataset: xr.Dataset) -> xr.DataArray:
        if self.feature_names_ is None:
            raise RuntimeError(f"{self.name} must be .fit() before .predict().")
        missing = [c for c in self.feature_names_ if c not in dataset.data_vars]
        if missing:
            raise ValueError(f"{self.name}: features missing from prediction dataset: {missing}")
        X, _, template = _stack_xy(dataset, self.feature_names_)
        finite = np.all(np.isfinite(X), axis=1)
        y_full = np.full(X.shape[0], np.nan, dtype=float)
        if finite.any():
            y_full[finite] = self.estimator.predict(X[finite])
        return xr.DataArray(
            y_full.reshape(template.shape),
            dims=template.dims,
            coords=template.coords,
            name=f"pred_{self.name}",
        )

    def feature_importance(self) -> dict[str, float] | None:
        """Returns ``coef_`` for linear models, ``feature_importances_`` for trees."""
        if self.feature_names_ is None:
            return None
        if hasattr(self.estimator, "coef_"):
            arr = np.asarray(self.estimator.coef_).ravel()
        elif hasattr(self.estimator, "feature_importances_"):
            arr = np.asarray(self.estimator.feature_importances_).ravel()
        else:
            return None
        return {n: float(c) for n, c in zip(self.feature_names_, arr)}
