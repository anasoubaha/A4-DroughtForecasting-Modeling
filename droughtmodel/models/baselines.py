"""Baselines: Climatology, Persistence, AR(p). See v3 §7.0.

All three implement `BaseModel`. They expect a feature dataset built by
`droughtmodel.features.build_dataset` (containing `spei3`, optional `spei3_lag*`,
and `target`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import Ridge

from droughtmodel.models.base import BaseModel


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------

class ClimatologyBaseline(BaseModel):
    """Per-cell per-calendar-month mean of the training target.

    Forecast at issue time *t* for lead *L*: the historical mean of SPEI3
    at cell (lat, lon) for calendar month ``(t + L).month``, computed over the
    training period.
    """

    name = "climatology"

    def __init__(self):
        self.monthly_means: xr.DataArray | None = None

    @staticmethod
    def _target_months(time_coord, lead: int) -> np.ndarray:
        target_times = pd.DatetimeIndex(time_coord.values if hasattr(time_coord, "values") else time_coord)
        target_times = target_times + pd.DateOffset(months=lead)
        return target_times.month.values

    def fit(self, train: xr.Dataset, val: xr.Dataset | None = None) -> "ClimatologyBaseline":
        if "target" not in train.data_vars:
            raise ValueError("Training dataset must contain a 'target' variable.")
        lead = int(train.attrs.get("lead", 0))
        target = train["target"]
        target_months = self._target_months(train["time"], lead)
        target = target.assign_coords(target_month=("time", target_months))
        self.monthly_means = target.groupby("target_month").mean(dim="time")
        return self

    def predict(self, dataset: xr.Dataset) -> xr.DataArray:
        if self.monthly_means is None:
            raise RuntimeError("ClimatologyBaseline must be .fit() before .predict().")
        lead = int(dataset.attrs.get("lead", 0))
        target_months = self._target_months(dataset["time"], lead)
        sel = xr.DataArray(target_months, dims="time", coords={"time": dataset["time"]})
        pred = self.monthly_means.sel(target_month=sel)
        pred.name = "pred_climatology"
        return pred


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class PersistenceBaseline(BaseModel):
    """Persistence forecast: SPEI3(t+L) = SPEI3(t).

    Returns the contemporary SPEI3 value as the prediction. Since both `spei3`
    and `target` are in the standardization exception list (both already in
    z-score units), this is directly comparable to the target.
    """

    name = "persistence"

    def __init__(self, spei3_column: str = "spei3"):
        self.spei3_column = spei3_column

    def fit(self, train: xr.Dataset, val: xr.Dataset | None = None) -> "PersistenceBaseline":
        if self.spei3_column not in train.data_vars:
            raise ValueError(f"'{self.spei3_column}' not found in training dataset.")
        return self  # nothing to learn

    def predict(self, dataset: xr.Dataset) -> xr.DataArray:
        pred = dataset[self.spei3_column].copy()
        pred.name = "pred_persistence"
        return pred


# ---------------------------------------------------------------------------
# AR(p)
# ---------------------------------------------------------------------------

class ARBaseline(BaseModel):
    """AR(p) — Ridge regression on SPEI3 lags only.

    Uses ``spei3`` (contemporary, lag 0) plus ``spei3_lag1, ..., spei3_lag{p-1}``
    if available in the dataset. If fewer than `p` lag features exist (because
    PACF/CCF didn't select all of them), only the available ones are used and a
    warning is logged via the `feature_names_` attribute.

    Pooled across all cells of the input dataset (typically Morocco cells only
    after the pipeline applies the region mask). One set of Ridge coefficients
    is fit on the stacked (time × cell) sample.

    Parameters
    ----------
    p
        Order of the AR model. AR(1) uses only the contemporary SPEI3.
        AR(3) uses spei3 + spei3_lag1 + spei3_lag2.
    alpha
        Ridge regularization strength.
    """

    name = "ar"

    def __init__(self, p: int = 3, alpha: float = 1.0):
        if p < 1:
            raise ValueError(f"AR order p must be >= 1; got {p}")
        self.p = int(p)
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.feature_names_: list[str] | None = None

    def _feature_columns(self, dataset: xr.Dataset) -> list[str]:
        cols = []
        if "spei3" in dataset.data_vars:
            cols.append("spei3")
        for k in range(1, self.p):
            col = f"spei3_lag{k}"
            if col in dataset.data_vars:
                cols.append(col)
        return cols

    @staticmethod
    def _stack(dataset: xr.Dataset, cols: list[str]) -> np.ndarray:
        """Stack features along the last axis: (..., n_features)."""
        return np.stack([dataset[c].values for c in cols], axis=-1)

    def fit(self, train: xr.Dataset, val: xr.Dataset | None = None) -> "ARBaseline":
        if "target" not in train.data_vars:
            raise ValueError("Training dataset must contain a 'target' variable.")
        cols = self._feature_columns(train)
        if not cols:
            raise ValueError(
                f"AR({self.p}) requires `spei3` and/or `spei3_lag*` features in the dataset."
            )
        X = self._stack(train, cols).reshape(-1, len(cols))
        y = train["target"].values.ravel()
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[mask], y[mask]
        if X.shape[0] == 0:
            raise ValueError("No finite training samples after dropping NaN rows.")

        model = Ridge(alpha=self.alpha, fit_intercept=True).fit(X, y)
        self.coef_ = model.coef_
        self.intercept_ = float(model.intercept_)
        self.feature_names_ = cols
        return self

    def predict(self, dataset: xr.Dataset) -> xr.DataArray:
        if self.coef_ is None or self.feature_names_ is None:
            raise RuntimeError("ARBaseline must be .fit() before .predict().")
        X = self._stack(dataset, self.feature_names_)  # (time, lat, lon, n_features)
        shape = X.shape[:-1]
        X_flat = X.reshape(-1, X.shape[-1])
        y_flat = X_flat @ self.coef_ + self.intercept_
        y = y_flat.reshape(shape)

        template = dataset["target"] if "target" in dataset.data_vars else dataset[self.feature_names_[0]]
        pred = xr.DataArray(y, dims=template.dims, coords=template.coords, name="pred_ar")
        return pred

    def feature_importance(self) -> dict[str, float] | None:
        if self.coef_ is None or self.feature_names_ is None:
            return None
        return {name: float(c) for name, c in zip(self.feature_names_, self.coef_)}
