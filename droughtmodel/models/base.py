"""Abstract base class for all forecasters (baselines and ML models)."""

from __future__ import annotations

import abc

import xarray as xr


class BaseModel(abc.ABC):
    """Common interface for baselines (climatology, persistence, AR) and ML models.

    Models receive an `xr.Dataset` containing time-indexed features and a `target`
    variable with dims (time, lat, lon). They return an `xr.DataArray` of predictions
    with the same dims as `target`.

    The pipeline (Phase 10) restricts the dataset to Morocco cells before calling
    fit/predict, so models don't need to know about the region mask themselves.
    """

    #: Short identifier used in registry, results tables, and config filenames.
    name: str = "abstract"

    @abc.abstractmethod
    def fit(self, train: xr.Dataset, val: xr.Dataset | None = None) -> "BaseModel":
        """Fit on training data; `val` is optional for HP tuning / early stopping."""
        ...

    @abc.abstractmethod
    def predict(self, dataset: xr.Dataset) -> xr.DataArray:
        """Predict targets for the (time, lat, lon) grid in `dataset`."""
        ...

    def feature_importance(self) -> dict[str, float] | None:
        """Optional: per-feature importance. Defaults to None for non-interpretable models."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
