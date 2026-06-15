"""Model registry — central name → class lookup."""

from __future__ import annotations

from typing import Type

from droughtmodel.models.base import BaseModel
from droughtmodel.models.baselines import (
    ARBaseline,
    ClimatologyBaseline,
    PersistenceBaseline,
)
from droughtmodel.models.linear import (
    ElasticNetModel,
    LassoModel,
    LinearRegressionModel,
    RidgeModel,
)
from droughtmodel.models.tree import RandomForestModel, XGBoostModel


REGISTRY: dict[str, Type[BaseModel]] = {
    "climatology": ClimatologyBaseline,
    "persistence": PersistenceBaseline,
    "ar": ARBaseline,
    "ols": LinearRegressionModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "elasticnet": ElasticNetModel,
    "rf": RandomForestModel,
    "xgboost": XGBoostModel,
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Look up a model class by name and instantiate it with kwargs."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
