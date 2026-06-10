"""Model registry — central name → class lookup."""

from __future__ import annotations

from typing import Type

from droughtmodel.models.base import BaseModel
from droughtmodel.models.baselines import (
    ARBaseline,
    ClimatologyBaseline,
    PersistenceBaseline,
)


REGISTRY: dict[str, Type[BaseModel]] = {
    "climatology": ClimatologyBaseline,
    "persistence": PersistenceBaseline,
    "ar": ARBaseline,
    # Linear / RF / XGBoost added in Phases 7–8
}


def get_model(name: str, **kwargs) -> BaseModel:
    """Look up a model class by name and instantiate it with kwargs."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
