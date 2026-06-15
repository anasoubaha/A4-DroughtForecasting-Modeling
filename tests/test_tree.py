"""Unit tests for tree-based models (v3 §7.1)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from droughtmodel.models.registry import REGISTRY, get_model
from droughtmodel.models.tree import RandomForestModel, TreeBaseModel, XGBoostModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(
    n_time: int = 240, n_lat: int = 4, n_lon: int = 4, lead: int = 3,
    noise: float = 0.3, seed: int = 0,
) -> xr.Dataset:
    """target = 1.5·spei3 + 0.5·precip + 0·noise_var + ε (contemporaneous)."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("1950-01", periods=n_time, freq="MS")
    lats = np.linspace(28, 36, n_lat)
    lons = np.linspace(-13, -1, n_lon)
    coords = {"time": times, "lat": lats, "lon": lons}
    dims = ("time", "lat", "lon")

    spei3 = rng.standard_normal((n_time, n_lat, n_lon))
    precip = rng.standard_normal((n_time, n_lat, n_lon))
    noise_var = rng.standard_normal((n_time, n_lat, n_lon))

    target = (
        1.5 * spei3 + 0.5 * precip
        + noise * rng.standard_normal((n_time, n_lat, n_lon))
    )
    target[-lead:] = np.nan

    ds = xr.Dataset({
        "spei3": xr.DataArray(spei3, dims=dims, coords=coords),
        "precip": xr.DataArray(precip, dims=dims, coords=coords),
        "noise_var": xr.DataArray(noise_var, dims=dims, coords=coords),
        "target": xr.DataArray(target, dims=dims, coords=coords),
    })
    ds.attrs["lead"] = lead
    return ds


def _split_train_val(ds: xr.Dataset, val_frac: float = 0.25) -> tuple[xr.Dataset, xr.Dataset]:
    n = ds.sizes["time"]
    n_val = int(n * val_frac)
    return ds.isel(time=slice(None, n - n_val)), ds.isel(time=slice(n - n_val, None))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def test_rf_fit_predict_shapes():
    ds = _make_dataset(n_time=120)
    m = RandomForestModel(n_estimators=20, random_state=0).fit(ds)
    pred = m.predict(ds)
    assert pred.dims == ds["target"].dims
    assert pred.shape == ds["target"].shape


def test_rf_feature_importance_ranks_signal():
    ds = _make_dataset(n_time=400, noise=0.1, seed=42)
    m = RandomForestModel(n_estimators=100, max_depth=8, random_state=0).fit(ds)
    fi = m.feature_importance()
    assert set(fi) == {"noise_var", "precip", "spei3"}
    # spei3 (coef 1.5) is the dominant signal → should top the ranking
    assert fi["spei3"] > fi["precip"]
    assert fi["spei3"] > fi["noise_var"]


def test_rf_max_depth_limits_capacity():
    ds = _make_dataset(n_time=300, noise=0.5)
    m_shallow = RandomForestModel(n_estimators=30, max_depth=2, random_state=0).fit(ds)
    m_deep = RandomForestModel(n_estimators=30, max_depth=None, random_state=0).fit(ds)
    # Deep RF should fit training data more tightly (lower MSE)
    train_mse = lambda m: float(np.nanmean((m.predict(ds).values - ds["target"].values) ** 2))
    assert train_mse(m_deep) < train_mse(m_shallow)


def test_rf_predict_propagates_nan():
    ds = _make_dataset(n_time=120)
    m = RandomForestModel(n_estimators=20, random_state=0).fit(ds)
    ds_pred = ds.copy()
    ds_pred["spei3"] = ds_pred["spei3"].copy()
    ds_pred["spei3"].values[:5, 0, 0] = np.nan
    pred = m.predict(ds_pred)
    assert np.all(np.isnan(pred.values[:5, 0, 0]))
    assert np.all(np.isfinite(pred.values[:5, 1, 1]))


# ---------------------------------------------------------------------------
# XGBoost — without val (early stopping disabled)
# ---------------------------------------------------------------------------

def test_xgboost_fit_without_val_uses_full_n_estimators():
    ds = _make_dataset(n_time=150, noise=0.2)
    m = XGBoostModel(n_estimators=20, learning_rate=0.1, random_state=0).fit(ds)
    pred = m.predict(ds)
    assert pred.shape == ds["target"].shape
    # No early stopping → best_iteration is None or the cap
    assert m.best_iteration in (None, m.n_estimators - 1, m.n_estimators)


def test_xgboost_feature_importance_ranks_signal():
    ds = _make_dataset(n_time=400, noise=0.1, seed=42)
    m = XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=4,
                     early_stopping_rounds=None, random_state=0).fit(ds)
    fi = m.feature_importance()
    assert set(fi) == {"noise_var", "precip", "spei3"}
    assert fi["spei3"] > fi["precip"]
    assert fi["spei3"] > fi["noise_var"]


# ---------------------------------------------------------------------------
# XGBoost — with val (early stopping enabled)
# ---------------------------------------------------------------------------

def test_xgboost_early_stopping_with_val():
    ds = _make_dataset(n_time=300, noise=0.2, seed=42)
    train, val = _split_train_val(ds, val_frac=0.25)
    m = XGBoostModel(
        n_estimators=500, learning_rate=0.1, max_depth=4,
        early_stopping_rounds=10, random_state=0,
    )
    m.fit(train, val=val)
    # Early stopping should kick in BEFORE the full 500
    assert m.best_iteration is not None
    assert m.best_iteration < 499


def test_xgboost_predict_propagates_nan():
    ds = _make_dataset(n_time=150, noise=0.3)
    m = XGBoostModel(n_estimators=20, early_stopping_rounds=None, random_state=0).fit(ds)
    ds_pred = ds.copy()
    ds_pred["spei3"] = ds_pred["spei3"].copy()
    ds_pred["spei3"].values[:5, 0, 0] = np.nan
    pred = m.predict(ds_pred)
    assert np.all(np.isnan(pred.values[:5, 0, 0]))
    assert np.all(np.isfinite(pred.values[:5, 1, 1]))


# ---------------------------------------------------------------------------
# Common error handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda: RandomForestModel(n_estimators=10, random_state=0),
    lambda: XGBoostModel(n_estimators=10, early_stopping_rounds=None, random_state=0),
])
def test_predict_missing_feature_raises(factory):
    ds = _make_dataset(n_time=80)
    m = factory().fit(ds)
    ds_bad = ds.drop_vars("noise_var")
    with pytest.raises(ValueError, match="features missing"):
        m.predict(ds_bad)


@pytest.mark.parametrize("factory", [
    lambda: RandomForestModel(n_estimators=10, random_state=0),
    lambda: XGBoostModel(n_estimators=10, early_stopping_rounds=None, random_state=0),
])
def test_requires_target_in_train(factory):
    ds = _make_dataset(n_time=80).drop_vars("target")
    with pytest.raises(ValueError, match="'target'"):
        factory().fit(ds)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["rf", "xgboost"])
def test_registry_includes_trees(name):
    assert name in REGISTRY


def test_get_model_rf_with_params():
    m = get_model("rf", n_estimators=50, max_depth=10, random_state=7)
    assert isinstance(m, RandomForestModel)
    assert m.n_estimators == 50
    assert m.max_depth == 10


def test_get_model_xgboost_with_params():
    m = get_model("xgboost", max_depth=8, learning_rate=0.05, reg_lambda=10.0,
                  early_stopping_rounds=20)
    assert isinstance(m, XGBoostModel)
    assert m.max_depth == 8
    assert m.learning_rate == 0.05
    assert m.reg_lambda == 10.0
    assert m.early_stopping_rounds == 20


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [
    lambda: RandomForestModel(n_estimators=20, random_state=0),
    lambda: XGBoostModel(n_estimators=20, early_stopping_rounds=None, random_state=0),
])
def test_all_tree_models_implement_basemodel_interface(factory):
    ds = _make_dataset(n_time=120)
    m = factory()
    assert isinstance(m, TreeBaseModel)
    m.fit(ds)
    assert m.feature_names_ is not None
    pred = m.predict(ds)
    assert isinstance(pred, xr.DataArray)
    assert pred.shape == ds["target"].shape
    fi = m.feature_importance()
    assert fi is not None
    assert set(fi) == set(m.feature_names_)
