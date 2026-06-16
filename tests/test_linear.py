"""Unit tests for linear models (v3 §7.1)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from droughtmodel.models.linear import (
    ElasticNetModel,
    LassoModel,
    LinearModel,
    LinearRegressionModel,
    RidgeModel,
)
from droughtmodel.models.registry import REGISTRY, get_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(
    n_time: int = 240, n_lat: int = 4, n_lon: int = 4, lead: int = 3,
    noise: float = 0.3, seed: int = 0,
) -> xr.Dataset:
    """Linear dataset: target = 1.5·spei3 + 0.5·precip + 0·noise_var + ε.

    Target is a function of *contemporaneous* features so that the linear
    models can recover the construction coefficients. The last ``lead``
    time steps have NaN target to mimic the lead-shift drop pattern that
    the real pipeline produces (`build_dataset` shifts SPEI3 by L).
    """
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


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

def test_ols_fit_predict_shapes():
    ds = _make_dataset(n_time=120)
    m = LinearRegressionModel().fit(ds)
    pred = m.predict(ds)
    assert pred.dims == ds["target"].dims
    assert pred.shape == ds["target"].shape


def test_ols_recovers_known_linear_signal():
    ds = _make_dataset(n_time=400, noise=0.1, lead=1, seed=42)
    m = LinearRegressionModel().fit(ds)
    fi = m.feature_importance()
    assert set(fi) == {"noise_var", "precip", "spei3"}
    assert abs(fi["spei3"] - 1.5) < 0.05
    assert abs(fi["precip"] - 0.5) < 0.05
    assert abs(fi["noise_var"]) < 0.05


def test_ols_requires_target():
    ds = _make_dataset().drop_vars("target")
    with pytest.raises(ValueError, match="'target'"):
        LinearRegressionModel().fit(ds)


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

def test_ridge_alpha_shrinks_coefs():
    ds = _make_dataset(n_time=300, seed=42)
    m_low = RidgeModel(alpha=0.001).fit(ds)
    m_high = RidgeModel(alpha=1000.0).fit(ds)
    fi_low = m_low.feature_importance()
    fi_high = m_high.feature_importance()
    # Strong regularization should shrink the signal coefficient noticeably
    assert abs(fi_high["spei3"]) < abs(fi_low["spei3"])


def test_ridge_fit_predict_shapes():
    ds = _make_dataset()
    m = RidgeModel(alpha=1.0).fit(ds)
    pred = m.predict(ds)
    assert pred.shape == ds["target"].shape


# ---------------------------------------------------------------------------
# Lasso
# ---------------------------------------------------------------------------

def test_lasso_kills_noise_feature_at_strong_alpha():
    ds = _make_dataset(n_time=400, noise=0.1, lead=1, seed=42)
    m = LassoModel(alpha=0.05).fit(ds)
    fi = m.feature_importance()
    assert abs(fi["spei3"]) > 0.5            # true signal kept
    assert abs(fi["noise_var"]) < 0.01       # noise feature zeroed by L1


def test_lasso_default_alpha_recovers_signal():
    ds = _make_dataset(n_time=400, noise=0.1, lead=1, seed=42)
    m = LassoModel(alpha=0.001).fit(ds)      # ~weak penalty
    fi = m.feature_importance()
    assert abs(fi["spei3"] - 1.5) < 0.2


# ---------------------------------------------------------------------------
# Elastic Net
# ---------------------------------------------------------------------------

def test_elasticnet_recovers_signal_combines_l1_l2():
    ds = _make_dataset(n_time=400, noise=0.1, lead=1, seed=42)
    m = ElasticNetModel(alpha=0.01, l1_ratio=0.5).fit(ds)
    fi = m.feature_importance()
    assert abs(fi["spei3"]) > 0.5
    # At this strength some L1 zeroing of noise expected
    assert abs(fi["noise_var"]) < 0.1


def test_elasticnet_l1_ratio_zero_equivalent_to_ridge():
    ds = _make_dataset(n_time=300, noise=0.1, lead=1, seed=0)
    m_en = ElasticNetModel(alpha=0.1, l1_ratio=0.0001, max_iter=20000).fit(ds)
    m_rg = RidgeModel(alpha=0.1).fit(ds)
    fi_en = m_en.feature_importance()
    fi_rg = m_rg.feature_importance()
    # Should be close but not exactly equal (different solvers, different scaling conventions)
    for k in fi_en:
        assert abs(fi_en[k] - fi_rg[k]) < 0.2


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_predict_propagates_nan_for_nan_feature_rows():
    ds = _make_dataset(n_time=200, seed=0)
    m = RidgeModel().fit(ds)
    ds_pred = ds.copy()
    # Inject NaN in spei3 at cell (0, 0) for the first 5 time steps
    ds_pred["spei3"] = ds_pred["spei3"].copy()
    ds_pred["spei3"].values[:5, 0, 0] = np.nan
    pred = m.predict(ds_pred)
    assert np.all(np.isnan(pred.values[:5, 0, 0]))
    assert np.all(np.isfinite(pred.values[:5, 1, 1]))


def test_fit_drops_nan_rows():
    # The last `lead` time steps have NaN target — fit should still succeed
    ds = _make_dataset(n_time=100, lead=3)
    m = LinearRegressionModel().fit(ds)
    assert m.feature_names_ is not None


def test_predict_missing_feature_raises():
    ds = _make_dataset(n_time=100)
    m = RidgeModel().fit(ds)
    ds_bad = ds.drop_vars("noise_var")
    with pytest.raises(ValueError, match="features missing"):
        m.predict(ds_bad)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ols", "ridge", "lasso", "elasticnet"])
def test_registry_includes_linear(name):
    assert name in REGISTRY


def test_get_model_ridge_with_params():
    m = get_model("ridge", alpha=2.5)
    assert isinstance(m, RidgeModel)
    assert m.alpha == 2.5


def test_get_model_lasso_with_params():
    m = get_model("lasso", alpha=0.1, max_iter=5000)
    assert isinstance(m, LassoModel)
    assert m.alpha == 0.1
    assert m.max_iter == 5000


def test_get_model_elasticnet_with_params():
    m = get_model("elasticnet", alpha=0.5, l1_ratio=0.7)
    assert isinstance(m, ElasticNetModel)
    assert m.l1_ratio == 0.7


def test_get_model_ols_no_params():
    m = get_model("ols")
    assert isinstance(m, LinearRegressionModel)


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel])
def test_all_linear_models_implement_basemodel_interface(factory):
    """Each linear model must expose fit / predict / feature_importance correctly."""
    ds = _make_dataset(n_time=120, lead=2)
    m = factory()
    assert isinstance(m, LinearModel)
    m.fit(ds)
    assert m.feature_names_ is not None
    pred = m.predict(ds)
    assert isinstance(pred, xr.DataArray)
    assert pred.shape == ds["target"].shape
    fi = m.feature_importance()
    assert fi is not None
    assert set(fi) == set(m.feature_names_)


# ---------------------------------------------------------------------------
# Heterogeneous feature dimensions (regression — real-data shape mismatch)
# ---------------------------------------------------------------------------

def _make_heterogeneous_dataset(n_time: int = 150, seed: int = 0) -> xr.Dataset:
    """Mixed-dimensionality features: 3-D gridded vars + 1-D climate indices.

    Mirrors the real-data shape that broke `_stack_xy` before broadcasting
    was added: NAO/ENSO/MO arrive as `(time,)` series alongside `(time, lat, lon)`
    predictors. The target depends on both kinds.
    """
    rng = np.random.default_rng(seed)
    n_lat, n_lon = 3, 3
    times = pd.date_range("1950-01", periods=n_time, freq="MS")
    lats = np.linspace(28, 36, n_lat)
    lons = np.linspace(-13, -1, n_lon)
    coords3d = {"time": times, "lat": lats, "lon": lons}

    # 3-D gridded predictors
    spei3 = rng.standard_normal((n_time, n_lat, n_lon))
    precip = rng.standard_normal((n_time, n_lat, n_lon))

    # 1-D climate indices (same value at every cell at a given month)
    nao = rng.standard_normal(n_time)
    enso = rng.standard_normal(n_time)

    # Target depends on a gridded var AND a 1-D index
    target = (
        1.2 * spei3
        + 0.6 * nao[:, None, None]                          # broadcast manually
        + 0.1 * rng.standard_normal((n_time, n_lat, n_lon))
    )

    return xr.Dataset({
        "spei3": xr.DataArray(spei3, dims=("time","lat","lon"), coords=coords3d),
        "precip": xr.DataArray(precip, dims=("time","lat","lon"), coords=coords3d),
        "nao":   xr.DataArray(nao,  dims=("time",), coords={"time": times}),
        "enso":  xr.DataArray(enso, dims=("time",), coords={"time": times}),
        "target": xr.DataArray(target, dims=("time","lat","lon"), coords=coords3d),
    })


def test_fit_predict_with_heterogeneous_feature_dims():
    """1-D climate indices alongside 3-D gridded vars must fit without shape errors."""
    ds = _make_heterogeneous_dataset(n_time=200, seed=42)
    m = RidgeModel(alpha=0.1).fit(ds)
    assert m.feature_names_ is not None
    assert set(m.feature_names_) == {"enso", "nao", "precip", "spei3"}
    pred = m.predict(ds)
    assert pred.dims == ds["target"].dims
    assert pred.shape == ds["target"].shape


def test_heterogeneous_features_recover_known_signal():
    """OLS on the mixed-dim dataset must recover ~1.2 for spei3 and ~0.6 for nao."""
    ds = _make_heterogeneous_dataset(n_time=400, seed=42)
    m = LinearRegressionModel().fit(ds)
    fi = m.feature_importance()
    assert fi is not None
    assert abs(fi["spei3"] - 1.2) < 0.05
    assert abs(fi["nao"] - 0.6) < 0.05
    assert abs(fi["precip"]) < 0.05
    assert abs(fi["enso"]) < 0.05
