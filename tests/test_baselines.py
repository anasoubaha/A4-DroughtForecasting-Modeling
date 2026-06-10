"""Unit tests for baselines (v3 §7.0)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from droughtmodel.models.baselines import (
    ARBaseline,
    ClimatologyBaseline,
    PersistenceBaseline,
)
from droughtmodel.models.registry import REGISTRY, get_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(n_time: int = 120, n_lat: int = 4, n_lon: int = 4, lead: int = 3,
                  seed: int = 0) -> xr.Dataset:
    """Build a minimal feature dataset with spei3, spei3_lag1/2, and target."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("1950-01", periods=n_time, freq="MS")
    lats = np.linspace(28, 36, n_lat)
    lons = np.linspace(-13, -1, n_lon)

    # Build an AR(2) SPEI3 process so the baselines have signal to fit
    spei3 = np.zeros((n_time, n_lat, n_lon))
    spei3[0] = rng.standard_normal((n_lat, n_lon))
    spei3[1] = rng.standard_normal((n_lat, n_lon))
    for t in range(2, n_time):
        spei3[t] = 0.6 * spei3[t - 1] - 0.2 * spei3[t - 2] + 0.5 * rng.standard_normal((n_lat, n_lon))

    spei3_da = xr.DataArray(spei3, dims=("time", "lat", "lon"),
                             coords={"time": times, "lat": lats, "lon": lons}, name="spei3")
    target = spei3_da.shift(time=-lead)
    target.name = "target"

    ds = xr.Dataset({
        "spei3": spei3_da,
        "spei3_lag1": spei3_da.shift(time=1),
        "spei3_lag2": spei3_da.shift(time=2),
        "target": target,
    })
    ds.attrs["lead"] = lead
    return ds


# ---------------------------------------------------------------------------
# ClimatologyBaseline
# ---------------------------------------------------------------------------

def test_climatology_fit_predict_shapes():
    ds = _make_dataset(n_time=120, lead=3)
    clim = ClimatologyBaseline().fit(ds)
    assert clim.monthly_means.dims == ("target_month", "lat", "lon")
    assert set(clim.monthly_means["target_month"].values) <= set(range(1, 13))

    pred = clim.predict(ds)
    assert pred.dims == ds["target"].dims
    assert pred.shape == ds["target"].shape


def test_climatology_predicts_same_month_same_value():
    """At the same cell, climatology predictions for the same target month must be identical."""
    ds = _make_dataset(n_time=240, lead=3)
    clim = ClimatologyBaseline().fit(ds)
    pred = clim.predict(ds)
    # Pick two time steps whose target month (issue_month + 3) is the same.
    # issue=Jan, target=Apr;  issue=Jan one year later, target=Apr → same target month
    target_months = (pd.DatetimeIndex(ds["time"].values) + pd.DateOffset(months=3)).month
    idx_apr = np.where(target_months == 4)[0]
    assert len(idx_apr) >= 2
    assert np.allclose(pred.values[idx_apr[0]], pred.values[idx_apr[1]])


def test_climatology_requires_target():
    ds = _make_dataset().drop_vars("target")
    with pytest.raises(ValueError, match="'target' variable"):
        ClimatologyBaseline().fit(ds)


# ---------------------------------------------------------------------------
# PersistenceBaseline
# ---------------------------------------------------------------------------

def test_persistence_returns_spei3():
    ds = _make_dataset(lead=3)
    pred = PersistenceBaseline().fit(ds).predict(ds)
    assert np.array_equal(pred.values, ds["spei3"].values, equal_nan=True)


def test_persistence_fit_is_noop_no_error():
    ds = _make_dataset()
    p = PersistenceBaseline().fit(ds, val=None)
    # No state to verify; just confirm fit() returns self
    assert isinstance(p, PersistenceBaseline)


def test_persistence_missing_spei3_raises():
    ds = _make_dataset().drop_vars("spei3")
    with pytest.raises(ValueError, match="'spei3' not found"):
        PersistenceBaseline().fit(ds)


# ---------------------------------------------------------------------------
# ARBaseline
# ---------------------------------------------------------------------------

def test_ar_fit_predict_shapes():
    ds = _make_dataset(n_time=200, lead=3)
    # Drop the last `lead` rows for fitting (target is NaN there)
    train = ds.isel(time=slice(2, 195))  # leave lag rows out, leave room for target shift
    ar = ARBaseline(p=3, alpha=1.0).fit(train)
    assert ar.feature_names_ == ["spei3", "spei3_lag1", "spei3_lag2"]
    assert ar.coef_.shape == (3,)
    assert isinstance(ar.intercept_, float)

    pred = ar.predict(train)
    assert pred.dims == train["target"].dims


def test_ar_p1_uses_only_contemporary_spei3():
    ds = _make_dataset(n_time=200, lead=3).isel(time=slice(2, 195))
    ar = ARBaseline(p=1, alpha=1.0).fit(ds)
    assert ar.feature_names_ == ["spei3"]
    assert ar.coef_.shape == (1,)


def test_ar_skips_missing_lag_columns():
    """If fewer than p−1 spei3_lag* features exist, AR uses only what's available."""
    ds = _make_dataset(n_time=200, lead=3).isel(time=slice(2, 195))
    ds = ds.drop_vars(["spei3_lag2"])  # only spei3 + spei3_lag1 remain
    ar = ARBaseline(p=3, alpha=1.0).fit(ds)
    assert ar.feature_names_ == ["spei3", "spei3_lag1"]


def test_ar_predict_reasonable_on_synthetic_ar2_process():
    """AR(2) at lead 1 on a synthetic AR(2) process should have substantial training Pearson r."""
    # Use lead=1 so we're predicting one-step-ahead — predictable for AR(2).
    # At lead 3 the predictability collapses to near zero (impulse response decays fast).
    ds = _make_dataset(n_time=400, lead=1, seed=42).isel(time=slice(2, 398))
    ar = ARBaseline(p=2, alpha=0.1).fit(ds)
    pred = ar.predict(ds)
    truth = ds["target"]
    mask = np.isfinite(truth.values) & np.isfinite(pred.values)
    r = float(np.corrcoef(pred.values[mask], truth.values[mask])[0, 1])
    # AR(2) one-step-ahead on the synthetic AR(2) process should achieve r ≈ 0.5–0.7
    assert 0.3 < r < 1.0, f"Expected substantial training correlation, got r = {r:.3f}"


def test_ar_feature_importance():
    ds = _make_dataset(n_time=200, lead=3).isel(time=slice(2, 195))
    ar = ARBaseline(p=3, alpha=1.0).fit(ds)
    fi = ar.feature_importance()
    assert set(fi.keys()) == {"spei3", "spei3_lag1", "spei3_lag2"}
    for v in fi.values():
        assert isinstance(v, float)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lookup():
    assert "climatology" in REGISTRY
    assert "persistence" in REGISTRY
    assert "ar" in REGISTRY


def test_get_model_instantiates():
    m = get_model("ar", p=2, alpha=0.5)
    assert isinstance(m, ARBaseline)
    assert m.p == 2
    assert m.alpha == 0.5


def test_get_model_unknown_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model("not_a_model")
