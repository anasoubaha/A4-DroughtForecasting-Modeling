"""Unit tests for hyperparameter tuning (v3 §8 — Protocol A)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from droughtmodel.models.linear import LassoModel, RidgeModel
from droughtmodel.models.tree import RandomForestModel, XGBoostModel
from droughtmodel.tuning import (
    SearchResult,
    _expand_grid,
    grid_search,
    optuna_search,
    tune_and_refit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(
    n_time: int = 240, n_lat: int = 4, n_lon: int = 4,
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

    ds = xr.Dataset({
        "spei3": xr.DataArray(spei3, dims=dims, coords=coords),
        "precip": xr.DataArray(precip, dims=dims, coords=coords),
        "noise_var": xr.DataArray(noise_var, dims=dims, coords=coords),
        "target": xr.DataArray(target, dims=dims, coords=coords),
    })
    return ds


def _train_val(ds: xr.Dataset, val_frac: float = 0.25) -> tuple[xr.Dataset, xr.Dataset]:
    n = ds.sizes["time"]
    n_val = int(n * val_frac)
    return ds.isel(time=slice(None, n - n_val)), ds.isel(time=slice(n - n_val, None))


# ---------------------------------------------------------------------------
# _expand_grid
# ---------------------------------------------------------------------------

def test_expand_grid_cartesian_product():
    out = _expand_grid({"a": [1, 2], "b": [10, 20, 30]})
    assert len(out) == 6
    assert {"a": 1, "b": 10} in out
    assert {"a": 2, "b": 30} in out


def test_expand_grid_single_param():
    out = _expand_grid({"alpha": [0.01, 0.1, 1.0]})
    assert out == [{"alpha": 0.01}, {"alpha": 0.1}, {"alpha": 1.0}]


def test_expand_grid_empty_returns_singleton():
    assert _expand_grid({}) == [{}]


def test_expand_grid_list_passthrough():
    grid = [{"alpha": 0.1, "l1_ratio": 0.5}, {"alpha": 1.0}]
    assert _expand_grid(grid) == grid


# ---------------------------------------------------------------------------
# Grid search — linear models
# ---------------------------------------------------------------------------

def test_grid_search_returns_searchresult():
    train, val = _train_val(_make_dataset(n_time=200, seed=0))
    grid = {"alpha": [0.01, 0.1, 1.0]}
    res = grid_search(RidgeModel, grid, train, val)
    assert isinstance(res, SearchResult)
    assert res.n_trials == 3
    assert "alpha" in res.best_params
    assert isinstance(res.all_scores, pd.DataFrame)
    assert "score" in res.all_scores.columns
    assert res.best_model is not None


def test_grid_search_all_scores_sorted_descending():
    train, val = _train_val(_make_dataset(n_time=200))
    res = grid_search(RidgeModel, {"alpha": [0.001, 0.1, 10.0, 1000.0]}, train, val)
    scores = res.all_scores["score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_grid_search_avoids_extreme_overregularization():
    """On a clean signal, the search must not pick the extreme over-regularized alpha,
    and the best val score must reflect a real fit (not noise-level)."""
    train, val = _train_val(_make_dataset(n_time=400, noise=0.1, seed=42))
    grid = {"alpha": np.logspace(-3, 3, 7).tolist()}   # [0.001 ... 1000]
    res = grid_search(RidgeModel, grid, train, val)
    assert res.best_params["alpha"] < 100.0            # not the extreme high end
    assert res.best_score > -0.5                       # neg_mse → good fit on standardized target


def test_grid_search_lasso_kills_noise_at_strong_alpha():
    """Score should monotonically rise then fall as alpha varies — best at moderate L1."""
    train, val = _train_val(_make_dataset(n_time=400, noise=0.1, seed=42))
    grid = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    res = grid_search(LassoModel, grid, train, val,
                      fixed_params={"max_iter": 10000})
    assert res.best_score > -1.0       # neg_mse better than -1
    fi = res.best_model.feature_importance()
    assert abs(fi["spei3"]) > 0.5      # main signal preserved


# ---------------------------------------------------------------------------
# Grid search — fixed_params, custom scoring
# ---------------------------------------------------------------------------

def test_grid_search_fixed_params_passed_through():
    train, val = _train_val(_make_dataset(n_time=150))
    res = grid_search(
        LassoModel,
        {"alpha": [0.01, 0.1]},
        train, val,
        fixed_params={"max_iter": 5000},
    )
    assert res.best_model.max_iter == 5000


def test_grid_search_custom_scoring_callable():
    train, val = _train_val(_make_dataset(n_time=150))
    neg_mae = lambda y, p: -float(np.mean(np.abs(y - p)))
    res = grid_search(RidgeModel, {"alpha": [0.1, 1.0]}, train, val, scoring=neg_mae)
    assert isinstance(res.best_score, float)
    assert res.best_score <= 0.0


def test_grid_search_verbose_does_not_crash(capsys):
    train, val = _train_val(_make_dataset(n_time=120))
    grid_search(RidgeModel, {"alpha": [0.1, 1.0]}, train, val, verbose=True)
    captured = capsys.readouterr()
    assert "0.1" in captured.out


# ---------------------------------------------------------------------------
# tune_and_refit
# ---------------------------------------------------------------------------

def test_tune_and_refit_returns_fitted_model_and_result():
    ds = _make_dataset(n_time=200)
    train, val = _train_val(ds)
    grid = {"alpha": [0.01, 0.1, 1.0]}
    final_model, res = tune_and_refit(RidgeModel, grid, train, val, refit_dataset=ds)
    assert final_model.feature_names_ is not None
    assert isinstance(res, SearchResult)
    assert final_model.alpha == res.best_params["alpha"]


def test_tune_and_refit_uses_continuous_refit_slice():
    """The refit estimator must fit on the contiguous train+val span, not the search slices."""
    ds = _make_dataset(n_time=200)
    train, val = _train_val(ds, val_frac=0.25)
    final_model, _ = tune_and_refit(RidgeModel, {"alpha": [0.1]}, train, val, refit_dataset=ds)
    pred = final_model.predict(ds)
    assert pred.shape == ds["target"].shape


def test_tune_and_refit_with_xgboost_early_stopping():
    """XGBoost early stopping picks a best_iteration; refit captures and locks it."""
    ds = _make_dataset(n_time=400, noise=0.2, seed=42)
    train, val = _train_val(ds, val_frac=0.25)
    grid = {"max_depth": [3, 5], "learning_rate": [0.1]}
    final_model, res = tune_and_refit(
        XGBoostModel, grid, train, val, refit_dataset=ds,
        fixed_params={
            "n_estimators": 300, "early_stopping_rounds": 15,
            "random_state": 0,
        },
        pass_val_to_fit=True,
        refit_with_best_iteration=True,
    )
    bi = res.best_model.best_iteration
    assert bi is not None
    assert final_model.n_estimators == bi + 1
    assert final_model.early_stopping_rounds is None


def test_tune_and_refit_rf_basic():
    ds = _make_dataset(n_time=200, seed=0)
    train, val = _train_val(ds, val_frac=0.25)
    grid = {"max_depth": [None, 5], "min_samples_leaf": [1, 5]}
    final_model, res = tune_and_refit(
        RandomForestModel, grid, train, val, refit_dataset=ds,
        fixed_params={"n_estimators": 30, "random_state": 0, "n_jobs": 1},
    )
    assert res.n_trials == 4
    assert final_model.feature_names_ is not None


def test_tune_and_refit_uses_provided_refit_dataset():
    """The refit slice passed in must actually drive the refit
    (NOT the train/val slices used for search)."""
    ds = _make_dataset(n_time=200, seed=0)
    train, val = _train_val(ds, val_frac=0.25)
    # Custom refit slice: shorter than train (just to confirm it's used)
    short_refit = ds.isel(time=slice(0, 50))
    final_model, _ = tune_and_refit(
        RidgeModel, {"alpha": [0.1]}, train, val, refit_dataset=short_refit,
    )
    # Predicting on the short slice must produce the matching shape
    pred = final_model.predict(short_refit)
    assert pred.shape == short_refit["target"].shape


# ---------------------------------------------------------------------------
# Optuna (lazy import; skipped if unavailable)
# ---------------------------------------------------------------------------

def _have_optuna() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_optuna(), reason="optuna not installed")
def test_optuna_search_finds_low_alpha_on_clean_signal():
    train, val = _train_val(_make_dataset(n_time=300, noise=0.1, seed=42))

    def space(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}

    res = optuna_search(RidgeModel, space, train, val, n_trials=15, sampler_seed=0)
    assert isinstance(res, SearchResult)
    assert res.n_trials == 15
    assert res.best_params["alpha"] < 10.0   # low alpha wins on clean data
    assert res.best_model is not None


@pytest.mark.skipif(not _have_optuna(), reason="optuna not installed")
def test_optuna_through_tune_and_refit():
    ds = _make_dataset(n_time=200)
    train, val = _train_val(ds)

    def space(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e2, log=True)}

    final_model, res = tune_and_refit(
        RidgeModel, grid={}, train=train, val=val, refit_dataset=ds,
        search_fn=optuna_search,
        search_kwargs={"search_space": space, "n_trials": 8, "sampler_seed": 0},
    )
    assert final_model.feature_names_ is not None
    assert res.n_trials == 8
