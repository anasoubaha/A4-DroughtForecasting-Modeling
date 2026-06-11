"""Unit tests for feature importance diagnostics (v3 §6)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from droughtmodel.selection import (
    SCORING_FUNCS,
    permutation_importance_scores,
    tree_shap_importance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def signal_noise_data():
    """y depends only on feature 'a'; b/c/d/e are pure noise."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 5))
    y = 2.0 * X[:, 0] + 0.1 * rng.standard_normal(n)
    return X, y, ["a", "b", "c", "d", "e"]


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def test_permutation_returns_well_formed_dataframe(signal_noise_data):
    X, y, names = signal_noise_data
    model = LinearRegression().fit(X, y)
    df = permutation_importance_scores(model.predict, X, y, names, n_repeats=3)
    assert list(df.columns) == ["feature", "importance_mean", "importance_std"]
    assert len(df) == 5
    assert set(df["feature"]) == set(names)


def test_permutation_ranks_signal_above_noise(signal_noise_data):
    X, y, names = signal_noise_data
    model = LinearRegression().fit(X, y)
    df = permutation_importance_scores(model.predict, X, y, names, n_repeats=10)
    assert df.iloc[0]["feature"] == "a"
    # Signal importance should dwarf any noise feature
    assert df.iloc[0]["importance_mean"] > 5 * df.iloc[1]["importance_mean"]


def test_permutation_is_sorted_descending(signal_noise_data):
    X, y, names = signal_noise_data
    model = LinearRegression().fit(X, y)
    df = permutation_importance_scores(model.predict, X, y, names, n_repeats=3)
    means = df["importance_mean"].values
    assert all(means[i] >= means[i + 1] for i in range(len(means) - 1))


def test_permutation_scoring_keys(signal_noise_data):
    X, y, names = signal_noise_data
    model = LinearRegression().fit(X, y)
    for key in ("neg_mse", "neg_mae", "r2"):
        df = permutation_importance_scores(model.predict, X, y, names, n_repeats=3, scoring=key)
        assert df.iloc[0]["feature"] == "a"


def test_permutation_custom_scoring_callable(signal_noise_data):
    X, y, names = signal_noise_data
    model = LinearRegression().fit(X, y)
    custom = lambda yt, yp: -float(np.mean(np.abs(yt - yp)))
    df = permutation_importance_scores(model.predict, X, y, names, n_repeats=3, scoring=custom)
    assert df.iloc[0]["feature"] == "a"


def test_permutation_drops_nan_rows(signal_noise_data):
    X, y, names = signal_noise_data
    X, y = X.copy(), y.copy()
    X[:5, 0] = np.nan
    y[5:10] = np.nan
    # Fit on the clean tail; importance is computed on the same X/y after NaN drop
    model = LinearRegression().fit(X[10:], y[10:])
    df = permutation_importance_scores(model.predict, X, y, names, n_repeats=3)
    assert df.iloc[0]["feature"] == "a"
    assert df["importance_mean"].notna().all()


def test_permutation_feature_name_length_mismatch_raises(signal_noise_data):
    X, y, _ = signal_noise_data
    model = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match="feature_names length"):
        permutation_importance_scores(model.predict, X, y, ["a", "b"], n_repeats=2)


def test_permutation_empty_after_filtering_raises():
    X = np.full((10, 3), np.nan)
    y = np.zeros(10)
    with pytest.raises(ValueError, match="No finite samples"):
        permutation_importance_scores(
            lambda x: np.zeros(x.shape[0]), X, y, ["a", "b", "c"], n_repeats=1,
        )


def test_permutation_does_not_mutate_X(signal_noise_data):
    X, y, names = signal_noise_data
    X_copy = X.copy()
    model = LinearRegression().fit(X, y)
    permutation_importance_scores(model.predict, X, y, names, n_repeats=3)
    assert np.array_equal(X, X_copy)


# ---------------------------------------------------------------------------
# SHAP importance
# ---------------------------------------------------------------------------

def test_shap_returns_df_and_values(signal_noise_data):
    X, y, names = signal_noise_data
    model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    df, sv = tree_shap_importance(model, X, names)
    assert list(df.columns) == ["feature", "mean_abs_shap"]
    assert sv.shape == (X.shape[0], X.shape[1])
    assert set(df["feature"]) == set(names)


def test_shap_ranks_signal_above_noise(signal_noise_data):
    X, y, names = signal_noise_data
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0).fit(X, y)
    df, _ = tree_shap_importance(model, X, names)
    assert df.iloc[0]["feature"] == "a"


def test_shap_respects_max_samples(signal_noise_data):
    X, y, names = signal_noise_data
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    _, sv = tree_shap_importance(model, X, names, max_samples=100)
    assert sv.shape[0] == 100


def test_shap_feature_name_mismatch_raises(signal_noise_data):
    X, y, _ = signal_noise_data
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="feature_names length"):
        tree_shap_importance(model, X, ["a", "b"])


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def test_scoring_funcs_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert SCORING_FUNCS["neg_mse"](y, y) == 0.0
    assert SCORING_FUNCS["neg_mae"](y, y) == 0.0
    assert SCORING_FUNCS["r2"](y, y) == pytest.approx(1.0)


def test_scoring_funcs_constant_predictions_r2_zero():
    y = np.array([1.0, 2.0, 3.0])
    yp = np.full_like(y, y.mean())   # mean predictor → R² = 0 exactly
    assert SCORING_FUNCS["r2"](y, yp) == pytest.approx(0.0)
