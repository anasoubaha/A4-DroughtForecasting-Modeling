"""Unit tests for droughtmodel.evaluation.block_bootstrap_ci."""

import numpy as np
import pytest

from droughtmodel import evaluation as deval


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(200)
    y_pred = y_true + 0.3 * rng.standard_normal(200)
    result = deval.block_bootstrap_ci(
        deval.rmse, y_pred, y_true,
        n_replicates=200, mean_block_length=12, seed=42,
    )
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_bootstrap_returns_expected_keys():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(60)
    result = deval.block_bootstrap_ci(deval.mae, y, y, n_replicates=50, seed=0)
    assert set(result.keys()) == {"estimate", "lower", "upper", "std", "n_replicates"}
    assert result["n_replicates"] == 50


def test_bootstrap_reproducibility_with_seed():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(100)
    y_pred = y_true + 0.5 * rng.standard_normal(100)
    r1 = deval.block_bootstrap_ci(deval.rmse, y_pred, y_true, n_replicates=100, seed=42)
    r2 = deval.block_bootstrap_ci(deval.rmse, y_pred, y_true, n_replicates=100, seed=42)
    assert r1["lower"] == r2["lower"]
    assert r1["upper"] == r2["upper"]
    assert r1["std"] == r2["std"]


def test_bootstrap_three_arg_metric():
    """Bootstrap with a 3-arg metric (ACC needs climatology)."""
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(100)
    y_pred = y_true + 0.4 * rng.standard_normal(100)
    y_clim = rng.standard_normal(100)
    result = deval.block_bootstrap_ci(
        deval.acc, y_pred, y_true, y_clim,
        n_replicates=100, mean_block_length=12, seed=42,
    )
    assert -1.0 <= result["lower"] <= result["upper"] <= 1.0
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_bootstrap_mismatched_lengths_raises():
    a = np.zeros(10)
    b = np.zeros(11)
    with pytest.raises(ValueError, match="same length along time_axis"):
        deval.block_bootstrap_ci(deval.mae, a, b, n_replicates=10, seed=0)


def test_bootstrap_multidim_resampling():
    """For 2-D arrays (time × cell), bootstrap should resample time and keep all cells per time step."""
    rng = np.random.default_rng(0)
    n_time, n_cell = 100, 20
    y_true = rng.standard_normal((n_time, n_cell))
    y_pred = y_true + 0.5 * rng.standard_normal((n_time, n_cell))
    result = deval.block_bootstrap_ci(
        deval.rmse, y_pred, y_true,
        n_replicates=100, mean_block_length=12, seed=42, time_axis=0,
    )
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert result["std"] > 0  # variation expected from bootstrap


def test_bootstrap_width_shrinks_with_more_data():
    """CI should be tighter (smaller std) with more samples (~ sqrt-of-N scaling)."""
    rng = np.random.default_rng(0)
    y_short = rng.standard_normal(50)
    pred_short = y_short + 0.3 * rng.standard_normal(50)
    y_long = rng.standard_normal(500)
    pred_long = y_long + 0.3 * rng.standard_normal(500)
    short = deval.block_bootstrap_ci(deval.rmse, pred_short, y_short, n_replicates=200, seed=0)
    long_ = deval.block_bootstrap_ci(deval.rmse, pred_long, y_long, n_replicates=200, seed=0)
    assert long_["std"] < short["std"]


def test_bootstrap_winter_block_length():
    """Block length should be configurable (e.g., 4 for winter year-blocks)."""
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(100)
    y_pred = y_true + 0.3 * rng.standard_normal(100)
    r4 = deval.block_bootstrap_ci(deval.rmse, y_pred, y_true, n_replicates=100, mean_block_length=4, seed=42)
    r12 = deval.block_bootstrap_ci(deval.rmse, y_pred, y_true, n_replicates=100, mean_block_length=12, seed=42)
    # Both should bracket the estimate; both should be reasonable
    assert r4["lower"] <= r4["estimate"] <= r4["upper"]
    assert r12["lower"] <= r12["estimate"] <= r12["upper"]
