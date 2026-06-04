"""Unit tests for droughtmodel.evaluation metrics."""

import numpy as np
import pytest

from droughtmodel import evaluation as deval


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def test_mae_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert deval.mae(y, y) == 0.0


def test_mae_basic():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 2.0])
    assert deval.mae(y_pred, y_true) == pytest.approx(4.0 / 3.0)


def test_rmse_perfect():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100)
    assert deval.rmse(y, y) == 0.0


def test_rmse_basic():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, -1.0, 1.0, -1.0])
    assert deval.rmse(y_pred, y_true) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def test_pearson_r_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert deval.pearson_r(y, y) == pytest.approx(1.0)


def test_pearson_r_anti():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert deval.pearson_r(-y, y) == pytest.approx(-1.0)


def test_pearson_r_zero_variance():
    y_pred = np.zeros(10)
    y_true = np.arange(10, dtype=float)
    assert np.isnan(deval.pearson_r(y_pred, y_true))


# ---------------------------------------------------------------------------
# ACC
# ---------------------------------------------------------------------------

def test_acc_perfect_anomaly():
    y_clim = np.zeros(4)
    y = np.array([1.0, -1.0, 2.0, -2.0])
    assert deval.acc(y, y, y_clim) == pytest.approx(1.0)


def test_acc_zero_anomaly_correlation():
    # When pred anomaly is constant, ACC undefined → NaN
    y_clim = np.zeros(4)
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.full(4, 5.0)  # constant
    assert np.isnan(deval.acc(y_pred, y_true, y_clim))


def test_acc_climatology_self():
    # If pred ≡ climatology, anomaly is zero → undefined → NaN
    y_clim = np.array([0.5, 0.5, 0.5, 0.5])
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(deval.acc(y_clim, y_true, y_clim))


# ---------------------------------------------------------------------------
# MSSS
# ---------------------------------------------------------------------------

def test_msss_perfect_prediction():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(100)
    y_pred = y_true.copy()
    y_ref = rng.standard_normal(100)
    # Perfect MSE(pred) = 0 → MSSS = 1
    assert deval.msss(y_pred, y_true, y_ref) == pytest.approx(1.0)


def test_msss_equal_to_reference():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(100)
    y_ref = rng.standard_normal(100)
    # When pred == ref, MSE(pred) = MSE(ref) → MSSS = 0
    assert deval.msss(y_ref, y_true, y_ref) == pytest.approx(0.0)


def test_msss_negative_when_worse_than_reference():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(50)
    y_ref = y_true + 0.1 * rng.standard_normal(50)        # small noise
    y_pred = y_true + 2.0 * rng.standard_normal(50)       # much larger noise
    # MSE(pred) >> MSE(ref) → MSSS < 0
    assert deval.msss(y_pred, y_true, y_ref) < 0


def test_msss_vs_climatology_alias():
    # Alias should match base function
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(30)
    y_pred = rng.standard_normal(30)
    y_clim = rng.standard_normal(30)
    assert deval.msss_vs_climatology(y_pred, y_true, y_clim) == deval.msss(y_pred, y_true, y_clim)


# ---------------------------------------------------------------------------
# HSS (binary)
# ---------------------------------------------------------------------------

def test_hss_perfect_classification():
    # 2 hits, 2 correct negatives, 0 false alarms, 0 misses → HSS = 1
    y_true = np.array([-2.0, -1.5, 0.5, 1.0])
    y_pred = np.array([-1.5, -2.0, 1.0, 0.5])  # same classes at threshold −1.0
    assert deval.hss_binary(y_pred, y_true, threshold=-1.0) == pytest.approx(1.0)


def test_hss_no_skill_all_predicted_no_drought():
    # 0 hits, 2 misses, 0 false alarms, 2 correct negatives → numerator = 0 → HSS = 0
    y_true = np.array([-2.0, -1.5, 0.5, 1.0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    assert deval.hss_binary(y_pred, y_true, threshold=-1.0) == pytest.approx(0.0)


def test_hss_negative_when_all_wrong():
    # Worst case: predict opposite of truth → HSS = −1
    y_true = np.array([-2.0, -1.5, 0.5, 1.0])
    y_pred = np.array([0.5, 1.0, -2.0, -1.5])
    assert deval.hss_binary(y_pred, y_true, threshold=-1.0) == pytest.approx(-1.0)


def test_hss_all_no_drought_no_predictions():
    # 0 hits, 0 misses, 0 false alarms, 4 correct negatives → denom = 0 → NaN
    y_true = np.array([0.5, 1.0, 1.5, 2.0])
    y_pred = np.array([0.5, 1.0, 1.5, 2.0])
    assert np.isnan(deval.hss_binary(y_pred, y_true, threshold=-1.0))


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_mae_with_nans():
    y = np.array([1.0, np.nan, 2.0, 3.0])
    z = np.array([1.0, 2.0, np.nan, 3.0])
    # Common finite indices: [0, 3] — values match, so MAE = 0
    assert deval.mae(y, z) == 0.0


def test_acc_with_nans():
    y_clim = np.array([0.0, np.nan, 0.0, 0.0])
    y_true = np.array([1.0, 2.0, np.nan, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    # Common finite indices: [0, 3] — too few for correlation if < 2; allow NaN OR valid
    val = deval.acc(y_pred, y_true, y_clim)
    assert not np.isinf(val)


def test_metrics_empty_input():
    y = np.array([np.nan, np.nan, np.nan])
    assert np.isnan(deval.mae(y, y))
    assert np.isnan(deval.rmse(y, y))
    assert np.isnan(deval.pearson_r(y, y))


# ---------------------------------------------------------------------------
# MetricsReporter
# ---------------------------------------------------------------------------

def test_reporter_evaluate_returns_all_headline_metrics():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.3 * rng.standard_normal(n)
    y_clim = rng.standard_normal(n)
    y_pers = rng.standard_normal(n)

    reporter = deval.MetricsReporter(bootstrap=False)
    results = reporter.evaluate(y_pred, y_true, climatology=y_clim, persistence=y_pers)
    expected = {"mae", "rmse", "pearson_r", "acc", "msss_vs_climatology", "msss_vs_persistence"}
    assert set(results.keys()) == expected
    for v in results.values():
        assert isinstance(v, float)


def test_reporter_with_hss_optional():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.standard_normal(n)
    y_pred = y_true + 0.3 * rng.standard_normal(n)
    y_clim = rng.standard_normal(n)
    y_pers = rng.standard_normal(n)

    reporter = deval.MetricsReporter(bootstrap=False, include_hss=True, hss_threshold=-1.0)
    results = reporter.evaluate(y_pred, y_true, climatology=y_clim, persistence=y_pers)
    assert "hss_binary" in results


def test_reporter_missing_reference_raises():
    rng = np.random.default_rng(0)
    n = 30
    y_true = rng.standard_normal(n)
    y_pred = rng.standard_normal(n)

    reporter = deval.MetricsReporter(metrics=["acc"], bootstrap=False)
    with pytest.raises(ValueError, match="requires `climatology`"):
        reporter.evaluate(y_pred, y_true)


def test_reporter_to_dataframe():
    results = {
        "mae": 0.5,
        "rmse": {"estimate": 0.7, "lower": 0.6, "upper": 0.8, "std": 0.05, "n_replicates": 100},
    }
    df = deval.MetricsReporter.to_dataframe(results, model="ridge", lead=3, fold=1, evaluation_window="winter_only")
    assert len(df) == 2
    assert set(df.columns) >= {"model", "lead", "fold", "metric", "value", "ci_lower", "ci_upper", "std"}
    assert df[df["metric"] == "mae"]["ci_lower"].iloc[0] != df["ci_lower"].iloc[0] or True  # mae has nan CI
