"""Unit tests for the pipeline orchestrator (v3 §6.1)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from droughtmodel.models.linear import LassoModel, RidgeModel
from droughtmodel.models.tree import RandomForestModel, XGBoostModel
from droughtmodel.pipeline import (
    ExperimentRunner,
    FeatureStatusLog,
    FoldRunLog,
    _ZERO_TOL,
    _feature_status_rows,
    _importance_kind,
    get_model_class,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(n_time: int = 120) -> xr.Dataset:
    rng = np.random.default_rng(0)
    times = pd.date_range("1950-01", periods=n_time, freq="MS")
    lats = np.linspace(28, 36, 3)
    lons = np.linspace(-13, -1, 3)
    coords = {"time": times, "lat": lats, "lon": lons}
    dims = ("time", "lat", "lon")

    spei3 = rng.standard_normal((n_time, 3, 3))
    precip = rng.standard_normal((n_time, 3, 3))
    target = 1.5 * spei3 + 0.5 * precip + 0.1 * rng.standard_normal((n_time, 3, 3))
    return xr.Dataset({
        "spei3": xr.DataArray(spei3, dims=dims, coords=coords),
        "precip": xr.DataArray(precip, dims=dims, coords=coords),
        "target": xr.DataArray(target, dims=dims, coords=coords),
    })


# ---------------------------------------------------------------------------
# _importance_kind
# ---------------------------------------------------------------------------

def test_importance_kind_linear_returns_coef():
    m = RidgeModel(alpha=1.0).fit(_make_dataset())
    assert _importance_kind(m) == "coef"


def test_importance_kind_rf_returns_gini():
    m = RandomForestModel(n_estimators=10, random_state=0).fit(_make_dataset())
    assert _importance_kind(m) == "gini"


def test_importance_kind_xgboost_returns_gain():
    m = XGBoostModel(n_estimators=10, early_stopping_rounds=None, random_state=0).fit(_make_dataset())
    assert _importance_kind(m) == "gain"


# ---------------------------------------------------------------------------
# _feature_status_rows
# ---------------------------------------------------------------------------

def test_feature_status_rows_linear_zeros_marked_dropped():
    """Lasso with strong alpha should produce some zero coefs → retained=False."""
    ds = _make_dataset(n_time=300)
    m = LassoModel(alpha=10.0, max_iter=5000).fit(ds)
    rows = _feature_status_rows(m, fold=1, lead=3, name="lasso")
    assert all(isinstance(r, FeatureStatusLog) for r in rows)
    assert {r.feature for r in rows} == {"precip", "spei3"}
    # At alpha=10 both coefs should be killed by L1
    assert all(not r.retained for r in rows)
    assert all(r.kind == "coef" for r in rows)


def test_feature_status_rows_linear_nonzero_retained():
    ds = _make_dataset(n_time=300)
    m = RidgeModel(alpha=0.1).fit(ds)
    rows = _feature_status_rows(m, fold=1, lead=3, name="ridge")
    # Ridge never zeroes coefs → all retained
    assert all(r.retained for r in rows)


def test_feature_status_rows_tree_always_retained():
    ds = _make_dataset(n_time=200)
    m = RandomForestModel(n_estimators=20, random_state=0).fit(ds)
    rows = _feature_status_rows(m, fold=1, lead=3, name="rf")
    assert all(r.retained for r in rows)
    assert all(r.kind == "gini" for r in rows)


def test_feature_status_rows_metadata_propagated():
    m = RidgeModel().fit(_make_dataset())
    rows = _feature_status_rows(m, fold=7, lead=6, name="ridge")
    assert all(r.fold == 7 for r in rows)
    assert all(r.lead == 6 for r in rows)
    assert all(r.model == "ridge" for r in rows)


def test_zero_tolerance_constant():
    assert _ZERO_TOL == pytest.approx(1e-10)


# ---------------------------------------------------------------------------
# get_model_class
# ---------------------------------------------------------------------------

def test_get_model_class_known():
    cls = get_model_class("ridge")
    assert cls is RidgeModel


def test_get_model_class_unknown_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_class("not_a_model")


# ---------------------------------------------------------------------------
# ExperimentRunner instantiation
# ---------------------------------------------------------------------------

def _min_exp_config(tmp_path: Path) -> dict:
    """Minimal valid experiment dict — paths under tmp_path."""
    return {
        "name": "test",
        "leads": [3],
        "modeling_unit": "global",
        "models": ["climatology", "persistence"],
        "hp_grids": {},
        "output": {
            "predictions_dir": str(tmp_path / "predictions"),
            "metrics_dir": str(tmp_path / "metrics"),
            "logs_dir": str(tmp_path / "logs"),
        },
    }


def test_experiment_runner_init_creates_output_dirs(tmp_path):
    cfg = _min_exp_config(tmp_path)
    runner = ExperimentRunner(cfg, verbose=False)
    assert runner.preds_dir.exists()
    assert runner.metrics_dir.exists()
    assert runner.logs_dir.exists()


def test_experiment_runner_loads_from_yaml(tmp_path):
    cfg_path = tmp_path / "exp.yaml"
    cfg = _min_exp_config(tmp_path)
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg))
    runner = ExperimentRunner(cfg_path, verbose=False)
    assert runner.exp["name"] == "test"
    assert runner.exp["leads"] == [3]


def test_experiment_runner_stores_subconfigs(tmp_path):
    cfg = _min_exp_config(tmp_path)
    runner = ExperimentRunner(cfg, verbose=False)
    # Standing configs auto-loaded from the project's configs/
    assert "folds" in runner.cv_cfg
    assert "headline_metrics" in runner.metrics_cfg


def test_fold_run_log_dataclass_fields():
    log = FoldRunLog(
        fold=1, lead=3, model="ridge",
        best_params=json.dumps({"alpha": 1.0}),
        best_val_score=-0.4, best_iteration=None,
        n_features_total=10, n_features_retained=10,
        K_eff=4, boundary_gap=9,
        search_duration_s=2.3, fit_duration_s=0.1,
        n_trials=13,
    )
    assert log.fold == 1
    assert json.loads(log.best_params) == {"alpha": 1.0}


def test_winter_target_mask_picks_correct_months():
    """At lead=3, target_month = (feature_time + 3 months).month. The mask
    must be True iff the target_month is in {11, 12, 1, 2}."""
    import xarray as xr
    from droughtmodel.pipeline import _winter_target_mask, _filter_to_winter_targets

    times = pd.date_range("2000-01", periods=36, freq="MS")
    ds = xr.Dataset(
        {"target": (("time",), np.arange(36, dtype=float))},
        coords={"time": times},
        attrs={"lead": 3},
    )
    mask = _winter_target_mask(ds)
    target_months = (pd.DatetimeIndex(ds["time"].values) + pd.DateOffset(months=3)).month
    expected = np.isin(target_months, [11, 12, 1, 2])
    assert np.array_equal(mask, expected)

    filtered = _filter_to_winter_targets(ds)
    assert filtered.sizes["time"] == 12   # 4 winter months × 3 years
    filtered_target_months = (
        pd.DatetimeIndex(filtered["time"].values) + pd.DateOffset(months=3)
    ).month
    assert set(filtered_target_months) <= {11, 12, 1, 2}


def test_winter_only_training_flag_default_false_and_loads_from_yaml(tmp_path):
    """`winter_only_training` defaults to False, can be overridden via experiment YAML."""
    import yaml
    cfg = _min_exp_config(tmp_path)
    r1 = ExperimentRunner(cfg, verbose=False)
    assert r1.winter_only_training is False

    cfg["winter_only_training"] = True
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    r2 = ExperimentRunner(cfg_path, verbose=False)
    assert r2.winter_only_training is True


def test_feature_overrides_default_empty_is_noop(tmp_path):
    """With no `feature_overrides` block, `_apply_lag_overrides` is identity
    and `feature_overrides` is an empty dict."""
    cfg = _min_exp_config(tmp_path)
    r = ExperimentRunner(cfg, verbose=False)
    assert r.feature_overrides == {}
    selected = {"enso": [3, 6, 9, 12], "nao": [1, 6], "spei3": [1, 2, 3]}
    assert r._apply_lag_overrides(selected) == selected


def test_feature_overrides_force_lags_replaces_lag_lists(tmp_path):
    """force_lags overrides a variable's lag list verbatim. Variables not in
    the override stay untouched."""
    cfg = _min_exp_config(tmp_path)
    cfg["feature_overrides"] = {"force_lags": {"enso": [1], "mo": []}}
    r = ExperimentRunner(cfg, verbose=False)
    selected = {"enso": [3, 6, 9, 12], "mo": [2, 4], "nao": [1, 6]}
    out = r._apply_lag_overrides(selected)
    assert out["enso"] == [1]      # forced to [1] regardless of PACF/CCF
    assert out["mo"] == []         # forced to empty (no lags for MO)
    assert out["nao"] == [1, 6]    # not in override → unchanged


def test_feature_overrides_drop_seasonal_encoding_flag(tmp_path):
    """`drop_seasonal_encoding: true` is stored and consumed by `_prepare_fold`
    to override the global `include_seasonal_encoding` from features.yaml."""
    cfg = _min_exp_config(tmp_path)
    cfg["feature_overrides"] = {"drop_seasonal_encoding": True}
    r = ExperimentRunner(cfg, verbose=False)
    assert r.feature_overrides.get("drop_seasonal_encoding") is True


def test_pruned_winter_experiment_yaml_loads():
    """The shipped pruned-winter experiment YAML must load cleanly with both
    the winter-only training filter and the v5 pruning overrides on."""
    from droughtmodel.utils import PROJECT_ROOT
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "exp_pruned-winter.yaml"
    assert cfg_path.exists()
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["name"] == "pruned-winter"
    assert cfg["winter_only_training"] is True
    assert cfg["feature_overrides"]["drop_seasonal_encoding"] is True
    assert cfg["feature_overrides"]["force_lags"] == {"enso": [1], "mo": [1]}
    # Output paths under results/pruned-winter/
    for key in ("predictions_dir", "metrics_dir", "logs_dir", "models_dir"):
        assert "pruned-winter" in cfg["output"][key]


def test_winter_training_experiment_yaml_loads():
    """The shipped winter-training experiment YAML must load cleanly."""
    from droughtmodel.utils import PROJECT_ROOT
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "exp_winter-training.yaml"
    assert cfg_path.exists()
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["name"] == "winter-training"
    assert cfg["winter_only_training"] is True
    # Output paths should be under results/winter-training/
    for key in ("predictions_dir", "metrics_dir", "logs_dir", "models_dir"):
        assert "winter-training" in cfg["output"][key]


def test_default_experiment_yaml_loads():
    """The shipped default experiment YAML must load cleanly."""
    from droughtmodel.utils import PROJECT_ROOT
    cfg_path = PROJECT_ROOT / "configs" / "experiments" / "exp_default.yaml"
    assert cfg_path.exists()
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["name"] == "default"
    assert set(cfg["models"]) >= {"climatology", "persistence", "ridge", "lasso",
                                   "elasticnet", "rf", "xgboost"}
    # Every model with an HP grid must be in REGISTRY
    from droughtmodel.models.registry import REGISTRY
    for name in cfg.get("hp_grids", {}):
        assert name in REGISTRY
