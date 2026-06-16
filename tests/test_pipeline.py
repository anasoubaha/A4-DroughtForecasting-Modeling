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
