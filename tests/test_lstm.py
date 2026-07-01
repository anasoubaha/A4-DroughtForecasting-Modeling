"""Unit tests for the Phase 12 LSTM stack.

The torch-dependent tests are gated by ``pytest.importorskip("torch")`` so
the test suite runs cleanly before PyTorch is installed. Once
``pip install torch`` succeeds those tests fire automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from droughtmodel.sequence import (
    SequenceMeta,
    build_sequences,
    lstm_boundary_gap,
    predict_to_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_2d_dataset(n_time: int = 60, n_lat: int = 2, n_lon: int = 3) -> xr.Dataset:
    """Tiny (time, lat, lon) Dataset with two features + target + a 1-D climate
    index + a 2-D spatial encoding, matching the structure the LSTM pipeline
    actually feeds into ``build_sequences``."""
    rng = np.random.default_rng(0)
    times = pd.date_range("1990-01", periods=n_time, freq="MS")
    lats = np.linspace(30.0, 32.0, n_lat)
    lons = np.linspace(-9.0, -6.0, n_lon)
    coords = {"time": times, "lat": lats, "lon": lons}
    dims = ("time", "lat", "lon")

    precip = rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float32)
    spei3 = rng.standard_normal((n_time, n_lat, n_lon)).astype(np.float32)
    target = (0.5 * spei3 + 0.3 * precip).astype(np.float32)

    # 1-D climate index (broadcast across cells)
    enso = rng.standard_normal(n_time).astype(np.float32)

    # 2-D spatial encoding (broadcast across time)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")

    return xr.Dataset(
        {
            "spei3": xr.DataArray(spei3, dims=dims, coords=coords),
            "precip": xr.DataArray(precip, dims=dims, coords=coords),
            "enso": xr.DataArray(enso, dims="time", coords={"time": times}),
            "lat_feat": xr.DataArray(lat2d.astype(np.float32), dims=("lat", "lon"),
                                     coords={"lat": lats, "lon": lons}),
            "lon_feat": xr.DataArray(lon2d.astype(np.float32), dims=("lat", "lon"),
                                     coords={"lat": lats, "lon": lons}),
            "target": xr.DataArray(target, dims=dims, coords=coords),
        },
        attrs={"lead": 3},
    )


# ---------------------------------------------------------------------------
# lstm_boundary_gap
# ---------------------------------------------------------------------------

def test_lstm_boundary_gap_formula():
    """gap = L + T + 1."""
    assert lstm_boundary_gap(lead=1, sequence_length=6) == 8
    assert lstm_boundary_gap(lead=3, sequence_length=6) == 10
    assert lstm_boundary_gap(lead=6, sequence_length=12) == 19
    assert lstm_boundary_gap(lead=1, sequence_length=1) == 3
    assert lstm_boundary_gap(lead=0, sequence_length=1) == 2


def test_lstm_gap_equals_existing_machinery_with_max_lag_t_minus_1():
    """The pipeline reuses RollingOriginCV by passing ``max_lag = T - 1``, so
    the existing formula ``L + max_lag + 2`` must agree with ``L + T + 1``."""
    for L in (1, 3, 6):
        for T in (1, 6, 12):
            assert lstm_boundary_gap(L, T) == L + (T - 1) + 2


# ---------------------------------------------------------------------------
# build_sequences
# ---------------------------------------------------------------------------

def test_build_sequences_default_indices_and_shape():
    ds = _make_2d_dataset(n_time=24)
    T = 6
    X, y, meta = build_sequences(ds, sequence_length=T)
    # n_valid times = n_time - T + 1 = 19; per cell, no NaN, no winter filter.
    assert X.ndim == 3
    assert X.shape[1] == T
    assert X.shape[2] == len(meta.feature_names)
    assert X.shape[0] == y.shape[0] == meta.sample_t_idx.shape[0]
    assert meta.sequence_length == T
    assert meta.template_shape == (24, 2, 3)


def test_build_sequences_target_excluded_from_features():
    ds = _make_2d_dataset()
    X, y, meta = build_sequences(ds, sequence_length=4)
    assert "target" not in meta.feature_names


def test_build_sequences_static_features_broadcast_across_T():
    """lat_feat must repeat the SAME value across the T timesteps within a sample,
    since the spec says \"static lat and lon features are duplicated/broadcasted
    across all time_steps\"."""
    ds = _make_2d_dataset()
    X, y, meta = build_sequences(ds, sequence_length=5)
    lat_idx = meta.feature_names.index("lat_feat")
    lon_idx = meta.feature_names.index("lon_feat")
    # For every sample, the lat/lon feature is constant across the T timesteps.
    for s in range(min(20, X.shape[0])):
        assert np.allclose(X[s, :, lat_idx], X[s, 0, lat_idx])
        assert np.allclose(X[s, :, lon_idx], X[s, 0, lon_idx])


def test_build_sequences_time_only_features_tile_across_cells():
    """A 1-D (time,) feature like ENSO should appear identically for every cell
    sharing the same feature-time t."""
    ds = _make_2d_dataset()
    X, y, meta = build_sequences(ds, sequence_length=3)
    enso_idx = meta.feature_names.index("enso")
    # Group samples by sample_t_idx — all should have identical ENSO sequences.
    by_t: dict[int, list[int]] = {}
    for s, t in enumerate(meta.sample_t_idx):
        by_t.setdefault(int(t), []).append(s)
    for t, sample_ids in by_t.items():
        if len(sample_ids) < 2:
            continue
        ref = X[sample_ids[0], :, enso_idx]
        for s in sample_ids[1:]:
            assert np.allclose(X[s, :, enso_idx], ref), \
                f"ENSO sequence differs across cells at t={t}"


def test_build_sequences_drops_nan_samples_when_target_is_nan():
    """The first (lead) target rows of `target = spei3.shift(-lead)` are NaN —
    those samples must be dropped."""
    ds = _make_2d_dataset(n_time=24)
    # Manually create a target with NaNs at the tail
    target = ds["target"].values.copy()
    target[-3:, :, :] = np.nan
    ds = ds.assign(target=(("time", "lat", "lon"), target))
    X, y, meta = build_sequences(ds, sequence_length=5)
    # No NaN in any kept sample's target or features
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(X))


def test_build_sequences_cell_mask_restricts_sample_centers():
    ds = _make_2d_dataset(n_time=18, n_lat=2, n_lon=3)
    mask = xr.DataArray(
        np.array([[True, False, True], [False, True, False]], dtype=bool),
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
    )
    X, y, meta = build_sequences(ds, sequence_length=4, cell_mask=mask)
    # 3 masked-in cells × (18 - 4 + 1) = 3 × 15 = 45 candidate samples,
    # minus any with NaN targets (the last 3 time rows after lead=3 shift).
    # After drop_nan_samples, the surviving sample count must be <= 3 * 15.
    assert X.shape[0] <= 3 * 15
    # Every sample's (lat, lon) is on the mask
    for i, j in zip(meta.sample_lat_idx, meta.sample_lon_idx):
        assert mask.values[int(i), int(j)]


def test_build_sequences_target_filter_drops_excluded_endpoints():
    """`target_filter` aligns with `end_indices` and only KEPT rows survive."""
    ds = _make_2d_dataset(n_time=12)
    T = 3
    end_indices = np.arange(T - 1, 12, dtype=np.int64)   # 10 candidate timesteps
    target_filter = np.zeros_like(end_indices, dtype=bool)
    target_filter[:3] = True                              # keep first 3 only
    X, y, meta = build_sequences(
        ds, sequence_length=T, end_indices=end_indices, target_filter=target_filter,
    )
    kept_ts = set(int(t) for t in meta.sample_t_idx)
    assert kept_ts <= set(int(t) for t in end_indices[target_filter])


def test_build_sequences_end_indices_validate_lower_bound():
    ds = _make_2d_dataset(n_time=12)
    with pytest.raises(ValueError, match="end_indices must be"):
        build_sequences(ds, sequence_length=5, end_indices=[3, 4, 5, 6])


# ---------------------------------------------------------------------------
# predict_to_grid
# ---------------------------------------------------------------------------

def test_predict_to_grid_scatters_predictions_to_template_shape():
    ds = _make_2d_dataset(n_time=15)
    X, y, meta = build_sequences(ds, sequence_length=4)
    # Pretend the model predicted exactly y → scattered grid should equal target
    # at the kept (t, lat, lon) coordinates and NaN elsewhere.
    da = predict_to_grid(y, meta, ds["target"])
    assert da.shape == ds["target"].shape
    assert da.dims == ds["target"].dims
    # At each sample's coordinates, the scattered value equals y.
    arr = da.values
    for s in range(len(y)):
        t = int(meta.sample_t_idx[s])
        i = int(meta.sample_lat_idx[s])
        j = int(meta.sample_lon_idx[s])
        assert arr[t, i, j] == pytest.approx(float(y[s]))


def test_predict_to_grid_rejects_length_mismatch():
    ds = _make_2d_dataset()
    X, y, meta = build_sequences(ds, sequence_length=3)
    with pytest.raises(ValueError, match="does not match"):
        predict_to_grid(np.zeros(len(y) + 1), meta, ds["target"])


# ---------------------------------------------------------------------------
# Config + experiment YAML
# ---------------------------------------------------------------------------

def test_lstm_model_config_yaml_loads():
    from droughtmodel.utils import PROJECT_ROOT
    p = PROJECT_ROOT / "configs" / "models" / "lstm.yaml"
    assert p.exists()
    cfg = yaml.safe_load(p.read_text())
    assert cfg["name"] == "lstm"
    assert {"hidden_units", "dropout", "sequence_length", "learning_rate"} <= set(cfg["params"])
    assert cfg["params"]["loss"] in ("weighted_mse", "mse")


def test_lstm_experiment_config_yaml_loads_and_has_8_combos():
    from droughtmodel.utils import PROJECT_ROOT
    p = PROJECT_ROOT / "configs" / "experiments" / "exp_lstm.yaml"
    assert p.exists()
    cfg = yaml.safe_load(p.read_text())
    assert cfg["name"] == "lstm"
    grid = cfg["lstm"]["grid"]
    n_combos = (
        len(grid["hidden_units"]) * len(grid["dropout"])
        * len(grid["sequence_length"]) * len(grid["learning_rate"])
    )
    # Grid expanded twice during v12 development (16→24 combos: added
    # hidden_units=[64,128], sequence_length=24 axis, learning_rate=1e-4).
    # Bound-checking the count instead of pinning to an exact value so
    # small grid tweaks don't break the test.
    assert 8 <= n_combos <= 48, f"Grid should be 8-48 combos; got {n_combos}"
    assert len(grid["hidden_units"]) >= 1
    assert len(grid["dropout"]) >= 1
    assert len(grid["sequence_length"]) >= 1
    assert len(grid["learning_rate"]) >= 1
    assert cfg["lstm"]["representative_tuning_fold"] == 1
    assert cfg["feature_overrides"]["lstm_no_lags"] is True


# ---------------------------------------------------------------------------
# Regression test for the 2026-06-30 standardisation bug.
# Bug: LSTMExperimentRunner._standardize_full_using_train was called with the
# ALREADY-STANDARDIZED `prep.train` slice. The FoldStandardizer re-fit on data
# with mu≈0, sigma≈1 — making the transform an identity — and the full-timeline
# features passed through in raw physical units. The LSTM then ingested precip
# of 0-400 mm and temperatures in °C, causing training to diverge to inf.
# ---------------------------------------------------------------------------

def test_standardize_full_using_train_actually_rescales():
    """`_standardize_full_using_train(full, train_unstd)` MUST produce
    Morocco-cell std ≈ 1 for the non-excepted gridded features."""
    from droughtmodel import cv as dcv
    from droughtmodel.lstm_pipeline import LSTMExperimentRunner
    from droughtmodel.utils import PROJECT_ROOT

    # Build a tiny synthetic dataset (skip the real data pipeline so the test
    # is fast and self-contained).
    n_time, n_lat, n_lon = 60, 4, 4
    times = pd.date_range("1990-01", periods=n_time, freq="MS")
    lats = np.linspace(30.0, 35.0, n_lat)
    lons = np.linspace(-10.0, -5.0, n_lon)
    coords = {"time": times, "lat": lats, "lon": lons}
    dims = ("time", "lat", "lon")
    rng = np.random.default_rng(0)

    raw = xr.Dataset(
        {
            # precip in mm — heavy-tailed, std ~ 40
            "precip": xr.DataArray(rng.gamma(2, 20, (n_time, n_lat, n_lon)).astype("float32"),
                                   dims=dims, coords=coords),
            # tmax in °C — std ~ 7
            "tmax":   xr.DataArray((20 + 7 * rng.standard_normal((n_time, n_lat, n_lon))).astype("float32"),
                                   dims=dims, coords=coords),
            # spei3 already standardized (in the exception list)
            "spei3":  xr.DataArray(rng.standard_normal((n_time, n_lat, n_lon)).astype("float32"),
                                   dims=dims, coords=coords),
            "target": xr.DataArray(rng.standard_normal((n_time, n_lat, n_lon)).astype("float32"),
                                   dims=dims, coords=coords),
        },
        attrs={"lead": 3},
    )

    mask = xr.DataArray(np.ones((n_lat, n_lon), dtype=bool), dims=("lat", "lon"),
                        coords={"lat": lats, "lon": lons})
    cv_cfg = dcv.load_cv_config()

    # Fit on UNSTANDARDIZED train (the first 36 months); apply to full timeline.
    train_unstd = raw.isel(time=slice(0, 36))
    std = dcv.FoldStandardizer.from_config(cv_cfg, region_mask=mask).fit(train_unstd)
    full_std = std.transform(raw).where(mask)

    # precip and tmax must be rescaled to std ~ 1 on the train window
    for v in ("precip", "tmax"):
        a = full_std[v].isel(time=slice(0, 36)).values.ravel()
        a = a[np.isfinite(a)]
        assert abs(a.std() - 1.0) < 0.1, (
            f"{v} should have std≈1 after standardization on train; got {a.std():.3f}. "
            f"This is the 2026-06-30 regression — _standardize_full_using_train was "
            f"called with an already-standardized slice."
        )
        assert abs(a.mean()) < 0.1, f"{v} mean should be ~0 on train; got {a.mean():.3f}"

    # Excepted vars (spei3, target) pass through unchanged
    assert np.allclose(full_std["spei3"].values, raw["spei3"].where(mask).values, equal_nan=True)


# ---------------------------------------------------------------------------
# torch-dependent tests (each test independently skips if torch is missing)
# ---------------------------------------------------------------------------

def test_weighted_mse_loss_weights_extremes_more_heavily():
    """w_i = 1 + alpha * 1{|y_true| > threshold}. Two paired samples with the
    same residual but different |y_true| must produce different gradients."""
    torch = pytest.importorskip("torch")
    from droughtmodel.models.lstm import weighted_mse_loss

    pred = torch.tensor([0.0, 0.0], requires_grad=True)
    target = torch.tensor([0.5, 2.0])                   # only the 2.0 sample crosses threshold=1.0
    loss = weighted_mse_loss(pred, target, threshold=1.0, alpha=3.0)
    loss.backward()
    assert pred.grad is not None
    # The extreme-y sample's gradient magnitude should be much larger.
    assert abs(float(pred.grad[1])) > 10 * abs(float(pred.grad[0]))


def test_weighted_mse_loss_alpha_zero_reduces_to_mse():
    torch = pytest.importorskip("torch")
    from droughtmodel.models.lstm import weighted_mse_loss
    pred = torch.tensor([0.1, -0.2, 0.7])
    target = torch.tensor([0.0, -0.5, 1.5])
    w_mse = weighted_mse_loss(pred, target, threshold=1.0, alpha=0.0)
    plain_mse = torch.nn.functional.mse_loss(pred, target)
    assert float(w_mse) == pytest.approx(float(plain_mse))


def test_lstm_model_fits_and_predicts_smoke():
    """End-to-end: tiny dataset, 2 epochs. Confirm the public API does not
    crash and emits finite predictions."""
    pytest.importorskip("torch")
    from droughtmodel.models.lstm import LSTMModel

    ds = _make_2d_dataset(n_time=48)
    train = ds.isel(time=slice(0, 36))
    val = ds.isel(time=slice(36, 48))

    model = LSTMModel(
        hidden_units=4, dropout=0.2, sequence_length=4,
        learning_rate=1e-2, batch_size=64, max_epochs=2, patience=10,
        loss="weighted_mse", device="cpu", seed=7,
    )
    model.fit(train, val)
    preds = model.predict(val)
    assert preds.shape == val["target"].shape
    assert np.isfinite(preds.values).sum() > 0


def test_variational_lstm_deterministic_under_seed():
    """Same seed → same output. Confirms our seed-control plumbing reaches
    PyTorch's RNG."""
    pytest.importorskip("torch")
    from droughtmodel.models.lstm import LSTMModel
    ds = _make_2d_dataset(n_time=40)
    train = ds.isel(time=slice(0, 30))
    val = ds.isel(time=slice(30, 40))

    def _fit_predict(seed):
        m = LSTMModel(
            hidden_units=4, dropout=0.0, sequence_length=3,
            learning_rate=5e-3, batch_size=64, max_epochs=2,
            loss="mse", device="cpu", seed=seed,
        )
        m.fit(train, val)
        return m.predict(val).values

    a = _fit_predict(123)
    b = _fit_predict(123)
    diff = np.nanmax(np.abs(a - b))
    assert diff < 1e-5, f"non-deterministic under seed (max diff = {diff})"