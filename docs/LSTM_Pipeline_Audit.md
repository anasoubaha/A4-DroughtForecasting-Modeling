# LSTM Pipeline Audit (Phase 12)

Audit of the SPEI3-forecasting LSTM pipeline: data flow, stage-by-stage
invariants, verification queries, and an incident log of the bugs found
between 2026-06-29 and 2026-06-30. Written after a multi-run failure whose
final root cause turned out to be **PyTorch's Metal (MPS) backend silently
producing NaN**, not any of the intermediate causes we chased first.

## 1. The bird's-eye view

```
                     ┌──────────────────────────────────────────────────────┐
                     │  configs/data.yaml    configs/features.yaml          │
                     │  configs/cv.yaml      configs/metrics.yaml           │
                     │  configs/models/lstm.yaml                            │
                     │  configs/experiments/exp_lstm.yaml                   │
                     └──────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ①  DATA ACQUISITION                                                    │
   │    droughtmodel/data.py::load_all(data_cfg)                            │
   │    Reads all NetCDF/Zarr inputs → dict[name, xr.Dataset]               │
   │    OUTPUT: 13 physical predictors + spei3 + morocco_mask, all monthly, │
   │            on the same (time, lat, lon) grid.                          │
   │    INVARIANT: no timestamp gaps; all vars share the master time index. │
   └───────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ②  FEATURE ASSEMBLY (per-lead)                                         │
   │    droughtmodel/features.py::build_dataset(..., lags={})               │
   │    Adds seasonal (sin/cos) + spatial (lat/lon) encodings;              │
   │    builds `target = SPEI3(t+L)` (shifted spei3).                       │
   │    LSTM-specific: NO PACF/CCF lag selection. lags={} means only        │
   │    contemporary predictors go in — the sliding window in step ⑤ is     │
   │    what gives the LSTM its history.                                    │
   │    OUTPUT: xr.Dataset with 17 features + target, dims (time, lat, lon) │
   │            for gridded; (time,) for climate indices; (lat, lon) for    │
   │            spatial encodings; sin/cos on (time,).                      │
   │    INVARIANT: `target` has NaN at the last L rows (shift boundary).    │
   └───────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ③  CV SPLIT + QUARANTINE                                               │
   │    droughtmodel/cv.py::RollingOriginCV.get_fold_indices(               │
   │        time_coord, fold, max_lag=T−1, lead=L)                          │
   │    5 rolling-origin folds, non-overlapping test windows 2000-2024.     │
   │    LSTM boundary gap = L + T + 1, implemented by passing               │
   │    max_lag=T−1 to the existing `L + max_lag + 2` machinery.            │
   │    OUTPUT: FoldIndices with (train_idx, val_idx, test_idx) as int      │
   │            positions and boundary_gap.                                 │
   │    INVARIANT: test_start_idx − train_end_idx ≥ L + T + 1.              │
   └───────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ④  FOLD-WISE STANDARDIZATION                                           │
   │    droughtmodel/cv.py::FoldStandardizer                                │
   │    Fit μ,σ on TRAIN (Morocco cells only), apply to all slices.         │
   │    Exceptions: enso, nao, mo, spei3, target (already standardised).    │
   │    LSTMExperimentRunner._standardize_full_using_train fits a SECOND    │
   │    FoldStandardizer on the same unstandardized train slice, applied    │
   │    to the FULL-TIMELINE dataset so backward-looking sequences in step  │
   │    ⑤ can read across slice boundaries without leaking targets.         │
   │    OUTPUT: xr.Datasets with the non-excepted features rescaled to      │
   │            μ≈0, σ≈1 (on Morocco cells).                                │
   │    INVARIANT: full_std[v].sel(morocco).std() ≈ 1 for every non-except v│
   │             — regression-tested in tests/test_lstm.py                  │
   │             (test_standardize_full_using_train_actually_rescales).     │
   └───────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ⑤  SLIDING-WINDOW RESHAPE                                              │
   │    droughtmodel/sequence.py::build_sequences(ds, T, ...)               │
   │    For each (lat, lon, t) center point:                                │
   │      X_sample = ds[t-T+1 : t+1, lat, lon, :]  → shape (T, F)           │
   │      y_sample = target[t, lat, lon]           → scalar                 │
   │    Static features (lat, lon, sin/cos, climate indices) are            │
   │    broadcast across the T timesteps and cells.                         │
   │    NaN samples are dropped.                                            │
   │    OUTPUT: (X, y, meta) with X shape (n_samples, T, F), y shape (n,).  │
   │    INVARIANT: `np.isfinite(X).all() and np.isfinite(y).all()`          │
   │             — the NaN-drop step guarantees this.                       │
   └───────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ⑥  TRAINING                                                            │
   │    droughtmodel/models/lstm.py::LSTMModel.fit_tensors                  │
   │    ┌─────────────────────────────────────────────────────────┐        │
   │    │ VariationalLSTM(input_size=F, hidden_size=H, dropout=p) │        │
   │    │   nn.LSTMCell(F, H) unrolled T timesteps in Python;     │        │
   │    │   input mask + hidden mask, one draw per SAMPLE,        │        │
   │    │   reused across all T timesteps (Gal & Ghahramani);     │        │
   │    │   nn.Linear(H, 1) head on final hidden state.           │        │
   │    └─────────────────────────────────────────────────────────┘        │
   │    Loss: weighted_mse (w = 1 + α·1{|y|>1}) or plain mse.               │
   │    Optim: Adam(lr, weight_decay).                                      │
   │    Safety: gradient-norm clip (max_norm=grad_clip_norm), NaN-loss      │
   │            abort. Early stopping on val loss (patience=10).            │
   │    DEVICE: hard-coded CPU. See §7 incident #4 — MPS silently produces  │
   │    NaN on the first batch of our LSTMCell loop.                        │
   │    OUTPUT: trained model + fit_state_ (train/val loss history).        │
   └───────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ⑦  HP TUNING — REPRESENTATIVE (FOLD-1) GRID SEARCH                     │
   │    droughtmodel/lstm_pipeline.py::_fold1_grid_search                   │
   │    16-24 combos over hidden_units × dropout × sequence_length × lr.    │
   │    Uses fold 1's (train, val) with gap sized by max_sequence_length    │
   │    so every combo sees the same window. Best combo (min val MSE) is    │
   │    LOCKED for folds 2-5. Log: results/lstm/logs/lstm_grid_search_L*.csv│
   └───────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │ ⑧  PER-FOLD REFIT + PREDICT                                            │
   │    For each fold: fit LSTM with locked HPs on `refit` (train ∪ val)    │
   │    with the fold's OWN gap; predict on `test`. Baselines (climatology, │
   │    persistence) fit on refit_full (unfiltered) for v3-equivalent MSSS  │
   │    reference.                                                          │
   │    OUTPUT: results/lstm/predictions/pooled_allLeads.nc                 │
   │            results/lstm/metrics/metrics_allLeads.csv                   │
   │            results/lstm/logs/fold_runs.csv                             │
   │            results/lstm/models/lstm_lead{L}_fold{F}.joblib             │
   └───────────────────────────────────────────────────────────────────────┘
```

## 2. Verification queries — how to spot-check each stage

Copy-pasteable one-liners to sanity-check any stage after a run. All use the
conda env `droughtforecast`.

### ② Feature dataset (17 features)

```python
from droughtmodel.lstm_pipeline import LSTMExperimentRunner
r = LSTMExperimentRunner('configs/experiments/exp_lstm.yaml', verbose=False)
r.setup_data()
ds = r._build_full_feature_dataset(3)
print(sorted(ds.data_vars))
# Expect: 13 physical + spei3 + sin_month + cos_month + lat_feat + lon_feat + target
```

### ③ Boundary gap

```python
from droughtmodel.sequence import lstm_boundary_gap
for L in (1, 3, 6):
    for T in (6, 12, 24):
        print(f'L={L}, T={T}: gap={lstm_boundary_gap(L, T)} months')
# Expect: 8, 14, 26 / 10, 16, 28 / 13, 19, 31
```

### ④ Standardization actually rescales

```python
prep = r._prepare_fold(3, r.cv.fold_specs[0], sequence_length=24)
full = r._standardize_full_using_train(r._build_full_feature_dataset(3), prep.train_unstd)
import numpy as np
mask = r._morocco_mask.values
for v in ('precip', 'tmax', 'rzsm', 'vpd', 'spei3'):
    a = full[v].values
    a = a[:, mask] if a.ndim == 3 else (a[mask] if a.ndim == 2 else a)
    a = a[np.isfinite(a)]
    print(f'{v:>8s}  std={a.std():.3f}  max={a.max():+.2f}  min={a.min():+.2f}')
# Expect: std≈1 for precip/tmax/rzsm/vpd; std≈1 for spei3 (already standardized upstream)
# Regression: if any std > 5 or max > 20, the standardizer was fit on the wrong slice.
```

### ⑤ Sliding-window sanity

```python
X_tr, y_tr, meta = r._build_window_tensors(full, prep.train, sequence_length=12, winter_filter=False)
print(f'X shape {X_tr.shape}  y shape {y_tr.shape}')
print(f'X finite? {np.isfinite(X_tr).all()}  y finite? {np.isfinite(y_tr).all()}')
print(f'X std={X_tr.std():.3f}  |X|_max={abs(X_tr).max():.2f}')
# Expect: shape (~80_000, 12, 17), all finite, std ≈ 1, |X| < 15 (outliers > 15 are a red flag)
```

### ⑥ First-batch loss sanity (the smoking gun for MPS incident)

```python
import torch, math
from droughtmodel.models.lstm import VariationalLSTM, weighted_mse_loss
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)
m = VariationalLSTM(input_size=X_tr.shape[-1], hidden_size=64, dropout=0.2)
m.train()
loader = DataLoader(TensorDataset(torch.as_tensor(X_tr), torch.as_tensor(y_tr)),
                    batch_size=512, shuffle=True)
xb, yb = next(iter(loader))
with torch.no_grad():
    pred = m(xb)
loss = weighted_mse_loss(pred, yb)
print(f'first-batch loss = {float(loss):.4f}  finite={math.isfinite(float(loss))}')
# Expect: finite loss in the 2-5 range. If NaN/inf, you are on MPS or standardization broke.
```

### ⑧ Predictions health

```python
import xarray as xr
ds = xr.open_dataset('results/lstm/predictions/pooled_allLeads.nc')
for L in ds['lead'].values:
    arr = ds['pred_lstm'].sel(lead=L).values
    finite = np.isfinite(arr).sum() / arr.size
    print(f'L={L}: finite fraction = {finite:.4f}')
# Expect: ~0.008-0.02 (only Morocco cells × months with usable sequences).
# If any lead reads 0.000, the LSTM diverged for that lead.
```

## 3. Config surface (what actually gets set where)

| Setting | File | Line-of-truth |
|---|---|---|
| Leads to sweep | `configs/experiments/exp_lstm.yaml` | `leads: [1, 3, 6]` |
| Grid axes | same | `lstm.grid` (currently 24 combos) |
| Fold-1 tuning flag | same | `lstm.representative_tuning_fold: 1` |
| Max T for gap sizing | same | `lstm.max_sequence_length: 24` |
| Winter-only training filter | same | `winter_only_training: false` |
| Drop seasonal encoding | same | `feature_overrides.drop_seasonal_encoding` |
| Save models | same | `save_models: true` |
| Loss & weighted-MSE knobs | `configs/models/lstm.yaml` | `params.loss / weighted_mse_*` |
| Regularization | same | `params.dropout / weight_decay / grad_clip_norm` |
| Optim ceiling | same | `params.max_epochs / patience / batch_size / learning_rate` |
| Device pin | same | `params.device: cpu` (see §7 incident #4) |
| Fold windows (2000-2024) | `configs/cv.yaml` | `folds` list |
| Standardization exceptions | same | `standardization_exceptions` |
| Contemporary predictors | `configs/features.yaml` | `contemporary_predictors` |
| Region mask (Morocco) | same | `region_mask.path` |

Anything set via a CLI arg or an environment variable is a bug — trace it
back to one of these YAMLs.

## 4. Test coverage

Unit tests in `tests/test_lstm.py`:

- `test_lstm_boundary_gap_formula` — `gap = L + T + 1` for all (L, T)
- `test_build_sequences_default_indices_and_shape` — output shapes
- `test_build_sequences_target_excluded_from_features` — no target leakage
- `test_build_sequences_static_features_broadcast_across_T` — lat/lon repeat across T
- `test_build_sequences_time_only_features_tile_across_cells` — ENSO/NAO/MO tile
- `test_build_sequences_drops_nan_samples_when_target_is_nan` — target-shift NaNs dropped
- `test_build_sequences_cell_mask_restricts_sample_centers` — Morocco mask honoured
- `test_predict_to_grid_scatters_predictions_to_template_shape` — reshape roundtrip
- `test_standardize_full_using_train_actually_rescales` — the ② incident regression test
- `test_weighted_mse_loss_weights_extremes_more_heavily` — loss weighting works
- `test_lstm_model_fits_and_predicts_smoke` — end-to-end fit/predict on synthetic data
- `test_variational_lstm_deterministic_under_seed` — seeding is honoured

Missing / would-be-nice:
- A GPU/MPS smoke test that would have caught incident #4 in CI.
- A "full-timeline standardized max value" assertion in the pipeline
  itself (raise if any Morocco-cell |x| > 20 after standardization).

## 5. Where to look when things go wrong

| Symptom | Most-likely cause | Where to look |
|---|---|---|
| `val_loss = inf`, `epochs_run = 1` at every combo | Device is MPS (or standardization broken) | `configs/models/lstm.yaml::params.device`; §2 stage ④/⑥ queries |
| Training descends but val turns around at epoch 2 | LR too high | Lower `learning_rate` to 1e-4 |
| Train loss plateaus at epoch 1 | Same as above OR model too small | Bigger hidden_units, longer T |
| `fit_duration_s < 1` for every combo | NaN abort firing → device or standardization | §2 stage ⑥ query |
| `total finite-cells = 0` for a lead | Model diverged at that lead → NaN weights | Load `.joblib`, check `fit_state_.train_loss_history` |
| Metrics CSV has real numbers for LSTM at some leads and NaN at others | Same as above — check per-lead predictions | §2 stage ⑧ query |
| `n_train_samples < n_val_samples` | Wrong slice sizes | `_train_window_end_indices` — likely gap too big |
| Predictions look identical to climatology | Model saturated (raw-scale inputs → tanh saturation) | §2 stage ④ query |

## 6. End-to-end command reference

```bash
# 1. Sanity-check the config
/opt/anaconda3/envs/droughtforecast/bin/python -m pytest tests/test_lstm.py -v

# 2. Run the sweep (~40 min wall time at CPU)
PYTHONUNBUFFERED=1 nohup /opt/anaconda3/envs/droughtforecast/bin/python \
    scripts/07_run_lstm.py --config configs/experiments/exp_lstm.yaml \
    > lstm_run_v6.log 2>&1 &
tail -f lstm_run_v6.log

# 3. Post-hoc: backfill train_rmse for the generalization diagnostic
/opt/anaconda3/envs/droughtforecast/bin/python scripts/08_compute_lstm_train_rmse.py \
    --exp-config configs/experiments/exp_lstm.yaml

# 4. Re-execute the comparison notebook
/opt/anaconda3/envs/droughtforecast/bin/jupyter nbconvert --execute --to notebook --inplace \
    notebooks/12_lstm_vs_default.ipynb
```

## 7. Incident log (2026-06-29 → 2026-06-30)

Chronological, so the reasoning trail is preserved.

### Incident #1 — Nested `VariationalLSTM` unpicklable (2026-06-29)

**Symptom:** All 15 saved `.joblib` model files were exactly 10 bytes.
`joblib.load(...)` raised `EOFError`.

**Cause:** `VariationalLSTM` was defined inside a factory function
`_make_variational_lstm()`. Pickle cannot serialize nested classes — it looks
up the class by dotted import path at unpickling time and can't find the
`.<locals>.` version.

**Fix:** Hoisted `VariationalLSTM` to module scope in
`droughtmodel/models/lstm.py`, guarded by `_HAS_TORCH`. Also hardened
`_save_lstm_model` to delete partial files on error and treat `<64 bytes` as
corrupt.

**Test:** Round-trip pickle test in the diagnostic run — 1840 bytes vs the
previous 10.

### Incident #2 — Winter-only-training filter emptying LSTM training data (2026-06-30 morning)

**Symptom:** LSTM predictions at L=3 and L=6 were entirely NaN after v3 run.

**Interpretation at the time:** We thought it was training divergence due to
exploding gradients, and added `grad_clip_norm=1.0` + NaN-loss abort.

**Actual cause:** Two compounding issues. The abort was firing on the first
batch because MPS (incident #4) was producing NaN. The clip and abort were
correct fixes — they just weren't the root cause.

### Incident #3 — Standardizer fit on already-standardized data (2026-06-30 afternoon)

**Symptom:** Diagnostic dump showed the "standardized" full-timeline dataset
had `precip max = +458.1`, `precip std = 35` — features in raw physical units.

**Cause:** `LSTMExperimentRunner._standardize_full_using_train(full_ds, prep.train)`
received `prep.train` — a slice that had ALREADY been passed through
`FoldStandardizer` inside `_prepare_fold`. Fitting a fresh standardizer on
data with `mu≈0, sigma≈1` produced an identity transform, which was then
applied to the raw `full_ds`.

**Fix:** Added a `train_unstd` field to `_PreparedLSTMFold` that carries the
raw pre-standardized slice through. Both callers of
`_standardize_full_using_train` and `scripts/08_compute_lstm_train_rmse.py`
now pass `prep.train_unstd`.

**Test:** New regression test
`test_standardize_full_using_train_actually_rescales` in
`tests/test_lstm.py`.

**Why this mattered:** With raw-scale inputs (precip up to 458 mm), the LSTM's
internal tanh/sigmoid activations saturated. Gradients were near-zero and the
model made no real progress; val loss plateaued at "predict climatology" from
epoch 1. This masked incident #4 for two runs — training was silently
useless on CPU too, just not obviously broken.

### Incident #4 — MPS silently produces NaN on `LSTMCell` loops (2026-06-30 evening)

**Symptom:** Even after fixing standardization and adding grad clipping, every
grid-search combo aborted at epoch 1 with `val_loss=inf`,
`fit_duration_s ≈ 20 ms`. This looked identical to incident #2 but was
happening for a different reason.

**Cause:** `device: auto` in the LSTM YAML resolved to `mps` on the user's
macOS machine. PyTorch's Metal backend has known numeric-stability bugs with
manually-unrolled `LSTMCell` loops that use in-place operations for
variational dropout masks. CPU produced finite losses that descended
normally on the same data / seed / architecture; MPS produced `NaN` on the
very first batch.

**Fix:** Hard-coded `device: cpu` in `configs/models/lstm.yaml`. Also updated
`LSTMModel._resolve_device`: `"auto"` no longer picks up MPS — it falls
through to CPU. To use MPS you must set `device: mps` EXPLICITLY, accepting
the risk.

**Why every prior "successful" run was actually broken:** The v2 run's
`RMSE ≈ 0.99` at L=1 that I once told the user "worked" was the LSTM
producing near-zero saturated outputs on MPS. Because the target has std ≈ 1
and near-zero predictions have MSE ≈ 1, that "worked" number was
indistinguishable from a broken model outputting garbage. Every LSTM output
prior to 2026-06-30 should be treated as invalid.

**Long-term guard:** The forward pass now runs on CPU by default. The
diagnostic queries in §2 will catch it if someone flips it back and things
go wrong. Should be revisited when PyTorch's MPS backend stabilizes for
custom-loop LSTMs (tracked at
https://github.com/pytorch/pytorch/issues?q=is%3Aissue+mps+lstm ).

## 8. Trust budget for the LSTM results

- Anything in `results/lstm/` created BEFORE 2026-06-30 19:00 → **discard**.
- Anything in `results/lstm/logs/lstm_grid_search_L*.csv` with `val_loss=inf`
  everywhere → **the run hit MPS**, discard.
- A run with the current CPU-pinned config that produces `val_loss` values in
  the 0.9-1.3 range on L=1 and 1.0-1.3 on L=3/6 → real training happened.
  Trust the numbers.
