# SPEI3 Forecasting Scheme — v3

## 1. Inputs (Before Preprocessing)

| Input group | Variable | Temporal coverage | Spatial resolution | Storage | Source |
| --- | --- | --- | --- | --- | --- |
| Climatological | Precipitation (mm/mon) | 1901–2024 | 0.5° × 0.5° | NetCDF | CRU |
|  | Min, Max temperature (°C) | 1901–2024 | 0.5° × 0.5° | NetCDF | CRU |
|  | PET (mm/day) | 1901–2024 | 0.5° × 0.5° | NetCDF | CRU |
|  | Solar radiation (J·m⁻² / W·m⁻²) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
|  | Wind speed at 2 m (m·s⁻¹) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
|  | VPD (hPa) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
|  | TCWV (kg/m²) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
|  | **Rootzone Soil Moisture (RZSM, 0–100 cm, m³/m³)** | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 (derived from swvl1–swvl3) |
| Large-scale climate indices | NAO Index | 1950–2026 | — | Text | NOAA CPC |
|  | ENSO Niño 3.4 | 1870–2025 | — | NetCDF | NOAA PSL |
|  | Mediterranean Oscillation (MO) | 1940–2025 | — | NetCDF | ERA5-derived (Gibraltar − Lod SLP, standardized 1981–2010) |

**Potential additions** deferred to a future iteration if v1 underperforms:

- AMO (Atlantic Multidecadal Oscillation) — documented linkage to North African precipitation
- AO (Arctic Oscillation) — partial NAO redundancy, possible additive skill

## 2. Preprocessing

**a.** Align time period **1950–2024** for all input variables.

**b.** Upscale ERA5 → 0.5° via conservative area-weighted averaging. All gridded data: 64 × 64 at 0.5°.

**c.** Missing-value handling (explicit):

- Temporal interpolation up to 2 consecutive missing months
- Drop cells with > 5 % missing in the training period of any fold

**d. RZSM derivation:**

- Source: ERA5 volumetric soil water layers 1–3 (swvl1: 0–7 cm; swvl2: 7–28 cm; swvl3: 28–100 cm)
- Depth-weighted average to 0–100 cm: `RZSM = (7·swvl1 + 21·swvl2 + 72·swvl3) / 100`
- Upscale to 0.5° via conservative area-weighted average (as for the other ERA5 variables)

**e.** Document SPEI3 computation method in study metadata: distribution (log-logistic), parameter estimator (L-moments), and fitting period. v1 default: full-record fit (1950–2024) with a sensitivity test on the last fold.

**f.** Stationarity diagnostics (run once, report in supplementary):

- Mann-Kendall trend test on annual mean SPEI3, per cell
- KS test on SPEI3 distributions for 1950–1989 vs 1990–2024
- Diagnostic only — does not gate modeling

## 3. Feature Engineering

### 3.1 Target Construction (Lead-Specific)

**y(t) = SPEI3(t + L), L ∈ {1, 3, 6} months**

Three independent datasets: `Dataset_L1`, `Dataset_L3`, `Dataset_L6`.

- Training: all months 1950–2024
- Evaluation: target month ∈ [Nov–Feb] only

### 3.2 Predictor Set at Forecast Issue Time *t*

**(A) Contemporary predictors (time *t*)**

`Precip, Tmax, Tmin, PET, VPD, Wind, Solar, TCWV, RZSM, ENSO, NAO, MO, SPEI3`

**(B) Lag selection methodology**

- **PACF as primary tool** (not ACF), computed per CV fold on training data only.
- Test lags 1–12 uniformly for **all variables** — let the data prune.
- Significance threshold: **|PACF| > 0.20**, with a sensitivity test at 0.10 and 0.30 reported in appendix. This is much stricter than 95 % statistical significance (Bartlett's threshold = 1.96/√N ≈ 0.065 for N = 900, ≈ 0.113 for N = 300 winter-only); 0.20 is a deliberate practical-relevance filter rather than a significance filter.
- **CCF screening** between SPEI3 at time *t* and each candidate predictor at time *t − k*, restricted to **target months t ∈ {Nov, Dec, Jan, Feb}** since that is the evaluation season. Predictors whose CCF crosses |CCF| > 0.20 at any lag 1..12 are included at that lag.
- **Selected lags = union of PACF-passing lags and CCF-passing lags.** Variables with no surviving lags are dropped from the lagged feature set (the contemporary feature is unaffected).

**Spatial aggregation for lag selection.** PACF and CCF require a single 1-D time series per variable, but the inputs are gridded. The procedure:

1. Load the Morocco boundary from `shapefiles/MAR_adm0.shp` (164 cells inside, out of 4096 in the full grid).
2. Restrict the gridded variable to Morocco cells.
3. **Approach A (default)** — `spatial_mean` of the Morocco cells, then PACF + CCF on the resulting 1-D series.
4. **Approach B (alternative)** — PACF + CCF on each Morocco cell's 1-D series, then average those 164 spectra; threshold applied to the averaged spectrum. Implemented for transparency / sensitivity; reported in the appendix.

The two approaches agree closely for region-coherent signals (NAO, ENSO, MO, SPEI3, precip) and differ slightly for spatially-heterogeneous variables (RZSM, TCWV, VPD) where Approach A retains more long lags.

**Important: spatial averaging is used *only* for lag selection.** The CV / modeling pipeline (Sections 4–7) treats every (cell, time) sample as an independent observation, with `(lat, lon)` included as features (Section 3.2 D). No spatial averaging is applied at training or inference time.

- For linear models: hand all surviving lags to **Lasso / Elastic Net** for final pruning.
- For RF / XGBoost: use the PACF + CCF output directly (trees handle redundancy).

Variables explicitly tested for long-memory lags (up to 12): SPEI3, RZSM, ENSO, NAO, MO, TCWV, Precip, VPD. Fast-response variables tested for lag 1 only by default: Tmin, Tmax, Solar, Wind (extendable if PACF justifies).

**(C) Seasonal encoding**

- `sin(2π · m / 12)`, `cos(2π · m / 12)` where *m* is month index.

**(D) Spatial encoding**

- `latitude`, `longitude` as features for the **global model** (Section 7.2).

## 4. Cross-Validation Strategy

### 5-fold rolling-origin, expanding training window, continuous test windows

Test windows are **contiguous** so the per-fold out-of-sample predictions stitch into an unbroken 2000–2024 array suitable for pooled-metric computation.

| Fold | Train (planned) | Validate (planned, 8 y) | Test (5 y, continuous) |
| --- | --- | --- | --- |
| 1 | 1950-01 → 1991-12 (42 y) | 1992-01 → 1999-12 | **2000-01 → 2004-12** |
| 2 | 1950-01 → 1996-12 (47 y) | 1997-01 → 2004-12 | **2005-01 → 2009-12** |
| 3 | 1950-01 → 2001-12 (52 y) | 2002-01 → 2009-12 | **2010-01 → 2014-12** |
| 4 | 1950-01 → 2006-12 (57 y) | 2007-01 → 2014-12 | **2015-01 → 2019-12** |
| 5 | 1950-01 → 2011-12 (62 y) | 2012-01 → 2019-12 | **2020-01 → 2024-12** |

Pooled out-of-sample series: **2000-01 → 2024-12 unbroken (25 y; ≈ 100 winter target months × spatial-pooling cells)**.

The "planned" sizes above are subsequently shrunk by a per-fold **boundary gap** (quarantine) — see §4.1.

**Leakage controls**:

- Target shifting performed before splitting; no future predictors.
- Per-fold standardization (Section 5).
- Per-fold feature selection (Section 6).
- Per-fold PACF / CCF lag selection (Section 3.2 B).
- **Per-fold boundary gap (quarantine) — Section 4.1.**

**Aggregation across folds**:

- **Pooled**: concatenate predictions across folds → compute headline metric once. With the continuous-test design this is exactly a single, unbroken 25-year out-of-sample array (100 winter months × cells). Pooled metrics are the primary reporting unit.
- **Per-fold**: same metrics reported per fold in supplementary table to show stability. Per-fold winter sample is small (20 months), so per-fold metrics are noisy by design; do not over-interpret single-fold differences.
- **Winter-only and all-months reporting**: the headline metric is computed on **winter target months only** (t ∈ {Nov, Dec, Jan, Feb}). A parallel **all-months evaluation** is computed on every test month and reported alongside as a supplementary diagnostic — it uses 4× more evaluation samples and tests whether skill is uniform across seasons or concentrated in the winter target.

### 4.1 Boundary gap (quarantine)

Two distinct leakage paths must be closed at every train→val and val→test boundary. The quarantine width must dominate **both** simultaneously.

**Path 1 — Lagged-predictor leakage.** A val (or test) sample at feature time *t* uses lagged predictors SPEI3(*t*−1), SPEI3(*t*−2), …, SPEI3(*t*−*K*), where *K* is the deepest selected lag for the fold. If those *t*−*k* months are themselves training **targets**, the model has effectively fit (X, y) at the same months whose feature values it now reads. Information flows backward through the lag features.

**Path 2 — Rolling-window contamination.** SPEI3 is a 3-month standardized index: SPEI3(τ) is a deterministic function of precipitation at {τ−2, τ−1, τ}. The same *physical precipitation month* can therefore appear inside **both** a training target SPEI3(t_train + *L*) **and** a val/test predictor SPEI3_lag(k)(t'), even when those targets and predictors are at different calendar months and *t' ≠ t_train + L*. The naïve "lead + 2" purge — which covers only the rolling window of the contemporaneous feature — misses this whenever *K* > 0.

**Derivation of the required gap.** Only variables whose lag features share **raw precipitation or PET ingredients** with the SPEI3 target can leak through shared months. We call these the *precip-touching* variables. In our feature set: SPEI3 (and its lags), precip (and its lags), and PET (only contemporaneous in our config). All other variables — NAO, ENSO, MO, RZSM, TCWV, VPD, wind, solar, tmin, tmax — are derived from **independent data sources** (NOAA pressure indices, ERA5 atmosphere, ERA5 soil moisture), so their lag features have only legitimate statistical dependence with future SPEI3; they impose **no** quarantine constraint.

Per precip-touching feature class, the precip-month footprint at val feature time *t* is:

| Feature class | Precip months touched | Disjoint-from-train-target constraint |
| --- | --- | --- |
| Contemporaneous precip(*t*), PET(*t*) | {*t*} | gap ≥ *L* |
| Lagged precip_lag(*k*)(*t*), PET_lag(*k*)(*t*) | {*t* − *k*} | gap ≥ *L* + *k* |
| Contemporaneous SPEI3(*t*) | [*t* − 2, *t*] | gap ≥ *L* + 2 |
| Lagged SPEI3_lag(*k*)(*t*) | [*t* − *k* − 2, *t* − *k*] | gap ≥ *L* + *k* + 2 |

Taking the maximum over all present precip-touching feature classes, define the **effective lag**:

K_eff = max( 0, max_{k ∈ SPEI3 lags} k, max_{k ∈ precip lags}(k − 2), max_{k ∈ PET lags}(k − 2) )

so that

> **gap = *L* + *K*_eff + 2**

closes all strict precip-month leakage. The floor at 0 enforces gap ≥ *L* + 2, which is the constraint from the contemporaneous SPEI3 feature (always present).

**Adaptive K_eff.** K_eff is determined per fold from PACF on the autoregressive variables + winter-only CCF on the climate indices, run over the **provisional** train slice; the per-variable lag dict is then mapped through the `compute_quarantine_max_lag` helper (precip-touching variables only) to obtain K_eff. The gap `L + K_eff + 2` is applied and indices are finalized:

```
for fold in cv_folds:
    1. Provisional split using planned windows above.
    2. Run PACF + winter-only CCF on the provisional train → selected_lags
       (a dict variable_name → list of selected lag depths).
    3. Compute K_eff:
         K_eff = max(
             0,
             max(selected_lags["spei3"], default=0),
             max(selected_lags["precip"], default=2) − 2,
             max(selected_lags["pet"],    default=2) − 2,
         )
       (variables not in the precip-touching set are ignored).
    4. Refine indices:
         gap = L + K_eff + 2
         train_indices = months in [train_start  ..  val_start  − gap − 1]
         val_indices   = months in [val_start    ..  test_start − gap − 1]
         test_indices  = months in [test_start   ..  test_end]    # never shrunk
    5. Fit fold-wise standardizer on train_indices only.
    6. Train; tune HPs on val_indices; predict on test_indices.
```

Test windows are **never** shrunk — the pooled out-of-sample stitching covers the continuous 2000-01 → 2024-12 span regardless of lead.

**Effective fold sizes for the worst case L = 3, K_eff = 12 → gap = 17** (e.g. SPEI3 PACF selects lag 12). For lighter precip-touching lag selections the gap shrinks toward the *L* + 2 floor; the per-fold log emitted by the pipeline records both *K*_eff and the resulting train/val sizes.

| Fold | Effective train | Effective val (HP tuning) | Test (5 y, unchanged) |
| --- | --- | --- | --- |
| 1 | 1950-01 → 1990-07 (40 y 7 mo) | 1992-01 → 1998-07 (6 y 7 mo) | 2000-01 → 2004-12 |
| 2 | 1950-01 → 1995-07 (45 y 7 mo) | 1997-01 → 2003-07 (6 y 7 mo) | 2005-01 → 2009-12 |
| 3 | 1950-01 → 2000-07 (50 y 7 mo) | 2002-01 → 2008-07 (6 y 7 mo) | 2010-01 → 2014-12 |
| 4 | 1950-01 → 2005-07 (55 y 7 mo) | 2007-01 → 2013-07 (6 y 7 mo) | 2015-01 → 2019-12 |
| 5 | 1950-01 → 2010-07 (60 y 7 mo) | 2012-01 → 2018-07 (6 y 7 mo) | 2020-01 → 2024-12 |

For *L* = 6 each train/val end shifts back another 3 months; for *L* = 1 each shifts forward 2 months relative to the table above. The cost (5 extra quarantined months per boundary vs the previous *K*-only gap) is small relative to the ~500+ months of train data per fold.

## 5. Standardization

Fold-wise standardization using **training-period statistics only**. The (μ, σ) statistics are always computed from the training slice exclusively (no leakage from val or test), and are always restricted to **Morocco cells** via the `MAR_adm0.shp` shapefile mask (164 of 4096 grid cells), so that non-Morocco grid points (e.g. Saharan fringe with extreme aridity distortion) don't bias the regional statistics.

**Pre-standardized exception list — do NOT re-standardize**:

- ENSO, NAO, MO (and their lags)
- **SPEI3 (and its lags)** when used as a predictor
- `target` (SPEI3 shifted forward by the lead)

All other variables — **including RZSM** — are standardized fold-wise.

**Two variants of (μ, σ) computation are implemented**, and the choice is tied to the modeling-unit choice in §7.2:

| Variant | (μ, σ) scope | Spatial heterogeneity | Used by |
| --- | --- | --- | --- |
| **`FoldStandardizer`** (pooled, **default**) | One (μ, σ) per variable, pooled across (time × Morocco cells). | **Preserved** in z-space — Marrakech and Tangier sit at different z-values reflecting their position in the regional distribution. The global model uses `(lat_feat, lon_feat)` to learn cell-specific behavior. | Global model (v1 headline; §7.2 default) |
| **`PerCellStandardizer`** (per-cell) | One (μ, σ) per (variable, lat, lon) cell, computed from that cell's own training-period time series. | **Removed** — every Morocco cell normalized to (μ=0, σ=1) over its own history. `(lat, lon)` features should be **dropped** (they become zero-variance constants within a cell). | Per-cell sensitivity test (§7.2) |

Both variants compute statistics from training data only, apply the Morocco mask, and respect the same exception list. Non-Morocco cells in the `PerCellStandardizer` pass through unchanged (identity transform).

Procedure per fold:

1. Compute mean / std on training data only (variant-dependent scope).
2. Apply the transform to training, validation, and test sets.
3. Exception-list variables pass through untouched, both variants.

The `03_cv_visualization` notebook demonstrates a side-by-side comparison of both variants on a representative fold, including verification that the exception list bypasses both standardizers.

## 6. Feature Selection (per fold)

A single selection method is used: **Lasso / Elastic Net** for linear models (selection happens during fitting via the L1 penalty). Tree-based models use no explicit selection — they tolerate redundant features natively. Permutation importance (RF) and SHAP (XGBoost) are computed for **interpretability and reporting only**, not used to drop features.

| Method | Role | Applied to |
| --- | --- | --- |
| **Lasso / Elastic Net** | Selection (during fit via L1) | Linear models — primary |
| Permutation importance | Diagnostic (post-hoc ranking) | Random Forest |
| SHAP values | Diagnostic (post-hoc ranking + interpretability) | XGBoost |

**Pipeline per model family**:

- **Linear models** (OLS / Ridge / Lasso / Elastic Net): All Section-3 features pass to the fitter; Lasso / Elastic Net's L1 penalty produces the final sparse coefficient vector. Multicollinearity is absorbed by the L1 / L2 regularizer; no separate VIF pre-filter is applied.
- **RF / XGBoost**: All Section-3 features pass to the fitter. After training, permutation importance (RF) and SHAP (XGBoost) are reported as paper diagnostics.

Selection is repeated per fold (Lasso is refit per fold), preserving the leakage discipline.

### 6.1 Per-fold pipeline at a glance

The diagram below traces the order of operations for one fold, tying together §3 (features), §4 (CV + boundary gap), §5 (standardization), §6 (selection), §7 (model fit), §8 (HP tuning), and §10 (evaluation). Every step uses **training-data-only** statistics; the test slice is never touched until step 9.

```
+----------------------------------------------------------------------+
|              PER-FOLD CV PIPELINE  (one of 5 folds, one lead L)      |
|   Inputs:  planned fold windows (cv.yaml)  +  raw datasets  +  L     |
+----------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 1. PROVISIONAL SPLIT  (Sec.4)                                    |
   |    Use planned windows [train_start..val_start..test_start..     |
   |    test_end].  No quarantine applied yet.                        |
   |    train_planned = [train_start .. val_start - 1]                |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 2. PER-FOLD LAG SELECTION  (Sec.3.2 B)                           |
   |    Restricted to train_planned + Morocco mask.                   |
   |     - PACF on long-memory vars (SPEI3, RZSM, TCWV, precip, ...)  |
   |     - Winter-only CCF on climate indices (ENSO, NAO, MO)         |
   |    -> selected_lags : dict[var -> list[k]]                       |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 3. ADAPTIVE BOUNDARY GAP  (Sec.4.1, strict precip-touching rule) |
   |    K_eff = compute_quarantine_max_lag(selected_lags)             |
   |            (SPEI3 lags + precip lags - 2; others ignored)        |
   |    gap   = L + K_eff + 2                                         |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 4. FINAL INDICES  (apply gap to end of train & val)              |
   |    train_idx = [train_start .. val_start  - gap - 1]             |
   |    val_idx   = [val_start   .. test_start - gap - 1]             |
   |    test_idx  = [test_start  .. test_end]   <- NEVER shrunk       |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 5. FEATURE-DATASET BUILD  (Sec.3)                                |
   |    Contemporaneous predictors + selected_lags features           |
   |    + seasonal sin/cos + (lat, lon)  + target SPEI3(t + L)        |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 6. PER-FOLD STANDARDIZATION  (Sec.5)                             |
   |    Fit (mu, sigma) on train_idx ONLY, Morocco mask applied.      |
   |    Apply to train / val / test.                                  |
   |    Variant set per experiment (modeling_unit):                   |
   |      * FoldStandardizer    (pooled over Morocco)   <- default    |
   |      * PerCellStandardizer (per (lat, lon) cell)   <- 7.2 sens.  |
   |    Exception list passes through unchanged                       |
   |    ({ENSO, NAO, MO, SPEI3, target, *_lag* of these}).            |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 7. MODEL FIT + HP TUNING  (Sec.7 / Sec.8 - Protocol A)           |
   |    For each (model_family, HP) on the grid:                      |
   |       fit on train_idx, score on val_idx                         |
   |    Linear models: L1 inside Lasso/ElasticNet does the selection. |
   |    Tree models  : RF / XGBoost use all features (no pre-filter). |
   |    -> best_hp                                                    |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 8. REFIT  with best_hp on the CONTIGUOUS slice                   |
   |    [train_start .. test_start - gap - 1]                         |
   |    (reclaims the train-val gap; only val->test gap remains)      |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 9. PREDICT  on test_idx                                          |
   |    Per-fold predictions stitched across folds                    |
   |    -> continuous out-of-sample array 2000-01 -> 2024-12.         |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 10. EVALUATION + POST-HOC DIAGNOSTICS  (Sec.10 / Sec.6)          |
   |     Metrics on (winter pool, all months) + block-bootstrap CIs.  |
   |     Post-hoc diagnostics - out-of-sample (test_idx) only:        |
   |       * Permutation importance  (RF)                             |
   |       * TreeSHAP mean(|SHAP|)   (XGBoost)                        |
   +------------------------------------------------------------------+
```

Steps 1–6 are pure per-fold preprocessing; steps 7–10 are per (fold, model_family, lead). Each fold is independent — the orchestrator (Phase 10) parallelises the outer loop trivially.

## 7. Model Configurations

### 7.0 Baselines

| Baseline | Forecast formula | Purpose |
| --- | --- | --- |
| Climatology | per-cell per-calendar-month training mean | Reference for skill scores |
| Persistence | SPEI3(t+L) = SPEI3(t) | Inertia benchmark |
| AR(p) | Linear regression on lagged SPEI3 only | Univariate autoregressive benchmark (subsumes damped persistence at p = 1) |

### 7.1 ML Models

| Model | Input | Purpose |
| --- | --- | --- |
| Linear (OLS) | Tabular | Statistical baseline |
| **Ridge** | Tabular | L2-regularized linear |
| **Lasso** | Tabular | L1-regularized (also feature selector) |
| **Elastic Net** | Tabular | L1 + L2 regularized |
| Random Forest | Tabular | Nonlinear ensemble |
| XGBoost | Tabular | Gradient boosting (primary tabular benchmark) |
| LSTM | Lagged sequences | Temporal memory (deferred) |
| CNN | Spatial grids | Spatial pattern extraction (deferred) |
| CNN-LSTM / ConvLSTM | Spatio-temporal sequences | Joint spatio-temporal (deferred) |

### 7.2 Modeling Unit

- **Default**: global model with (lat, lon) as features — one model per (model family, fold), trained on all 64 × 64 cells jointly.
- **Sensitivity test**: per-cell models for the best-performing family.

## 8. Hyperparameter Tuning

**Protocol A** — tune on validation, refit on train + val, evaluate on test:

```
for fold in cv_folds:
    best_hp = search(model_type, X_train, y_train, X_val, y_val)
    final_model = model_type(best_hp).fit([train_start, test_start - gap - 1], y_train U y_val)
    preds = final_model.predict(X_test)
```

**Search method**:

- **Grid search is the primary method for all models** (Ridge, Lasso, Elastic Net, RF, XGBoost) — reproducible, transparent, and acceptable in cost given the reduced XGBoost grid below.
- **Bayesian (Optuna)** is available as an **optional** alternative for RF and XGBoost (selectable via the experiment YAML), to be used if/when the grid becomes a bottleneck.

**Search spaces (grid)**:

| Model | Space | Combinations |
| --- | --- | --- |
| Ridge / Lasso | `alpha ∈ logspace(−3, 3, 13)` | 13 |
| Elastic Net | `alpha ∈ logspace(−3, 3, 7); l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` | 35 |
| RF | `n_estimators ∈ {200, 500, 1000}; max_depth ∈ {None, 5, 10, 20}; min_samples_leaf ∈ {1, 5, 20}; max_features ∈ {sqrt, 0.5, 1.0}` | 108 |
| XGBoost (reduced grid) | `max_depth ∈ {4, 6, 8}; lr ∈ {0.05, 0.1}; subsample ∈ {0.7, 1.0}; colsample_bytree ∈ {0.7, 1.0}; reg_lambda ∈ {0.1, 1.0, 10.0}; min_child_weight ∈ {1, 5}`; `n_estimators` via early stopping on val | 144 |

The XGBoost space is intentionally compact (144 combos vs. 6,075 in the full v2 draft) so that full grid search remains tractable across 5 folds × 3 leads. Optuna can explore the full space if needed.

## 9. Forecast Generation

For each issue month *t*:

1. Extract predictors up to month *t*.
2. Apply trained model for lead *L* (one model per (family, fold, lead)).
3. Produce forecast SPEI3̂(t + L).
4. Retain forecasts verifying in Nov–Feb only.

## 10. Evaluation Metrics

### 10.1 Deterministic (continuous SPEI3) — headline

| Metric | Formula | Role |
| --- | --- | --- |
| **MAE** | mean(\|ŷ − y\|) | Headline — error magnitude |
| **RMSE** | √[mean((ŷ − y)²)] | Headline — error magnitude |
| **Pearson *r*** | corr(ŷ, y) | Headline — pattern correlation |
| **ACC** | corr(ŷ − clim, y − clim) | Headline — anomaly correlation |
| **MSSS vs climatology** | 1 − MSE(model) / MSE(climatology) | Headline — % improvement over climatology |
| **MSSS vs persistence** | 1 − MSE(model) / MSE(persistence) | Headline — added value beyond inertia |

### 10.2 Optional metrics (implemented in code, not reported by default)

- **HSS at SPEI3 < −1.0** — binary categorical (drought / no drought). Available via the metrics config but not part of the v1 headline.
- **POD, FAR, CSI, ETS, multi-class HSS** — not implemented in v1; deferred to a follow-up if categorical evaluation becomes a focus.
- **Probabilistic (CRPS, Brier, reliability)** — deferred; requires quantile / probabilistic models.

### 10.3 Headline Metrics (committed)

The six deterministic metrics in §10.1 form the headline set. They are reported per (model, lead) on the **pooled** out-of-sample array (2000–2024 winter target months for the headline; all months for the supplementary table).

### 10.4 Reporting Structure

- Per lead time (L = 1, 3, 6).
- Per CV fold (supplementary table) + pooled across folds (headline).
- Spatial: per-cell skill maps + pooled metrics across cells.
- **Winter-only** evaluation (t ∈ Nov–Feb) is the **headline** unit; **all-months** evaluation is reported as a supplementary diagnostic alongside (uses 4× more samples, lets us check whether skill is winter-specific or season-uniform).
- **Block bootstrap 95 % CIs** on all headline metrics:
  - Stationary bootstrap; mean block ≈ 12 months.
  - For winter-only metrics: year-blocks (full Nov–Feb season per block).
  - 1000 replicates.

## 11. Outputs

| Output | Description |
| --- | --- |
| Grid-level forecasts | 0.5° SPEI3 predictions per (model, lead) |
| Spatial skill maps | Per-cell ACC, MSSS for each (model, lead) |
| Forecast-vs-truth time series | At selected cells (Tangier, Imilchil, Agadir) |
| Feature importance diagnostics | Permutation (RF), SHAP (XGBoost), per lead |
| Skill comparison tables | Models × leads × headline metrics, with bootstrap CIs (winter-only headline; all-months supplementary) |
| Per-fold stability tables | Same metrics broken by fold |
| Winter-vs-all-months skill diagnostic | Table comparing each model's metrics on winter targets vs all-month targets, per lead |
| Baseline-vs-ML skill plots | Bars showing MSSS for each model relative to each baseline |
| Lag selection diagnostics | PACF / CCF plots per variable per fold |
| Stationarity diagnostics | Mann-Kendall and KS test results per cell |

## Summary of Experimental Structure (v3)

| Dimension | Specification |
| --- | --- |
| Time period | 1950–2024 |
| Temporal resolution | Monthly |
| Spatial resolution | 64 × 64 grid at 0.5° over Morocco |
| Target | SPEI-3 at L = 1, 3, 6 months |
| Evaluation season | Winter (Nov–Feb) |
| Cross-validation | 5-fold rolling-origin, expanding train window, **continuous test windows (2000–2024 unbroken)** |
| Leakage control | Fold-wise standardization, feature selection, lag selection; strict target shifting; **boundary-gap quarantine `gap = L + K_eff + 2`** (precip-touching variables only) at train → val and val → test boundaries |
| Pre-standardized exceptions | ENSO, NAO, MO, **SPEI3 (as predictor)**, target |
| Climate drivers | ENSO, NAO, MO (+ optional AMO, AO in v2) |
| Lag selection | PACF + winter-only CCF on Morocco-masked spatial mean; threshold \|·\| > 0.20; sensitivity ∈ {0.10, 0.30}; Lasso finalizes for linear |
| Region mask for lag selection | `shapefiles/MAR_adm0.shp` (164 cells inside) — Approach A (spatial mean) default; Approach B (per-cell then mean) reported as appendix sensitivity |
| Modeling unit | Global model with (lat, lon) features (v1); per-cell as sensitivity |
| Standardization variants | `FoldStandardizer` (pooled over Morocco, default for global model); `PerCellStandardizer` (per (lat, lon), for §7.2 sensitivity) |
| Baselines | Climatology, persistence, AR(p) |
| Feature selection | Lasso / Elastic Net for linear (selection during fit); no explicit selection for trees; SHAP / permutation importance reported as diagnostics |
| ML models | OLS / Ridge / Lasso / Elastic Net, RF, XGBoost |
| DL models | LSTM, CNN, CNN-LSTM (or ConvLSTM) — deferred |
| HP tuning | Protocol A — tune on val, refit on train + val, eval on test |
| HP search | Grid search primary for all models (Ridge/Lasso/EN: 13–35 combos; RF: 108; XGBoost reduced grid: 144). Optuna optional for RF / XGBoost |
| Headline metrics (all deterministic) | MAE, RMSE, Pearson *r*, ACC, MSSS-vs-clim, MSSS-vs-persistence |
| Optional metric | HSS at SPEI3 < −1.0 (available in code, not in v1 paper headline) |
| Evaluation window | Winter-only (Nov–Feb) headline; all-months supplementary diagnostic |
| Uncertainty | Stationary block bootstrap, 1000 replicates, 95 % CI |
| Outputs | Forecasts, spatial skill maps, time series, importance, skill tables, CIs |
