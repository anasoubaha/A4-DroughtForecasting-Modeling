---
title: "SPEI3 Forecasting Scheme — v2"
author: "Anas Soubaha"
date: "2026-05-24"
geometry: margin=2cm
fontsize: 11pt
---

# SPEI3 Forecasting Scheme — v2

## 1. Inputs (Before Preprocessing)

| Input group | Variable | Temporal coverage | Spatial resolution | Storage | Source |
|---|---|---|---|---|---|
| Climatological | Precipitation (mm/mon) | 1901–2024 | 0.5° × 0.5° | NetCDF | CRU |
| | Min, Max temperature (°C) | 1901–2024 | 0.5° × 0.5° | NetCDF | CRU |
| | PET (mm/day) | 1901–2024 | 0.5° × 0.5° | NetCDF | CRU |
| | Solar radiation (J·m⁻² / W·m⁻²) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
| | Wind speed at 2 m (m·s⁻¹) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
| | VPD (hPa) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
| | TCWV (kg/m²) | 1940–2025 | 0.25° × 0.25° | NetCDF | ERA5 |
| | **Rootzone Soil Moisture (RZSM, 0–100 cm, m³/m³)** | 1950–2024 | 0.25° × 0.25° | NetCDF | ERA5-Land (derived from swvl1–swvl3) |
| Large-scale climate indices | NAO Index | 1950–2026 | — | Text | NOAA CPC |
| | ENSO Niño 3.4 | 1870–2025 | — | NetCDF | NOAA PSL |
| | Mediterranean Oscillation (MO) | 1940–2025 | — | NetCDF | ERA5-derived (Gibraltar − Lod SLP, standardized 1981–2010) |

**Potential additions** deferred to a future iteration if v1 underperforms:

- AMO (Atlantic Multidecadal Oscillation) — documented linkage to North African precipitation
- AO (Arctic Oscillation) — partial NAO redundancy, possible additive skill

## 2. Preprocessing

**a.** Align time period **1950–2024** for all input variables.

**b.** Upscale ERA5 → 0.5° via conservative area-weighted averaging. All gridded data: 64 × 64 at 0.5°.

**c.** Missing-value handling (explicit):

- Temporal interpolation up to 2 consecutive missing months
- Drop cells with > 5 % missing in the training period of any fold

**d.** **RZSM derivation** (replaces SSMI in v2):

- Source: ERA5-Land volumetric soil water layers 1–3 (swvl1: 0–7 cm; swvl2: 7–28 cm; swvl3: 28–100 cm)
- Depth-weighted average to 0–100 cm: `RZSM = (7·swvl1 + 21·swvl2 + 72·swvl3) / 100`
- Upscale to 0.5° via conservative area-weighted average (as for the other ERA5 variables)
- **Rationale**: SSMI is pre-standardized, which would introduce subtle look-ahead leakage similar to SPEI3. Using raw RZSM and standardizing fold-wise avoids this.

**e.** Document SPEI3 computation method in study metadata: distribution (log-logistic), parameter estimator (L-moments), and fitting period. v1 default: full-record fit (1950–2024) with a sensitivity test on the last fold.

**f.** Stationarity diagnostics (run once, report in supplementary):

- Mann-Kendall trend test on annual mean SPEI3, per cell
- KS test on SPEI3 distributions for 1950–1989 vs 1990–2024
- Diagnostic only — does not gate modeling

## 3. Feature Engineering

### 3.1 Target Construction (Lead-Specific)

**y_t = SPEI3(t + L),  L ∈ {1, 3, 6} months**

Three independent datasets: Dataset_L1, Dataset_L3, Dataset_L6.

- Training: all months 1950–2024
- Evaluation: target month ∈ [Nov–Feb] only

### 3.2 Predictor Set at Forecast Issue Time *t*

**(A) Contemporary predictors (time *t*)**

`Precip, Tmax, Tmin, PET, VPD, Wind, Solar, TCWV, RZSM, ENSO, NAO, MO, SPEI3`

**(B) Lag selection methodology**

- **PACF as primary tool** (not ACF), computed per CV fold on training data only
- Test lags 1–12 uniformly for **all variables** — let the data prune
- Significance threshold: **|PACF| > 0.20**, with a sensitivity test at 0.10 and 0.30 reported in appendix
- **CCF screening** between SPEI3 and each candidate predictor on the training fold; include the strongest-lag CCF result if not already selected by PACF
- For linear models: hand all surviving lags to **Lasso / Elastic Net** for final pruning
- For RF / XGBoost: use the PACF + CCF output directly (trees handle redundancy)

Variables explicitly tested for long-memory lags (up to 12): SPEI3, RZSM, ENSO, NAO, MO, TCWV, Precip, VPD.
Fast-response variables tested for lag 1 only by default: Tmin, Tmax, Solar, Wind (extendable if PACF justifies).

**(C) Seasonal encoding**

- `sin(2π · m / 12)`, `cos(2π · m / 12)` where *m* is month index

**(D) Spatial encoding**

- `latitude`, `longitude` as features for the **global model** (Section 7.2)

## 4. Cross-Validation Strategy

### 5-fold rolling-origin, expanding training window

| Fold | Train | Validate | Test |
|---|---|---|---|
| 1 | 1950–1989 (40 y) | 1990–1996 (7 y) | 1997–2001 (5 y) |
| 2 | 1950–1996 (47 y) | 1997–2001 (5 y) | 2002–2006 (5 y) |
| 3 | 1950–2006 (57 y) | 2007–2011 (5 y) | 2012–2016 (5 y) |
| 4 | 1950–2011 (62 y) | 2012–2016 (5 y) | 2017–2020 (4 y) |
| 5 | 1950–2016 (67 y) | 2017–2020 (4 y) | 2021–2024 (4 y) |

**Leakage controls**:

- Target shifting performed before splitting; no future predictors
- Per-fold standardization (Section 5)
- Per-fold feature selection (Section 6)
- Per-fold PACF / CCF lag selection (Section 3.2 B)

**Aggregation across folds**:

- **Pooled**: concatenate predictions across folds → compute headline metric once
- **Per-fold**: same metrics reported per fold in supplementary table to show stability

## 5. Standardization

Fold-wise standardization using **training-period statistics only**.

**Pre-standardized exception list — do NOT re-standardize**:

- ENSO, NAO, MO (and their lags)
- **SPEI3 (and its lags)** when used as a predictor

All other variables — **including RZSM** — are standardized fold-wise.

Procedure per fold:

1. Compute mean / std on training data only
2. Apply to training, validation, and test sets

## 6. Feature Selection (per fold)

| Method | Applied to | Purpose |
|---|---|---|
| VIF (threshold < 5) | Linear-family pre-filter | Remove multicollinearity |
| Lasso / Elastic Net | Final feature set for linear models | L1 sparsity, lag pruning |
| Permutation importance | RF | Rank features, report top-*k* |
| SHAP values | XGBoost | Interpretability and ranking |
| RFE | Tabular models (optional) | Subset-selection sensitivity |

**Pipeline per model family**:

- Linear / Ridge / Lasso / Elastic Net: VIF filter → Lasso / Elastic Net during fit
- RF: all features in; permutation importance for diagnostics
- XGBoost: all features in; SHAP for diagnostics

## 7. Model Configurations

### 7.0 Baselines

| Baseline | Forecast formula | Purpose |
|---|---|---|
| Climatology | per-cell per-calendar-month training mean | Reference for skill scores |
| Persistence | SPEI3(t+L) = SPEI3(t) | Inertia benchmark |
| Damped persistence | SPEI3(t+L) = α^L · SPEI3(t), α = lag-L autocorrelation | Optimal AR(1), strong mid-lead benchmark |
| AR(p) | Linear regression on lagged SPEI3 only | Univariate AR benchmark |
| Teleconnection-only (optional) | Linear regression on ENSO / NAO / MO lags only | Skill from large-scale drivers alone |

### 7.1 ML Models

| Model | Input | Purpose |
|---|---|---|
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

- **Default**: global model with (lat, lon) as features — one model per (model family, fold), trained on all 64 × 64 cells jointly
- **Sensitivity test**: per-cell models for the best-performing family

## 8. Hyperparameter Tuning

**Protocol A** — tune on validation, refit on train + val, evaluate on test:

```
for fold in cv_folds:
    best_hp = search(model_type, X_train, y_train, X_val, y_val)
    final_model = model_type(best_hp).fit(X_train ∪ X_val, y_train ∪ y_val)
    preds = final_model.predict(X_test)
```

**Search method**:

- **Grid search**: Ridge, Lasso, Elastic Net
- **Bayesian (Optuna)**: RF, XGBoost — 50–100 trials each

**Suggested search spaces**:

| Model | Space |
|---|---|
| Ridge / Lasso | `alpha ∈ logspace(−3, 3, 13)` |
| Elastic Net | `alpha ∈ logspace(−3, 3, 7); l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` |
| RF | `n_estimators ∈ {200, 500, 1000}; max_depth ∈ {None, 5, 10, 20}; min_samples_leaf ∈ {1, 5, 20}; max_features ∈ {sqrt, 0.5, 1.0}` |
| XGBoost | `n_estimators` via early stopping on val; `max_depth ∈ {3, 6, 10}; lr ∈ {0.01, 0.05, 0.1}; subsample, colsample_bytree ∈ {0.5, 0.8, 1.0}; reg_alpha, reg_lambda ∈ logspace(−3, 1, 5); min_child_weight ∈ {1, 5, 20}` |

## 9. Forecast Generation

For each issue month *t*:

1. Extract predictors up to month *t*
2. Apply trained model for lead *L* (one model per (family, fold, lead))
3. Produce forecast SPÊI3(t + L)
4. Retain forecasts verifying in Nov–Feb only

## 10. Evaluation Metrics

### 10.1 Deterministic (continuous SPEI3)

| Metric | Formula | Use |
|---|---|---|
| RMSE | √[mean((ŷ − y)²)] | Error magnitude |
| MAE | mean(|ŷ − y|) | Robust error |
| Pearson *r* | corr(ŷ, y) | Pattern correlation |
| **ACC** | corr(ŷ − clim, y − clim) | **Headline** — anomaly correlation |
| **MSSS vs climatology** | 1 − MSE(model) / MSE(climatology) | **Headline** — % improvement over climatology |
| MSSS vs persistence | 1 − MSE(model) / MSE(persistence) | Added value beyond inertia |

### 10.2 Categorical (drought classes — McKee thresholds)

| Class | SPEI3 |
|---|---|
| Extreme drought | < −2.0 |
| Severe drought | [−2.0, −1.5) |
| Moderate drought | [−1.5, −1.0) |
| Normal / wet | ≥ −1.0 |

| Metric | Use |
|---|---|
| POD (hit rate) | Drought detection rate |
| FAR | False alarm ratio |
| CSI | Combined accuracy |
| **HSS** | **Headline** — binary at SPEI3 < −1.0 and multi-class |
| ETS | Chance-corrected hits |

### 10.3 Probabilistic (optional)

CRPS, Brier skill score, reliability diagrams — required only if uncertainty quantification is added (recommended via quantile XGBoost in a follow-up).

### 10.4 Headline Metrics (committed)

1. **ACC**
2. **MSSS vs climatology**
3. **HSS for SPEI3 < −1.0**

### 10.5 Reporting Structure

- Per lead time (L = 1, 3, 6)
- Per CV fold (supplementary table) + pooled across folds (headline)
- Spatial: per-cell skill maps + pooled metrics across cells
- **Block bootstrap 95 % CIs** on all headline metrics:
  - Stationary bootstrap; mean block ≈ 12 months
  - For winter-only metrics: year-blocks (full Nov–Feb season per block)
  - 1000 replicates

## 11. Outputs

| Output | Description |
|---|---|
| Grid-level forecasts | 0.5° SPEI3 predictions per (model, lead) |
| Spatial skill maps | Per-cell ACC, MSSS, HSS for each (model, lead) |
| Forecast-vs-truth time series | At selected cells (Casablanca, Marrakech, Agadir) |
| Feature importance diagnostics | Permutation (RF), SHAP (XGBoost), per lead |
| Skill comparison tables | Models × leads × headline metrics, with bootstrap CIs |
| Per-fold stability tables | Same metrics broken by fold |
| Baseline-vs-ML skill plots | Bars showing MSSS for each model relative to each baseline |
| Lag selection diagnostics | PACF / CCF plots per variable per fold |
| Stationarity diagnostics | Mann-Kendall and KS test results per cell |

## Summary of Experimental Structure (v2)

| Dimension | Specification |
|---|---|
| Time period | 1950–2024 |
| Temporal resolution | Monthly |
| Spatial resolution | 64 × 64 grid at 0.5° over Morocco |
| Target | SPEI-3 at L = 1, 3, 6 months |
| Evaluation season | Winter (Nov–Feb) |
| Cross-validation | 5-fold rolling-origin, expanding train window |
| Leakage control | Fold-wise standardization, feature selection, lag selection; strict target shifting |
| Pre-standardized exceptions | ENSO, NAO, MO, **SPEI3 (as predictor)** |
| Climate drivers | ENSO, NAO, MO (+ optional AMO, AO in v2) |
| Lag selection | PACF + CCF, threshold &#124;PACF&#124; > 0.20, sensitivity ∈ {0.10, 0.30}, Lasso finalizes for linear |
| Modeling unit | Global model with (lat, lon) features (v1); per-cell as sensitivity |
| Baselines | Climatology, persistence, damped persistence, AR(p) |
| ML models | OLS / Ridge / Lasso / Elastic Net, RF, XGBoost |
| DL models | LSTM, CNN, CNN-LSTM (or ConvLSTM) — deferred |
| HP tuning | Protocol A — tune on val, refit on train + val, eval on test |
| HP search | Grid (linear), Optuna (RF, XGBoost); early stopping (XGBoost) |
| Deterministic metrics | RMSE, MAE, ACC, MSSS-vs-clim, MSSS-vs-persistence |
| Categorical metrics | POD, FAR, CSI, HSS, ETS — binary at SPEI3 < −1.0 + multi-class |
| Headline metrics | ACC, MSSS-vs-clim, HSS at SPEI3 < −1.0 |
| Uncertainty | Stationary block bootstrap, 1000 replicates, 95 % CI |
| Outputs | Forecasts, spatial skill maps, time series, importance, skill tables, CIs |
