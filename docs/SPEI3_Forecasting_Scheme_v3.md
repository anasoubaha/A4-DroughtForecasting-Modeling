SPEI3 Forecasting Scheme — v3 

Anas Oubaha 

2026-06-01 

**SPEI3 Forecasting Scheme — v3** 

**1\. Inputs (Before Preprocessing)** 

Input group Variable   
Temporal coverage   
Spatial 

resolution Storage Source 

Climatological Precipitation (mm/mon) 

Min, Max 

temperature 

(°C) 

PET 

(mm/day) 

Solar radiation 

(J·m�² / 

W·m�²) 

Wind speed at 

2 m (m·s�¹)   
1901–2024 0.5° × 0.5° NetCDF CRU 1901–2024 0.5° × 0.5° NetCDF CRU 

1901–2024 0.5° × 0.5° NetCDF CRU 1940–2025 0.25° × 0.25° NetCDF ERA5 

1940–2025 0.25° × 0.25° NetCDF ERA5 

VPD (hPa) 1940–2025 0.25° × 0.25° NetCDF ERA5 

TCWV 

(kg/m²) 

**Rootzone** 

**Soil Moisture (RZSM,** 

**0–100 cm,** 

**m³/m³)**   
1940–2025 0.25° × 0.25° NetCDF ERA5 

1940–2025 0.25° × 0.25° NetCDF ERA5 (derived from 

swvl1–swvl3) 

Large-scale climate indices   
NAO Index 1950–2026 — Text NOAA CPC ENSO Niño 3.4 1870–2025 — NetCDF NOAA PSL   
Mediterranean Oscillation 

(MO)   
1940–2025 — NetCDF ERA5-derived (Gibraltar − 

Lod SLP, 

standardized 

1981–2010) 

**Potential additions** deferred to a future iteration if v1 underperforms: 

• AMO (Atlantic Multidecadal Oscillation) — documented linkage to North African precipitation 1  
• AO (Arctic Oscillation) — partial NAO redundancy, possible additive skill 

**2\. Preprocessing** 

**a.** Align time period **1950–2024** for all input variables. 

**b.** Upscale ERA5 → 0.5° via conservative area-weighted averaging. All gridded data: 64 × 64 at 0.5°. **c.** Missing-value handling (explicit): 

• Temporal interpolation up to 2 consecutive missing months 

• Drop cells with \> 5 % missing in the training period of any fold 

**d. RZSM derivation:** 

• Source: ERA5 volumetric soil water layers 1–3 (swvl1: 0–7 cm; swvl2: 7–28 cm; swvl3: 28–100 cm) • Depth-weighted average to 0–100 cm: RZSM \= (7·swvl1 \+ 21·swvl2 \+ 72·swvl3) / 100 • Upscale to 0.5° via conservative area-weighted average (as for the other ERA5 variables) • 

**e.** Document SPEI3 computation method in study metadata: distribution (log-logistic), parameter estimator (L-moments), and fitting period. v1 default: full-record fit (1950–2024) with a sensitivity test on the last fold. 

**f.** Stationarity diagnostics (run once, report in supplementary): 

• Mann-Kendall trend test on annual mean SPEI3, per cell 

• KS test on SPEI3 distributions for 1950–1989 vs 1990–2024 

• Diagnostic only — does not gate modeling 

**3\. Feature Engineering** 

**3.1 Target Construction (Lead-Specific)** 

**y(t) \= SPEI3(t \+ L), L � {1, 3, 6} months** 

Three independent datasets: Dataset\_L1, Dataset\_L3, Dataset\_L6. 

• Training: all months 1950–2024 

• Evaluation: target month � \[Nov–Feb\] only 

**3.2 Predictor Set at Forecast Issue Time *t*** 

**(A) Contemporary predictors (time *t*)** 

Precip, Tmax, Tmin, PET, VPD, Wind, Solar, TCWV, RZSM, ENSO, NAO, MO, SPEI3 **(B) Lag selection methodology** 

• **PACF as primary tool** (not ACF), computed per CV fold on training data only • Test lags 1–12 uniformly for **all variables** — let the data prune 

• Significance threshold: **|PACF| \> 0.20**, with a sensitivity test at 0.10 and 0.30 reported in appendix. This is much stricter than 95 % statistical significance (Bartlett’s threshold \= 1.96/√N � 0.065 for N \= 900, � 0.113 for N \= 300 winter-only); 0.20 is a deliberate practical-relevance filter rather than a significance filter. 

• **CCF screening** between SPEI3 at time *t* and each candidate predictor at time *t − k*, restricted to **target months t � {Nov, Dec, Jan, Feb}** since that is the evaluation season. Predictors whose CCF crosses |CCF| \> 0.20 at any lag 1..12 are included at that lag. 

2  
• **Selected lags \= union of PACF-passing lags and CCF-passing lags.** Variables with no surviving lags are dropped from the lagged feature set (the contemporary feature is unaffected). 

**Spatial aggregation for lag selection.** PACF and CCF require a single 1-D time series per variable, but the inputs are gridded. The procedure: 

1\. Load the Morocco boundary from shapefiles/MAR\_adm0.shp (164 cells inside, out of 4096 in the full grid). 

2\. Restrict the gridded variable to Morocco cells. 

3\. **Approach A (default)** — spatial\_mean of the Morocco cells, then PACF \+ CCF on the resulting 1-D series. 

4\. **Approach B (alternative)** — PACF \+ CCF on each Morocco cell’s 1-D series, then average those 164 spectra; threshold applied to the averaged spectrum. Implemented for transparency / sensitivity; reported in the appendix. 

The two approaches agree closely for region-coherent signals (NAO, ENSO, MO, SPEI3, precip) and differ slightly for spatially-heterogeneous variables (RZSM, TCWV, VPD) where Approach A retains more long lags. 

**Important: spatial averaging is used *only* for lag selection.** The CV / modeling pipeline (Sections 4–7) treats every (cell, time) sample as an independent observation, with (lat, lon) included as features (Section 3.2 D). No spatial averaging is applied at training or inference time. 

• For linear models: hand all surviving lags to **Lasso / Elastic Net** for final pruning • For RF / XGBoost: use the PACF \+ CCF output directly (trees handle redundancy) 

Variables explicitly tested for long-memory lags (up to 12): SPEI3, RZSM, ENSO, NAO, MO, TCWV, Precip, VPD. Fast-response variables tested for lag 1 only by default: Tmin, Tmax, Solar, Wind (extendable if PACF justifies). 

**(C) Seasonal encoding** 

• sin(2� · m / 12), cos(2� · m / 12\) where *m* is month index 

**(D) Spatial encoding** 

• latitude, longitude as features for the **global model** (Section 7.2) 

**4\. Cross-Validation Strategy** 

**5-fold rolling-origin, expanding training window, continuous test windows** 

Test windows are **contiguous** so the per-fold out-of-sample predictions stitch into an unbroken 2000–2024 array suitable for pooled-metric computation. 

Fold Train (planned) Validate (planned, 8 y) Test (5 y, continuous) 

1 1950-01 → 1991-12 (42 y) 

2 1950-01 → 1996-12 (47 y) 

3 1950-01 → 2001-12 (52 y) 

4 1950-01 → 2006-12 (57 y)   
1992-01 → 1999-12 **2000-01 → 2004-12** 1997-01 → 2004-12 **2005-01 → 2009-12** 2002-01 → 2009-12 **2010-01 → 2014-12** 2007-01 → 2014-12 **2015-01 → 2019-12** 

3  
Fold Train (planned) Validate (planned, 8 y) Test (5 y, continuous) 

5 1950-01 → 2011-12 (62 y)   
2012-01 → 2019-12 **2020-01 → 2024-12** 

Pooled out-of-sample series: **2000-01 → 2024-12 unbroken (25 y; � 100 winter target months × spatial-pooling cells)**. 

The “planned” sizes above are subsequently shrunk by a per-fold **boundary gap** (quarantine) — see §4.1. **Leakage controls**: 

• Target shifting performed before splitting; no future predictors 

• Per-fold standardization (Section 5\) 

• Per-fold feature selection (Section 6\) 

• Per-fold PACF / CCF lag selection (Section 3.2 B) 

• **Per-fold boundary gap (quarantine) — Section 4.1** 

**Aggregation across folds**: 

• **Pooled**: concatenate predictions across folds → compute headline metric once. With the continuous test design this is exactly a single, unbroken 25-year out-of-sample array (100 winter months × cells). Pooled metrics are the primary reporting unit. 

• **Per-fold**: same metrics reported per fold in supplementary table to show stability. Per-fold winter sample is small (20 months), so per-fold metrics are noisy by design; do not over-interpret single-fold differences. 

• **Winter-only and all-months reporting**: the headline metric is computed on **winter target months only** (t in {Nov, Dec, Jan, Feb}). A parallel **all-months evaluation** is computed on every test month and reported alongside as a supplementary diagnostic — it uses 4× more evaluation samples and tests whether skill is uniform across seasons or concentrated in the winter target. 

**4.1 Boundary gap (quarantine)**

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

$$K_\text{eff} \;=\; \max\!\Big( 0,\;\; \max_{k \in \text{SPEI3 lags}}\!k,\;\; \max_{k \in \text{precip lags}}(k - 2),\;\; \max_{k \in \text{PET lags}}(k - 2) \Big)$$

so that

> **gap = *L* + *K*ₑff + 2**

closes all strict precip-month leakage. The floor at 0 enforces gap ≥ *L* + 2, which is the constraint from the contemporaneous SPEI3 feature (always present).

**Why the +2 vanishes for precip/PET lags but not SPEI3 lags.** SPEI3 is a 3-month rolling sum, so a SPEI3 lag at depth *k* touches a 3-month window [*t*−*k*−2, *t*−*k*] — the +2 covers the rolling-window width. Raw precip and PET are point-month quantities, so a precip lag at depth *k* touches a single month {*t* − *k*} — the +2 is unnecessary and we subtract it back out in the *K*ₑff formula.

**The three terms each close one specific leakage path:**
- **+*L***: the target shift — keeps any training target out of the val feature window.
- **+*K*ₑff**: the deepest precip-touching lag — keeps val features from reaching back into training-target precip months.
- **+2**: SPEI3's 3-month rolling window — closes the rolling-window contamination path.

**Worked example A — SPEI3 dominates** (L = 3; PACF selects SPEI3 lags up to 12; everything else shallow or non-precip-touching).
*K*ₑff = max(12, …) = 12 → gap = 3 + 12 + 2 = **17 months**.

Time-axis at the train → val boundary with *T* = last train feature time:

```
                          train target            quarantine                   val feature
                          precip footprint        (17 months)                  precip footprint
                          ↓                                                    ↓
month axis  →   ...  T-14  ...  T-2 T-1 [T] T+1 T+2 T+3  ▓▓ ... ▓▓ T+18  V-14  ...  V-2 V-1 [V] V+1
                     └────── feature footprint of last train (X_T) ─────┘
                                              └── target SPEI3(T+3) ──┘
                                                                                  └── feature footprint of first val ──→

  precip-end(train_target)   = T + L = T + 3
  precip-start(val_feature)  = V − K_eff − 2 = (T + 18) − 14 = T + 4
                                                  T + 3   <   T + 4   →  no shared precip month  ✓
```

**Worked example B — precip lag dominates** (L = 3; SPEI3 PACF selects only lag 2; PACF on precip selects lag 12; NAO CCF selects lag 12 but that is independent of precip).
Per-variable contributions: SPEI3 → 2; precip → 12 − 2 = 10; NAO → ignored. *K*ₑff = max(2, 10) = 10 → gap = 3 + 10 + 2 = **15 months**.

**Worked example C — no precip-touching lags** (L = 3; PACF picks nothing for SPEI3 or precip; only NAO and RZSM lags survive). *K*ₑff = 0 → gap = 3 + 0 + 2 = **5 months** (the contemporaneous-SPEI3 floor).

**Compare with the naïve "max over all lags" choice** (the previous, conservative implementation):

| Configuration | Naïve *K*ₐₗₗ | Strict *K*ₑff | Gap @ L=3 (naïve) | Gap @ L=3 (strict) | Months saved |
| --- | --- | --- | --- | --- | --- |
| SPEI3 deep (12), others shallow | 12 | 12 | 17 | 17 | 0 |
| Precip lag 12, SPEI3 lag 2, NAO 12 | 12 | 10 | 17 | 15 | 2 |
| NAO/ENSO lag 12, no SPEI3 / precip lags | 12 | 0 | 17 | 5 | **12** |
| Empty selection | 0 | 0 | 5 | 5 | 0 |

The savings depend on how lag selection distributes across precip-touching vs independent variables, and can be substantial when teleconnection indices (NAO, ENSO, MO) drive the deep lags.

**Compare with `gap = K` alone** (a pre-fix bug): val feature deepest precip reach = *V* − *K* − 2 = *T* − 1, which **overlaps** the train target precip {*T* + 1, *T* + 2, *T* + 3}. The naïve "lead + 2" purge proposed by some authors covers only the SPEI3 rolling window of the contemporaneous feature and misses lag contamination entirely.

**Adaptive *K*ₑff.** *K*ₑff is determined per fold from PACF on the autoregressive variables + winter-only CCF on the climate indices, run over the **provisional** train slice; the per-variable lag dict is then mapped through the `compute_quarantine_max_lag` helper (precip-touching variables only) to obtain *K*ₑff. The gap `L + K_eff + 2` is applied and indices are finalized:

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
         train_indices = months in [train_start .. val_start  − gap − 1]
         val_indices   = months in [val_start   .. test_start − gap − 1]
         test_indices  = months in [test_start  .. test_end]   # never shrunk
    5. Fit fold-wise standardizer on train_indices only.
    6. Train; tune HPs on val_indices; predict on test_indices.
```

Test windows are **never** shrunk — the pooled out-of-sample stitching covers the continuous 2000-01 → 2024-12 span regardless of lead.

**Effective fold sizes for the worst case L = 3, *K*ₑff = 12 → gap = 17** (e.g. SPEI3 PACF selects lag 12). For lighter precip-touching lag selections the gap shrinks toward the *L* + 2 floor; the per-fold log emitted by the pipeline records both *K*ₑff and the resulting train/val sizes.

| Fold | Effective train | Effective val (HP tuning) | Test (5 y, unchanged) |
| --- | --- | --- | --- |
| 1 | 1950-01 → 1990-07 (40 y 7 mo) | 1992-01 → 1998-07 (6 y 7 mo) | 2000-01 → 2004-12 |
| 2 | 1950-01 → 1995-07 (45 y 7 mo) | 1997-01 → 2003-07 (6 y 7 mo) | 2005-01 → 2009-12 |
| 3 | 1950-01 → 2000-07 (50 y 7 mo) | 2002-01 → 2008-07 (6 y 7 mo) | 2010-01 → 2014-12 |
| 4 | 1950-01 → 2005-07 (55 y 7 mo) | 2007-01 → 2013-07 (6 y 7 mo) | 2015-01 → 2019-12 |
| 5 | 1950-01 → 2010-07 (60 y 7 mo) | 2012-01 → 2018-07 (6 y 7 mo) | 2020-01 → 2024-12 |

For *L* = 6 each train/val end shifts back another 3 months; for *L* = 1 each shifts forward 2 months relative to the table above. The cost (5 extra quarantined months per boundary vs the previous *K*-only gap) is small relative to the ~500+ months of train data per fold.

**Implementation.** Standard libraries (e.g. `sklearn.model_selection.TimeSeriesSplit`) do not support this two-sided reverse-gap logic. The pipeline implements:

- `droughtmodel.cv.compute_quarantine_max_lag(selected_lags)` — maps the per-variable lag dict to *K*ₑff using the strict precip-touching rule above.
- `droughtmodel.cv.RollingOriginCV.get_fold_indices(time_coord, fold, max_lag=K_eff, lead=L)` — returns integer index arrays (`train_idx`, `val_idx`, `test_idx`) with `gap = L + K_eff + 2` subtracted from the end of train and the end of val. Models receive the indices directly and only ever see the non-quarantined rows.

The behavior is overridable: setting `configs/cv.yaml::boundary_gap_months` to a positive integer forces that fixed gap for every fold and every lead (e.g. 20 as a worst-case covering *L* ≤ 6, *K*ₑff ≤ 12). A caller that does not want the per-variable accounting can pass any conservative integer for `max_lag` (e.g. the max over all selected lags); the function will compute `gap = L + max_lag + 2` as before — over-quarantining is safe, just data-inefficient. 

**5\. Standardization** 

Fold-wise standardization using **training-period statistics only**. 

**Pre-standardized exception list — do NOT re-standardize**: 

• ENSO, NAO, MO (and their lags) 

• **SPEI3 (and its lags)** when used as a predictor 

All other variables — **including RZSM** — are standardized fold-wise. 

Procedure per fold: 

1\. Compute mean / std on training data only 

2\. Apply to training, validation, and test sets 

**6\. Feature Selection (per fold)** 

A single selection method is used: **Lasso / Elastic Net** for linear models (selection happens during fitting via the L1 penalty). Tree-based models use no explicit selection — they tolerate redundant features natively. Permutation importance (RF) and SHAP (XGBoost) are computed for **interpretability and reporting only**, not used to drop features. 

Method Role Applied to 

**Lasso / Elastic Net** Selection (during fit via L1) Linear models — primary Permutation importance Diagnostic (post-hoc ranking) Random Forest 

5  
Method Role Applied to 

SHAP values Diagnostic (post-hoc ranking \+ interpretability) 

**Pipeline per model family**:   
XGBoost 

• **Linear models** (OLS / Ridge / Lasso / Elastic Net): All Section-3 features pass to the fitter; Lasso / Elastic Net’s L1 penalty produces the final sparse coefficient vector. Multicollinearity is absorbed by the L1 / L2 regularizer; no separate VIF pre-filter is applied. 

• **RF / XGBoost**: All Section-3 features pass to the fitter. After training, permutation importance (RF) and SHAP (XGBoost) are reported as paper diagnostics. 

Selection is repeated per fold (Lasso is refit per fold), preserving the leakage discipline. 

**7\. Model Configurations** 

**7.0 Baselines** 

Baseline Forecast formula Purpose 

Climatology per-cell per-calendar-month training mean   
Reference for skill scores 

Persistence SPEI3(t+L) \= SPEI3(t) Inertia benchmark 

AR(p) Linear regression on lagged SPEI3 only 

**7.1 ML Models**   
Univariate autoregressive benchmark (subsumes damped persistence at p \= 1\) 

Model Input Purpose 

Linear (OLS) Tabular Statistical baseline **Ridge** Tabular L2-regularized linear **Lasso** Tabular L1-regularized (also feature selector) 

**Elastic Net** Tabular L1 \+ L2 regularized Random Forest Tabular Nonlinear ensemble XGBoost Tabular Gradient boosting (primary tabular benchmark) 

LSTM Lagged sequences Temporal memory (deferred) CNN Spatial grids Spatial pattern extraction (deferred) 

CNN-LSTM / ConvLSTM Spatio-temporal sequences Joint spatio-temporal (deferred) 

**7.2 Modeling Unit** 

• **Default**: global model with (lat, lon) as features — one model per (model family, fold), trained on all 64 × 64 cells jointly 

• **Sensitivity test**: per-cell models for the best-performing family 

6  
**8\. Hyperparameter Tuning** 

**Protocol A** — tune on validation, refit on train \+ val, evaluate on test: 

for fold in cv\_folds: 

best\_hp \= search(model\_type, X\_train, y\_train, X\_val, y\_val) 

final\_model \= model\_type(best\_hp).fit(X\_train � X\_val, y\_train � y\_val) preds \= final\_model.predict(X\_test) 

**Search method**: 

• **Grid search is the primary method for all models** (Ridge, Lasso, Elastic Net, RF, XGBoost) — reproducible, transparent, and acceptable in cost given the reduced XGBoost grid below. • **Bayesian (Optuna)** is available as an **optional** alternative for RF and XGBoost (selectable via the experiment YAML), to be used if/when the grid becomes a bottleneck. 

**Search spaces (grid)**: 

Model Space Combinations Ridge / Lasso alpha � logspace(−3, 3, 13\) 13   
Elastic Net alpha � logspace(−3, 3, 7);   
35 

l1\_ratio � {0.1, 0.3, 0.5, 

0.7, 0.9} 

RF n\_estimators � {200, 500,   
108 

1000}; max\_depth � {None, 5, 

10, 20}; min\_samples\_leaf � 

{1, 5, 20}; max\_features � 

{sqrt, 0.5, 1.0} 

XGBoost (reduced grid) max\_depth � {4, 6, 8}; lr �   
144 

{0.05, 0.1}; subsample � 

{0.7, 1.0}; 

colsample\_bytree � {0.7, 

1.0}; reg\_lambda � {0.1, 

1.0, 10.0}; 

min\_child\_weight � {1, 5}; 

n\_estimators via early stopping 

on val 

The XGBoost space is intentionally compact (144 combos vs. 6,075 in the full v2 draft) so that full grid search remains tractable across 5 folds × 3 leads. Optuna can explore the full space if needed. 

**9\. Forecast Generation** 

For each issue month *t*: 

1\. Extract predictors up to month *t* 

2\. Apply trained model for lead *L* (one model per (family, fold, lead)) 

3\. Produce forecast SPÊI3(t \+ L) 

4\. Retain forecasts verifying in Nov–Feb only 

7  
**10\. Evaluation Metrics** 

**10.1 Deterministic (continuous SPEI3) — headline** 

Metric Formula Role 

**MAE** mean( ŷ − y 

**RMSE** √\[mean((ŷ − y)²)\] Headline — error magnitude **Pearson *r*** corr(ŷ, y) Headline — pattern correlation **ACC** corr(ŷ − clim, y − clim) Headline — anomaly correlation 

**MSSS vs climatology** 1 − MSE(model) / MSE(climatology) 

**MSSS vs persistence** 1 − MSE(model) / MSE(persistence)   
Headline — % improvement over climatology 

Headline — added value beyond inertia 

**10.2 Optional metrics (implemented in code, not reported by default)** 

• **HSS at SPEI3 \< −1.0** — binary categorical (drought / no drought). Available via the metrics config but not part of the v1 headline. 

• **POD, FAR, CSI, ETS, multi-class HSS** — not implemented in v1; deferred to a follow-up if categorical evaluation becomes a focus. 

• **Probabilistic (CRPS, Brier, reliability)** — deferred; requires quantile / probabilistic models. 

**10.3 Headline Metrics (committed)** 

The six deterministic metrics in §10.1 form the headline set. They are reported per (model, lead) on the **pooled** out-of-sample array (2000–2024 winter target months for the headline; all months for the supplementary table). 

**10.4 Reporting Structure** 

• Per lead time (L \= 1, 3, 6\) 

• Per CV fold (supplementary table) \+ pooled across folds (headline) 

• Spatial: per-cell skill maps \+ pooled metrics across cells 

• **Winter-only** evaluation (t in Nov–Feb) is the **headline** unit; **all-months** evaluation is reported as a supplementary diagnostic alongside (uses 4× more samples, lets us check whether skill is winter-specific or season-uniform) 

• **Block bootstrap 95 % CIs** on all headline metrics: 

**–** Stationary bootstrap; mean block � 12 months 

**–** For winter-only metrics: year-blocks (full Nov–Feb season per block) 

**–** 1000 replicates 

**11\. Outputs** 

Output Description 

Grid-level forecasts 0.5° SPEI3 predictions per (model, lead) Spatial skill maps Per-cell ACC, MSSS for each (model, lead) Forecast-vs-truth time series At selected cells (Casablanca, Marrakech, Agadir) Feature importance diagnostics Permutation (RF), SHAP (XGBoost), per lead 

8  
Output Description 

Skill comparison tables Models × leads × headline metrics, with bootstrap CIs (winter-only headline; all-months 

supplementary) 

Per-fold stability tables Same metrics broken by fold Winter-vs-all-months skill diagnostic Table comparing each model’s metrics on winter targets vs all-month targets, per lead 

Baseline-vs-ML skill plots Bars showing MSSS for each model relative to each baseline 

Lag selection diagnostics PACF / CCF plots per variable per fold Stationarity diagnostics Mann-Kendall and KS test results per cell 

**Summary of Experimental Structure (v3)** 

Dimension Specification 

Time period 1950–2024 

Temporal resolution Monthly 

Spatial resolution 64 × 64 grid at 0.5° over Morocco Target SPEI-3 at L \= 1, 3, 6 months Evaluation season Winter (Nov–Feb) 

Cross-validation 5-fold rolling-origin, expanding train window, **continuous test windows (2000–2024** 

**unbroken)** 

Leakage control Fold-wise standardization, feature selection, lag selection; strict target shifting; **boundary-gap** 

**quarantine (adaptive to per-fold max** 

**selected lag)** at train → val and val → test 

boundaries 

Pre-standardized exceptions ENSO, NAO, MO, **SPEI3 (as predictor)** Climate drivers ENSO, NAO, MO (+ optional AMO, AO in v2) Lag selection PACF \+ winter-only CCF on Morocco-masked spatial mean; threshold |·| \> 0.20; sensitivity � 

{0.10, 0.30}; Lasso finalizes for linear 

Region mask for lag selection shapefiles/MAR\_adm0.shp (164 cells inside) — Approach A (spatial mean) default; Approach B 

(per-cell then mean) reported as appendix 

sensitivity 

Modeling unit Global model with (lat, lon) features (v1); per-cell as sensitivity 

Baselines Climatology, persistence, AR(p) Feature selection Lasso / Elastic Net for linear (selection during fit); no explicit selection for trees; SHAP / permutation 

importance reported as diagnostics 

ML models OLS / Ridge / Lasso / Elastic Net, RF, XGBoost DL models LSTM, CNN, CNN-LSTM (or ConvLSTM) — deferred 

HP tuning Protocol A — tune on val, refit on train \+ val, eval on test 

9  
Dimension Specification 

HP search Grid search primary for all models (Ridge/Lasso/EN: 13–35 combos; RF: 108; 

XGBoost reduced grid: 144). Optuna optional for 

RF / XGBoost 

Headline metrics (all deterministic) MAE, RMSE, Pearson *r*, ACC, MSSS-vs-clim, MSSS-vs-persistence 

Optional metric HSS at SPEI3 \< −1.0 (available in code, not in v1 paper headline) 

Evaluation window Winter-only (Nov–Feb) headline; all-months supplementary diagnostic 

Uncertainty Stationary block bootstrap, 1000 replicates, 95 % CI 

Outputs Forecasts, spatial skill maps, time series, importance, skill tables, CIs 

10