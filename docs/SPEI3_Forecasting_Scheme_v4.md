# SPEI3 Forecasting Scheme — v4

> **v4 Δ from v3 — Winter-only training filter.** The only methodological change between v3 and v4 is the training-set composition. v3 trains on all 12 calendar months of each year and evaluates winter-only (Nov–Feb). v4 **filters the training set to rows whose lead-shifted target falls in the evaluation season (Nov–Feb)**, dropping ≈ 66 % of training samples in favour of a model that concentrates its capacity on the winter forecasting task ("winter-expert" specialisation). All other methodology — features, cross-validation, boundary-gap quarantine, standardization mechanics, model families, HP-search backends, and evaluation metrics — is identical to v3. See §3.3 for the precise filter and the changes that flow downstream into §5 (standardization) and §10.4 (the all-months diagnostic is now expected to be degraded).
>
> Pipeline switch: `winter_only_training: true` in the experiment YAML (`configs/experiments/exp_winter-training.yaml`). Outputs land under `results/winter-training/…` to keep them separate from the v3 default sweep.

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

- **Training (v4)**: rows where target month ∈ [Nov–Feb] only — see §3.3 for the precise winter-only training filter and its motivation.
- **Evaluation**: target month ∈ [Nov–Feb] only (unchanged from v3).

### 3.2 Predictor Set at Forecast Issue Time *t*

**(A) Contemporary predictors (time *t*)**

`Precip, Tmax, Tmin, PET, VPD, Wind, Solar, TCWV, RZSM, ENSO, NAO, MO, SPEI3`

**(B) Lag selection methodology**

- **PACF as primary tool** (not ACF), computed per CV fold on training data only.
- Test lags 1–12 uniformly for **all variables** — let the data prune.
- Significance threshold: **|PACF| > 0.20**, with a sensitivity test at 0.10 and 0.30 reported in appendix. This is much stricter than 95 % statistical significance (Bartlett's threshold = 1.96/√N ≈ 0.065 for N = 900, ≈ 0.113 for N = 300 winter-only); 0.20 is a deliberate practical-relevance filter rather than a significance filter.
- **CCF screening** between SPEI3 at time *t* and each candidate predictor at time *t − k*, restricted to **target months t ∈ {Nov, Dec, Jan, Feb}** since that is the evaluation season. Predictors whose CCF crosses |CCF| > 0.20 at any lag 1..12 are included at that lag.
- **Selected lags = union of PACF-passing lags and CCF-passing lags.** Variables with no surviving lags are dropped from the lagged feature set (the contemporary feature is unaffected).

**v4 note on lag selection.** Lag selection runs on the **full continuous** training slice (not the winter-filtered slice). PACF measures the structural autocorrelation of each variable — a property of the time series, not of the target season — so filtering would break its interpretation. The CCF was already winter-only on the target side in v3. Therefore lag selection is **unchanged** in v4; only model training is filtered.

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
- **v4 note**: in winter-only training the variance of these two features collapses dramatically — the model sees only 4 of the 12 calendar months on the target side and a narrow band of feature-side months (see §3.3). Trees no longer "waste" splits separating summer from winter; full tree depth goes to the subtle teleconnection signals that drive winter SPEI3.

**(D) Spatial encoding**

- `latitude`, `longitude` as features for the **global model** (Section 7.2).

### 3.3 Winter-only Training Filter (v4)

The v4 specialisation drops every row from the training (and validation, and refit) slice whose **lead-shifted target** falls outside Nov–Feb. The test slice is **not** filtered — predictions are still produced for every test month, so the all-months supplementary evaluation can document the cost of specialisation on off-season targets.

**Calendar-month consequences per lead.** Because the model trains only on rows where target month ∈ {Nov, Dec, Jan, Feb}, the **feature months** seen at training time are also restricted to a narrow contiguous band per lead:

| Lead L | Target months | Feature months seen at time *t* | Climate-physics narrative |
| --- | --- | --- | --- |
| 1 | Nov, Dec, Jan, Feb | Oct, Nov, Dec, Jan | Late-autumn / early-winter conditions (recent soil moisture, recent SPEI3) drive next-month drought status. |
| 3 | Nov, Dec, Jan, Feb | Aug, Sep, Oct, Nov | Late-summer / autumn teleconnections (ENSO state in Aug–Sep, autumn NAO regime) set the stage for upcoming winter precipitation. |
| 6 | Nov, Dec, Jan, Feb | May, Jun, Jul, Aug | Early-summer oceanic anomalies (MO, NAO, ENSO summer states) project onto the following winter's hydroclimate. |

**Sample-size impact.** Training drops from 12 months/year to 4 months/year — ≈ 66 % reduction per fold. With 164 Morocco cells × ~40–60 years of training data per fold, this still leaves tens of thousands of pooled rows, which is comfortable for RF and XGBoost. The signal-to-noise ratio improves because every retained sample is directly relevant to the evaluation task.

**Where the filter lives in the pipeline.** Step 5b in the §6.1 diagram, between the per-fold train/val/test slicing (step 4–5) and the standardizer fit (step 6). Concretely: the `train`, `val`, and `refit` slices consumed by **ML models** are passed through `_filter_to_winter_targets()` in `droughtmodel/pipeline.py`; `test` is left as-is.

**Baselines bypass the filter.** Climatology, persistence, and AR(p) are reference baselines whose role is to be the fair comparison against which ML skill is judged. Filtering their training data to winter targets only would make them no longer meaningful references — a winter-only climatology has no values for off-season target months, breaking the all-months evaluation outright. Baselines therefore train on the **full unfiltered** per-fold slices (`train_full`, `val_full`, `refit_full` in `_PreparedFold`), even when `winter_only_training=true`. This keeps the v3 ↔ v4 MSSS-vs-climatology and MSSS-vs-persistence comparisons honest.

**Effect on §5 (standardization).** Because the standardizer fits AFTER the winter filter, the (μ, σ) statistics are now computed from the winter-target-only training subset. Mathematically purer: we standardise based on the empirical distribution of conditions the model will see at fit time (late-summer to mid-winter feature months), not on the full 12-month climatology.

**Specialisation is intentional.** A winter-expert model **should** underperform on July targets — that's the trade-off we're buying with the higher signal-to-noise ratio on the headline (winter) task. In the paper this needs to be stated explicitly: the v4 models are specialised "Winter Expert" models, not general-purpose SPEI3 forecasters. The all-months supplementary evaluation (§10.4) quantifies this specialisation cost.

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

The "planned" sizes above are subsequently shrunk by a per-fold **boundary gap** (quarantine) — see §4.1 — **and then** further reduced ~66 % by the §3.3 winter-only training filter. The test window is unaffected by the filter (only training).

**Leakage controls**:

- Target shifting performed before splitting; no future predictors.
- Per-fold standardization (Section 5).
- Per-fold feature selection (Section 6).
- Per-fold PACF / CCF lag selection (Section 3.2 B).
- **Per-fold boundary gap (quarantine) — Section 4.1.**

**Aggregation across folds**:

- **Pooled**: concatenate predictions across folds → compute headline metric once. With the continuous-test design this is exactly a single, unbroken 25-year out-of-sample array (100 winter months × cells). Pooled metrics are the primary reporting unit.
- **Per-fold**: same metrics reported per fold in supplementary table to show stability. Per-fold winter sample is small (20 months), so per-fold metrics are noisy by design; do not over-interpret single-fold differences.
- **Winter-only and all-months reporting**: the headline metric is computed on **winter target months only** (t ∈ {Nov, Dec, Jan, Feb}). A parallel **all-months evaluation** is computed on every test month and reported alongside as a supplementary diagnostic — it uses 4× more evaluation samples and tests whether skill is uniform across seasons or concentrated in the winter target. **In v4, the all-months diagnostic is expected to be degraded** because the model has not seen any non-winter training targets; this is the specialisation cost quantified explicitly.

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

For *L* = 6 each train/val end shifts back another 3 months; for *L* = 1 each shifts forward 2 months relative to the table above. The cost (5 extra quarantined months per boundary vs the previous *K*-only gap) is small relative to the ~500+ months of train data per fold — and the winter-only filter further reduces train counts to ~33 % of these effective sizes (§3.3).

## 5. Standardization

Fold-wise standardization using **training-period statistics only**. The (μ, σ) statistics are always computed from the training slice exclusively (no leakage from val or test), and are always restricted to **Morocco cells** via the `MAR_adm0.shp` shapefile mask (164 of 4096 grid cells), so that non-Morocco grid points (e.g. Saharan fringe with extreme aridity distortion) don't bias the regional statistics.

**v4 update.** Because the §3.3 winter-only filter is applied to the training slice **before** the standardizer fits, the (μ, σ) statistics in v4 are computed from the winter-target-only training subset. Mathematically the model standardises based on the empirical distribution of conditions it will actually see during fitting (late-summer through mid-winter feature months for L ∈ {1, 3, 6} respectively, per the table in §3.3), not on the full 12-month climatology. The transform is still applied to all four slices (train, val, refit, test) using those winter-fit statistics.

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

Both variants compute statistics from training data only (after the v4 winter filter, if active), apply the Morocco mask, and respect the same exception list. Non-Morocco cells in the `PerCellStandardizer` pass through unchanged (identity transform).

Procedure per fold:

1. Compute mean / std on training data only (winter-filtered in v4; variant-dependent scope).
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

**v4 addition: step 5b applies the §3.3 winter-only filter to train / val / refit (not test) before standardization.**

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
   | 5b. WINTER-ONLY TRAINING FILTER  (Sec.3.3 - v4 only)             |
   |     Drop rows from train / val / refit whose lead-shifted        |
   |     target month is not in {Nov, Dec, Jan, Feb}.                 |
   |     APPLIES TO ML MODELS ONLY (linear + trees). Baselines        |
   |     (climatology, persistence, AR) keep the unfiltered full      |
   |     slices so they remain fair v3-comparable references.         |
   |     Test slice is INTENTIONALLY left unfiltered so the           |
   |     all-months diagnostic (Sec.10.4) can quantify the cost of    |
   |     specialisation on off-season targets.                        |
   +------------------------------------------------------------------+
                                  |
                                  v
   +------------------------------------------------------------------+
   | 6. PER-FOLD STANDARDIZATION  (Sec.5)                             |
   |    Fit (mu, sigma) on train_idx ONLY (winter-filtered in v4),    |
   |    Morocco mask applied.  Apply to train / val / test / refit.   |
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

Steps 1–6 (now 1–6, with new 5b in v4) are pure per-fold preprocessing; steps 7–10 are per (fold, model_family, lead). Each fold is independent — the orchestrator (Phase 10) parallelises the outer loop trivially.

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

- **Linear models** (Ridge, Lasso, Elastic Net) — **grid search**. Their spaces are small (13–35 combos) and exhaustive enumeration is reproducible and cheap.
- **Tree models** (RF, XGBoost) — **Optuna TPE** with `n_trials=40` per (fold, lead). The categorical search spaces are 72 and 144 respectively; sampling at `n_trials=40` cuts tree wall time by ~3× compared with exhaustive grid while landing on near-best HPs (TPE focuses sampling on high-skill regions after a brief uniform-exploration phase). Grid is still available for either backend.
- **RF `n_estimators` trimmed to `{200, 500}`** (dropping 1000): the n=1000 configs dominate per-trial cost without meaningful gain in best-HP quality on Morocco-scale data (5 folds × ~80k pooled samples per fit). The trim cuts RF's average per-trial cost ~30–40 %; combined with Optuna's `n_trials=40` budget, this reduces total tree wall time from ~60–80 h (full grid) to ~9 h on a typical laptop.
- **v4 cost note.** Training on ~33 % of the original rows reduces per-trial fit time roughly proportionally for RF and XGBoost. The full v4 sweep is therefore expected to take ~3–5 h rather than ~9 h.

**Search spaces**:

| Model | Space | Backend | Sampled |
| --- | --- | --- | --- |
| Ridge / Lasso | `alpha ∈ logspace(−3, 3, 13)` | grid | all 13 |
| Elastic Net | `alpha ∈ logspace(−3, 3, 7); l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` | grid | all 35 |
| RF | `n_estimators ∈ {200, 500}; max_depth ∈ {None, 5, 10, 20}; min_samples_leaf ∈ {1, 5, 20}; max_features ∈ {sqrt, 0.5, 1.0}` | **Optuna TPE** | 40 of 72 |
| XGBoost | `max_depth ∈ {4, 6, 8}; lr ∈ {0.05, 0.1}; subsample ∈ {0.7, 1.0}; colsample_bytree ∈ {0.7, 1.0}; reg_lambda ∈ {0.1, 1.0, 10.0}; min_child_weight ∈ {1, 5}`; `n_estimators` via early stopping on val | **Optuna TPE** | 40 of 144 |


## 9. Forecast Generation

For each issue month *t*:

1. Extract predictors up to month *t*.
2. Apply trained model for lead *L* (one model per (family, fold, lead)).
3. Produce forecast SPEI3(t + L).
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
- Pooled across folds (headline). Per-fold supplementary table is a planned addition — currently the pipeline emits one `fold='pooled'` row per (model, lead, metric, evaluation_window).
- Spatial: per-cell skill maps + pooled metrics across cells.
- **Winter-only** evaluation (t ∈ Nov–Feb) is the **headline** unit; **all-months** evaluation is reported as a supplementary diagnostic alongside (uses 4× more samples, lets us check whether skill is winter-specific or season-uniform).
- **v4 interpretation of the all-months diagnostic.** Because v4 models never saw a non-winter target during training, the all-months metric is **expected to be substantially worse** than the winter-only metric. The size of that gap quantifies the cost of specialisation. For the paper, the v4 row in the all-months table is reported as a "specialisation diagnostic" rather than a competing skill claim — readers should know they're looking at a winter-expert model being asked to do something it wasn't trained for.
- **Block bootstrap 95 % CIs** on all headline metrics:
  - Stationary bootstrap; mean block ≈ 12 months.
  - For winter-only metrics: year-blocks (full Nov–Feb season per block).
  - 1000 replicates.

## 11. Outputs

The pipeline (`ExperimentRunner` in `droughtmodel/pipeline.py`) produces a fixed set of artefacts per experiment. Output filenames are prefixed by `exp.file_prefix` (default `""`); v4 uses an empty prefix but a distinct output subfolder (`results/winter-training/…`) so the v4 sweep does not overwrite the v3 default sweep.

**Pipeline artefacts** (written by `scripts/03_run_experiment.py`):

| File | v3 default path | v4 winter-training path |
| --- | --- | --- |
| Predictions | `results/predictions/pooled_allLeads.nc` | `results/winter-training/predictions/pooled_allLeads.nc` |
| Headline metrics | `results/metrics/metrics_allLeads.csv` | `results/winter-training/metrics/metrics_allLeads.csv` |
| Fold-runs log | `results/logs/fold_runs.csv` | `results/winter-training/logs/fold_runs.csv` |
| Feature-status log | `results/logs/feature_status.csv` | `results/winter-training/logs/feature_status.csv` |
| Persisted models (if `save_models: true`) | `results/models/{name}_lead{L}_fold{F}.joblib` | `results/winter-training/models/{name}_lead{L}_fold{F}.joblib` |

The NetCDF and CSV schemas are identical between v3 and v4 — only the file locations differ.

**Phase 11 presentation-layer artefacts** (built by the results notebooks under `notebooks/`):

| Output | Notebook | Description |
| --- | --- | --- |
| Headline skill tables (CSV + LaTeX) | `09_paper_figures.ipynb` | Pooled metrics with CIs + significance markers, ready for direct paper inclusion |
| Per-cell ACC and MSSS maps | `07_spatial_skill_maps.ipynb` | Model × lead grids; latitudinal skill profile by quartile band |
| Difference maps (ML − baseline) | `07`, `09` | Where each ML model adds value over the strongest baseline |
| L1 retention heatmaps | `08_feature_importance.ipynb` | ElasticNet / Lasso retention across folds per (lead, feature) |
| Top-feature bar charts | `08`, `09` | Built-in tree importance (Gini / gain) and post-hoc permutation / SHAP |
| Lift-over-best-baseline table | `06_results_tabular_ml.ipynb` | Per-(model, lead, metric) lift over the strongest baseline |
| Lag-selection diagnostics | `02_feature_engineering_diagnostics.ipynb` | PACF / CCF plots per variable per fold |
| Stationarity diagnostics | `01_eda.ipynb` | Mann-Kendall and KS-test results per cell |

**Note on feature importance.** `feature_status.csv` carries both the **built-in** importance for each model (standardized coefficients for linear, Gini for RF, gain for XGBoost) and, after running `scripts/05_compute_posthoc_importance.py`, **post-hoc** importance (permutation for RF, TreeSHAP for XGBoost) on the out-of-sample test slice. Notebook 09 Figure 4 uses the built-in numbers; Figure 6 uses the post-hoc numbers and is the recommended primary reporting unit when present.

## Summary of Experimental Structure (v4)

| Dimension | Specification |
| --- | --- |
| Time period | 1950–2024 |
| Temporal resolution | Monthly |
| Spatial resolution | 64 × 64 grid at 0.5° over Morocco |
| Target | SPEI-3 at L = 1, 3, 6 months |
| Evaluation season | Winter (Nov–Feb) |
| **Training filter (v4 Δ vs v3)** | **Winter-only: rows where target month ∈ [Nov, Feb] only; ≈66 % training-row reduction; test slice unfiltered** |
| Cross-validation | 5-fold rolling-origin, expanding train window, **continuous test windows (2000–2024 unbroken)** |
| Leakage control | Fold-wise standardization, feature selection, lag selection; strict target shifting; **boundary-gap quarantine `gap = L + K_eff + 2`** (precip-touching variables only) at train → val and val → test boundaries |
| Pre-standardized exceptions | ENSO, NAO, MO, **SPEI3 (as predictor)**, target |
| Climate drivers | ENSO, NAO, MO (+ optional AMO, AO) |
| Lag selection | PACF + winter-only CCF on Morocco-masked spatial mean; threshold \|·\| > 0.20; sensitivity ∈ {0.10, 0.30}; Lasso finalizes for linear. **Run on full (unfiltered) train slice** — PACF is a property of the series, not the target season |
| Region mask for lag selection | `shapefiles/MAR_adm0.shp` (164 cells inside) — Approach A (spatial mean) default; Approach B (per-cell then mean) reported as appendix sensitivity |
| Modeling unit | Global model with (lat, lon) features (v1); per-cell as sensitivity |
| Standardization variants | `FoldStandardizer` (pooled over Morocco, default for global model); `PerCellStandardizer` (per (lat, lon), for §7.2 sensitivity). **Fits AFTER the v4 winter-only training filter** |
| Baselines | Climatology, persistence, AR(p) |
| Feature selection | Lasso / Elastic Net for linear (selection during fit); no explicit selection for trees; SHAP / permutation importance reported as diagnostics |
| ML models | OLS / Ridge / Lasso / Elastic Net, RF, XGBoost |
| DL models | LSTM, CNN, CNN-LSTM (or ConvLSTM) — deferred |
| HP tuning | Protocol A — tune on val, refit on train + val, eval on test |
| HP search | Linear models: grid (Ridge/Lasso 13, ElasticNet 35). RF: Optuna TPE, `n_trials=40` over a 72-combo categorical space (`n_estimators` trimmed to {200, 500}). XGBoost: Optuna TPE, `n_trials=40` over a 144-combo categorical space; `n_estimators` via early stopping on val |
| Headline metrics (all deterministic) | MAE, RMSE, Pearson *r*, ACC, MSSS-vs-clim, MSSS-vs-persistence |
| Optional metric | HSS at SPEI3 < −1.0 (available in code, not in v1 paper headline) |
| Evaluation window | Winter-only (Nov–Feb) headline; all-months **specialisation diagnostic** (v4 expected to be degraded — this is the point) |
| Uncertainty | Stationary block bootstrap, 1000 replicates, 95 % CI |
| Outputs | Forecasts, spatial skill maps, time series, importance, skill tables, CIs — all under `results/winter-training/…` in v4 |