# SPEI3 Drought Forecasting over Morocco

Monthly SPEI3 forecasts at 0.5° resolution over Morocco, 1950–2024, evaluated on winter months (Nov–Feb).
Models: climatology / persistence / damped persistence / AR baselines, regularized linear models, Random Forest, XGBoost (tabular ML). LSTM / CNN / CNN-LSTM deferred to a follow-up.

## Repository layout

```
inputs/        Raw NetCDFs (CRU, ERA5, climate indices, SPEI3) — Git-LFS tracked
data/          Derived / processed data (gitignored, regenerable from inputs/)
configs/       YAML configuration files (one source of truth per run)
droughtmodel/  Python package — data, features, CV, models, evaluation
scripts/       CLI entry points (preprocessing, feature build, experiment runs, reporting)
notebooks/     Exploratory analysis and presentation-only notebooks
tests/         Unit tests
results/       Predictions, metrics (kept), figures (gitignored), logs (gitignored)
docs/          Forecasting scheme documents
```

## Environment

```bash
conda env create -f environment.yml
conda activate droughtforecast
pip install -e .
pytest
```

## Workflow

1. `python scripts/01_preprocess.py` — align, handle missing values, run stationarity diagnostics
2. `python scripts/02_build_features.py` — build Dataset_L1 / Dataset_L3 / Dataset_L6
3. `python scripts/03_run_experiment.py --config configs/experiments/exp_NAME.yaml`
4. `python scripts/04_run_all_experiments.py` — sweeps every experiment config
5. `python scripts/05_generate_report.py` — compiles results into final tables and figures

See `docs/forecasting_scheme_v2.md` for the full methodology.
