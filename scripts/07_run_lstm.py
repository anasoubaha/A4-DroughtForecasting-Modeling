#!/usr/bin/env python3
"""Run the Phase 12 LSTM experiment from a YAML config.

Usage:
    python scripts/07_run_lstm.py --config configs/experiments/exp_lstm.yaml

Outputs (paths configurable in the experiment YAML):
    results/lstm/predictions/pooled_allLeads.nc
    results/lstm/metrics/metrics_allLeads.csv
    results/lstm/logs/fold_runs.csv
    results/lstm/logs/lstm_grid_search_L{L}.csv     (one per lead)
    results/lstm/logs/lstm_locked_combos.json
    results/lstm/models/lstm_lead{L}_fold{F}.joblib (if save_models: true)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from droughtmodel.lstm_pipeline import LSTMExperimentRunner


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Phase 12 LSTM experiment.")
    ap.add_argument("--config", required=True, help="Path to the LSTM experiment YAML.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-fold progress logs.")
    args = ap.parse_args()

    runner = LSTMExperimentRunner(args.config, verbose=not args.quiet)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())