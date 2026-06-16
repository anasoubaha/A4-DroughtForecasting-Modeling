#!/usr/bin/env python3
"""Run one SPEI3-forecasting experiment from a YAML config.

Usage:
    python scripts/03_run_experiment.py --config configs/experiments/exp_default.yaml

Outputs (paths configurable in the experiment YAML):
    results/predictions/pooled_lead{L}.nc         — stitched OOS preds + truth
    results/metrics/metrics_lead{L}.csv           — headline metrics + bootstrap CIs
    results/logs/fold_runs.csv                    — one row per (fold, lead, model)
    results/logs/feature_status.csv               — one row per (fold, lead, model, feature)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/03_run_experiment.py …` to find the droughtmodel package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from droughtmodel.pipeline import ExperimentRunner


def main() -> int:
    ap = argparse.ArgumentParser(description="Run an SPEI3 forecasting experiment.")
    ap.add_argument("--config", required=True, help="Path to the experiment YAML.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-fold progress logs.")
    args = ap.parse_args()

    runner = ExperimentRunner(args.config, verbose=not args.quiet)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
