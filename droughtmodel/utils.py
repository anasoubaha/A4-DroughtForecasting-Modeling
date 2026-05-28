"""Shared utilities — path resolution, figure saving, logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def save_figure(
    fig: matplotlib.figure.Figure,
    name: str,
    subdir: str | None = None,
    dpi: int = 200,
    formats: Iterable[str] = ("png",),
    bbox_inches: str | None = "tight",
) -> Path:
    """Save a matplotlib figure to `results/figures/<subdir>/<name>.<fmt>`.

    Creates the destination directory if missing. Returns the directory path.
    """
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    return out_dir
