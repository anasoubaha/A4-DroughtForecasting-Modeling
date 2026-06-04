"""Evaluation metrics and block bootstrap CIs (v2 §10).

Headline deterministic metrics:
- ``mae``                — Mean Absolute Error
- ``rmse``               — Root Mean Squared Error
- ``pearson_r``          — Pearson correlation
- ``acc``                — Anomaly Correlation Coefficient (corr of anomalies vs climatology)
- ``msss_vs_climatology``  — Mean Squared Skill Score vs climatology baseline
- ``msss_vs_persistence``  — Mean Squared Skill Score vs persistence baseline

Optional categorical metric (off by default in config):
- ``hss_binary`` — Heidke Skill Score on a binary drought classification (e.g. SPEI3 < −1.0)

Block bootstrap CIs for all of the above via stationary block bootstrap on the
time axis. For winter-only evaluation, set ``mean_block_length = 4`` (4 winter
months per year). For all-months evaluation, set ``mean_block_length = 12``.

Metric functions are pure NumPy and take time-aligned (possibly multi-dim)
arrays. NaN-safe — finite pairs are extracted before computation. The caller
is responsible for restricting inputs to Morocco cells when working in the
Morocco-only modeling scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from droughtmodel.utils import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_metrics_config(path: str | Path = "configs/metrics.yaml") -> dict[str, Any]:
    """Load the metrics config (paths anchored at PROJECT_ROOT)."""
    p = Path(path)
    p = p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Return flat finite-aligned copies of `a` and `b`."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _finite_triple(a, b, c) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    c = np.asarray(c).ravel()
    mask = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    return a[mask], b[mask], c[mask]


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

def mae(y_pred, y_true) -> float:
    """Mean Absolute Error."""
    yp, yt = _finite_pair(y_pred, y_true)
    if yp.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yp - yt)))


def rmse(y_pred, y_true) -> float:
    """Root Mean Squared Error."""
    yp, yt = _finite_pair(y_pred, y_true)
    if yp.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def pearson_r(y_pred, y_true) -> float:
    """Pearson correlation."""
    yp, yt = _finite_pair(y_pred, y_true)
    if yp.size < 2:
        return float("nan")
    if yp.std() == 0 or yt.std() == 0:
        return float("nan")
    return float(np.corrcoef(yp, yt)[0, 1])


def acc(y_pred, y_true, y_clim) -> float:
    """Anomaly Correlation Coefficient = corr(pred − clim, truth − clim)."""
    yp, yt, yc = _finite_triple(y_pred, y_true, y_clim)
    if yp.size < 2:
        return float("nan")
    pred_anom = yp - yc
    true_anom = yt - yc
    if pred_anom.std() == 0 or true_anom.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred_anom, true_anom)[0, 1])


def msss(y_pred, y_true, y_ref) -> float:
    """Mean Squared Skill Score vs reference.

    MSSS = 1 − MSE(pred) / MSE(ref).
    Returns NaN if MSE(ref) is zero (degenerate reference) or no finite samples.
    """
    yp, yt, yr = _finite_triple(y_pred, y_true, y_ref)
    if yp.size == 0:
        return float("nan")
    mse_model = float(np.mean((yp - yt) ** 2))
    mse_ref = float(np.mean((yr - yt) ** 2))
    if mse_ref == 0:
        return float("nan")
    return 1.0 - mse_model / mse_ref


def msss_vs_climatology(y_pred, y_true, y_clim) -> float:
    """% improvement over climatology forecast."""
    return msss(y_pred, y_true, y_clim)


def msss_vs_persistence(y_pred, y_true, y_pers) -> float:
    """% improvement over persistence forecast."""
    return msss(y_pred, y_true, y_pers)


# ---------------------------------------------------------------------------
# Categorical metric (optional)
# ---------------------------------------------------------------------------

def hss_binary(y_pred, y_true, threshold: float = -1.0) -> float:
    """Heidke Skill Score for binary drought classification.

    Class: drought (value < threshold) vs no-drought (value ≥ threshold).

    .. math::
        \\text{HSS} = \\frac{2(ad - bc)}{(a+c)(c+d) + (a+b)(b+d)}

    where ``a`` = hits, ``b`` = false alarms, ``c`` = misses, ``d`` = correct negatives.

    HSS ranges from −∞ to 1. HSS = 1 is perfect; HSS = 0 is no skill vs random;
    HSS < 0 means the forecast is worse than random.
    """
    yp, yt = _finite_pair(y_pred, y_true)
    yp_d = yp < threshold
    yt_d = yt < threshold
    a = int(np.sum(yp_d & yt_d))
    b = int(np.sum(yp_d & ~yt_d))
    c = int(np.sum(~yp_d & yt_d))
    d = int(np.sum(~yp_d & ~yt_d))
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    if denom == 0:
        return float("nan")
    return float(2 * (a * d - b * c) / denom)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    metric_fn: Callable,
    *args,
    n_replicates: int = 1000,
    mean_block_length: int = 12,
    ci: float = 0.95,
    time_axis: int = 0,
    seed: int | None = None,
) -> dict[str, float]:
    """Stationary block bootstrap CI for a metric on time-aligned arrays.

    All positional ``args`` must share the same length along ``time_axis``.
    Each bootstrap replicate resamples time indices block-wise and applies the
    same resampling to every input array (cells / spatial structure stay together
    at each time step).

    Parameters
    ----------
    metric_fn
        Function taking ``*args`` and returning a scalar.
    *args
        Time-aligned arrays. For multi-dim arrays (e.g. ``(time, cell)``),
        only the time axis is resampled.
    n_replicates
        Number of bootstrap replicates.
    mean_block_length
        Mean block length in **time steps**. Year-block on winter-only data = 4;
        year-block on all-months data = 12.
    ci
        Confidence level (e.g. 0.95 → 95 % CI).
    time_axis
        Axis to resample.
    seed
        Random seed (None for non-deterministic).

    Returns
    -------
    dict with keys: ``estimate``, ``lower``, ``upper``, ``std``, ``n_replicates``.
    """
    from arch.bootstrap import StationaryBootstrap

    arrays = [np.asarray(a) for a in args]
    n = arrays[0].shape[time_axis]
    for a in arrays[1:]:
        if a.shape[time_axis] != n:
            raise ValueError(
                f"All args must have the same length along time_axis={time_axis}; "
                f"got {[x.shape[time_axis] for x in arrays]}"
            )

    estimate = metric_fn(*arrays)

    indices = np.arange(n)
    bs = StationaryBootstrap(mean_block_length, indices, seed=seed)
    estimates = []
    for data_tuple, _kw in bs.bootstrap(n_replicates):
        idx_resampled = data_tuple[0]
        resampled = tuple(np.take(a, idx_resampled, axis=time_axis) for a in arrays)
        estimates.append(metric_fn(*resampled))

    estimates = np.array(estimates, dtype=float)
    alpha = (1 - ci) / 2
    return {
        "estimate": float(estimate),
        "lower": float(np.nanpercentile(estimates, 100 * alpha)),
        "upper": float(np.nanpercentile(estimates, 100 * (1 - alpha))),
        "std": float(np.nanstd(estimates)),
        "n_replicates": int(n_replicates),
    }


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

# Map of metric name → (function, list of required reference args after y_pred, y_true)
# References are looked up by name from the inputs dict in `evaluate()`.
_METRIC_TABLE: dict[str, tuple[Callable, list[str]]] = {
    "mae": (mae, []),
    "rmse": (rmse, []),
    "pearson_r": (pearson_r, []),
    "acc": (acc, ["climatology"]),
    "msss_vs_climatology": (msss_vs_climatology, ["climatology"]),
    "msss_vs_persistence": (msss_vs_persistence, ["persistence"]),
}


class MetricsReporter:
    """Compute the configured headline metrics (+ optional HSS) on a single (preds, truth) pair.

    Parameters
    ----------
    metrics
        List of metric names to compute. Defaults to v2 headline set.
    include_hss
        If True, also compute ``hss_binary`` at ``hss_threshold``.
    hss_threshold
        Drought threshold (default −1.0 per v2 §10.2).
    bootstrap
        If True, compute block-bootstrap CIs for every metric.
    mean_block_length
        Bootstrap block length in time steps (4 for winter-only, 12 for all-months).
    n_replicates, ci, seed
        Bootstrap settings.
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        include_hss: bool = False,
        hss_threshold: float = -1.0,
        bootstrap: bool = True,
        mean_block_length: int = 12,
        n_replicates: int = 1000,
        ci: float = 0.95,
        seed: int | None = None,
    ):
        self.metrics = metrics or list(_METRIC_TABLE.keys())
        self.include_hss = include_hss
        self.hss_threshold = hss_threshold
        self.bootstrap = bootstrap
        self.mean_block_length = mean_block_length
        self.n_replicates = n_replicates
        self.ci = ci
        self.seed = seed

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None = None,
        evaluation_window: str = "winter_only",
    ) -> "MetricsReporter":
        """Build a reporter from `configs/metrics.yaml`.

        ``evaluation_window`` selects which `mean_block_length` to use from the
        config (``mean_block_length_winter`` vs ``mean_block_length_all``).
        """
        config = config if config is not None else load_metrics_config()
        bs = config.get("bootstrap", {})
        if evaluation_window == "winter_only":
            block_len = bs.get("mean_block_length_winter", 4)
        elif evaluation_window == "all_months":
            block_len = bs.get("mean_block_length_all", 12)
        else:
            raise ValueError(
                f"evaluation_window must be 'winter_only' or 'all_months', got: {evaluation_window!r}"
            )
        return cls(
            metrics=config.get("headline_metrics"),
            include_hss=bool(config.get("include_hss", False)),
            hss_threshold=float(config.get("hss_threshold", -1.0)),
            bootstrap=bool(bs.get("enabled", True)),
            mean_block_length=int(block_len),
            n_replicates=int(bs.get("n_replicates", 1000)),
            ci=float(bs.get("ci", 0.95)),
            seed=bs.get("seed"),
        )

    def evaluate(
        self,
        y_pred,
        y_true,
        climatology=None,
        persistence=None,
    ) -> dict[str, Any]:
        """Compute every configured metric. Returns a dict of metric name → result.

        Each result is either a scalar (if ``bootstrap=False``) or a dict with
        ``estimate``, ``lower``, ``upper``, ``std``, ``n_replicates``.
        """
        refs = {"climatology": climatology, "persistence": persistence}
        results: dict[str, Any] = {}

        for name in self.metrics:
            if name not in _METRIC_TABLE:
                raise KeyError(f"Unknown metric: {name!r}")
            fn, req = _METRIC_TABLE[name]
            extra_args = []
            for r in req:
                if refs.get(r) is None:
                    raise ValueError(f"Metric {name!r} requires `{r}` reference series.")
                extra_args.append(refs[r])
            results[name] = self._evaluate_one(fn, y_pred, y_true, *extra_args)

        if self.include_hss:
            def _hss(yp, yt):
                return hss_binary(yp, yt, threshold=self.hss_threshold)

            _hss.__name__ = f"hss_binary@{self.hss_threshold}"
            results["hss_binary"] = self._evaluate_one(_hss, y_pred, y_true)

        return results

    def _evaluate_one(self, fn, *args):
        if self.bootstrap:
            return block_bootstrap_ci(
                fn,
                *args,
                n_replicates=self.n_replicates,
                mean_block_length=self.mean_block_length,
                ci=self.ci,
                seed=self.seed,
            )
        return float(fn(*args))

    # ------------------------------------------------------------------
    # Tabular output
    # ------------------------------------------------------------------

    @staticmethod
    def to_dataframe(
        results: dict[str, Any],
        model: str = "",
        lead: int | str = "",
        fold: int | str = "",
        evaluation_window: str = "",
    ) -> pd.DataFrame:
        """Flatten a `evaluate()` result dict to a tidy DataFrame."""
        rows = []
        for metric, val in results.items():
            base = {
                "model": model, "lead": lead, "fold": fold,
                "evaluation_window": evaluation_window, "metric": metric,
            }
            if isinstance(val, dict):
                rows.append({
                    **base,
                    "value": val.get("estimate"),
                    "ci_lower": val.get("lower"),
                    "ci_upper": val.get("upper"),
                    "std": val.get("std"),
                    "n_replicates": val.get("n_replicates"),
                })
            else:
                rows.append({
                    **base,
                    "value": val,
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "std": float("nan"),
                    "n_replicates": 0,
                })
        return pd.DataFrame(rows)
