"""Hyperparameter tuning — Protocol A (v3 §8).

Protocol A:

    for fold in cv_folds:
        best_hp     = search(model_type, X_train, y_train, X_val, y_val)
        final_model = model_type(best_hp).fit(X_train ∪ X_val, y_train ∪ y_val)
        preds       = final_model.predict(X_test)

This module provides the per-fold ``search(...)`` and ``tune_and_refit(...)``
operations. ``grid_search`` is the primary backend (reproducible, transparent);
``optuna_search`` is the optional Bayesian alternative for RF / XGBoost
(lazy-imported so callers without ``optuna`` aren't penalised).

All searches use the **val slice** to score candidate HP combinations
(`higher is better`). The refit step concatenates train + val along the
time axis. For XGBoost, the search's ``best_iteration`` (from early stopping)
can be carried into the refit by passing ``refit_with_best_iteration=True``.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Type

import numpy as np
import pandas as pd
import xarray as xr

from droughtmodel.models.base import BaseModel
from droughtmodel.selection import SCORING_FUNCS


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Outcome of a hyperparameter search."""
    best_params: dict[str, Any]
    best_score: float
    all_scores: pd.DataFrame                # one row per HP combo, columns = HP names + 'score'
    best_model: BaseModel | None = None     # already fitted on train (NOT yet refit on train+val)
    duration_s: float = 0.0
    n_trials: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_grid(grid: dict[str, Iterable[Any]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand a `{name: [v1, v2, ...]}` grid into a list of dicts via cartesian product.

    If ``grid`` is already a list of dicts, it is returned unchanged so callers
    can supply pre-filtered combinations (e.g. excluding incompatible pairs).
    """
    if isinstance(grid, list):
        return grid
    if not grid:
        return [{}]
    names = list(grid.keys())
    value_lists = [list(grid[n]) for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def _resolve_scoring(scoring: str | Callable[[np.ndarray, np.ndarray], float]):
    if isinstance(scoring, str):
        return SCORING_FUNCS[scoring]
    return scoring


def _score_on_val(model: BaseModel, val: xr.Dataset, scoring) -> float:
    """Predict on val and return the score (higher = better). NaN-safe."""
    pred = model.predict(val).values
    y_true = val["target"].values
    mask = np.isfinite(y_true) & np.isfinite(pred)
    if not mask.any():
        return float("-inf")
    return float(scoring(y_true[mask], pred[mask]))


def _fit_one(
    model_class: Type[BaseModel],
    params: dict[str, Any],
    train: xr.Dataset,
    val: xr.Dataset | None,
    pass_val_to_fit: bool,
) -> BaseModel:
    model = model_class(**params)
    if pass_val_to_fit and val is not None:
        model.fit(train, val=val)
    else:
        model.fit(train)
    return model


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    model_class: Type[BaseModel],
    grid: dict[str, Iterable[Any]] | list[dict[str, Any]],
    train: xr.Dataset,
    val: xr.Dataset,
    *,
    scoring: str | Callable = "neg_mse",
    fixed_params: dict[str, Any] | None = None,
    pass_val_to_fit: bool = False,
    keep_best_model: bool = True,
    verbose: bool = False,
) -> SearchResult:
    """Exhaustive grid search over HP combinations.

    Parameters
    ----------
    model_class
        Constructor for the model under test (must implement BaseModel).
    grid
        Either a ``{name: [v, ...]}`` mapping (expanded via cartesian product)
        or a list of pre-built parameter dicts.
    train, val
        Training and validation datasets (xarray, must contain ``target``).
    scoring
        Key into :data:`droughtmodel.selection.SCORING_FUNCS` or a callable
        ``(y_true, y_pred) -> float``. Higher is better.
    fixed_params
        Parameters passed to *every* model constructor in addition to those
        coming from the grid (e.g. ``random_state``, ``n_jobs``).
    pass_val_to_fit
        If True, the model's ``fit(train, val=val)`` is called so the model
        can use the val set internally (XGBoost early stopping).
    keep_best_model
        If True, the best-fitted model is retained in the result. Set False
        to save memory when sweeping many folds.
    verbose
        If True, prints a one-line summary per combination.
    """
    fixed = fixed_params or {}
    combos = _expand_grid(grid)
    score_fn = _resolve_scoring(scoring)

    t0 = time.time()
    rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] = {}
    best_model: BaseModel | None = None

    for params in combos:
        full_params = {**fixed, **params}
        model = _fit_one(model_class, full_params, train, val, pass_val_to_fit)
        score = _score_on_val(model, val, score_fn)
        if verbose:
            print(f"  {params}  → {score:+.4f}")
        rows.append({**params, "score": score})
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model if keep_best_model else None

    return SearchResult(
        best_params=best_params,
        best_score=best_score,
        all_scores=pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True),
        best_model=best_model,
        duration_s=time.time() - t0,
        n_trials=len(combos),
    )


# ---------------------------------------------------------------------------
# Optuna (lazy, optional)
# ---------------------------------------------------------------------------

def optuna_search(
    model_class: Type[BaseModel],
    search_space: Callable[[Any], dict[str, Any]],
    train: xr.Dataset,
    val: xr.Dataset,
    *,
    n_trials: int = 50,
    scoring: str | Callable = "neg_mse",
    fixed_params: dict[str, Any] | None = None,
    pass_val_to_fit: bool = False,
    sampler_seed: int | None = 42,
    keep_best_model: bool = True,
    verbose: bool = False,
) -> SearchResult:
    """Bayesian (TPE) hyperparameter search via Optuna.

    Parameters
    ----------
    search_space
        Callable ``(trial: optuna.Trial) -> dict[str, Any]`` that uses the
        trial to suggest parameter values, e.g.::

            def space(trial):
                return {
                    "max_depth":     trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                }

    Other parameters mirror :func:`grid_search`.
    """
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "optuna is not installed. `pip install optuna` (or `conda install -c "
            "conda-forge optuna`) — or use grid_search instead."
        ) from e

    fixed = fixed_params or {}
    score_fn = _resolve_scoring(scoring)
    trials_log: list[dict[str, Any]] = []
    best_model_ref = {"model": None, "score": float("-inf"), "params": {}}

    def objective(trial):
        params = search_space(trial)
        full_params = {**fixed, **params}
        model = _fit_one(model_class, full_params, train, val, pass_val_to_fit)
        score = _score_on_val(model, val, score_fn)
        if verbose:
            print(f"  trial {trial.number}: {params}  → {score:+.4f}")
        trials_log.append({**params, "score": score})
        if score > best_model_ref["score"]:
            best_model_ref["score"] = score
            best_model_ref["params"] = params
            if keep_best_model:
                best_model_ref["model"] = model
        return score   # Optuna maximises by default given direction='maximize'

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return SearchResult(
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        all_scores=pd.DataFrame(trials_log).sort_values("score", ascending=False).reset_index(drop=True),
        best_model=best_model_ref["model"] if keep_best_model else None,
        duration_s=time.time() - t0,
        n_trials=len(trials_log),
    )


# ---------------------------------------------------------------------------
# Tune + refit (Protocol A end-to-end)
# ---------------------------------------------------------------------------

def tune_and_refit(
    model_class: Type[BaseModel],
    grid: dict[str, Iterable[Any]] | list[dict[str, Any]],
    train: xr.Dataset,
    val: xr.Dataset,
    refit_dataset: xr.Dataset,
    *,
    scoring: str | Callable = "neg_mse",
    fixed_params: dict[str, Any] | None = None,
    pass_val_to_fit: bool = False,
    refit_with_best_iteration: bool = False,
    search_fn: Callable = grid_search,
    search_kwargs: dict[str, Any] | None = None,
) -> tuple[BaseModel, SearchResult]:
    """Protocol A: tune on val, refit on a contiguous span, evaluate on test.

    Parameters
    ----------
    train, val
        Quarantined slices used for the HP search only. ``val`` is held out
        during the search so it can score candidate HP combinations.
    refit_dataset
        The contiguous time slice ``[train_start .. test_start − gap − 1]``
        used for the final refit. This **reclaims the train↔val quarantine
        gap** (which was only needed to hold val out during the search),
        leaving only the val→test quarantine at the right edge to protect
        the test set. The caller (Phase 10 orchestrator) is responsible for
        constructing this slice from the original full dataset.
    refit_with_best_iteration
        If True and the search's best model exposes ``best_iteration``
        (XGBoost early stopping), use that value as ``n_estimators`` for the
        refit and disable further early stopping. Use this for XGBoost so
        the refit uses the same number of trees that achieved the best val
        score.
    search_fn
        Which search backend to use: :func:`grid_search` (default) or
        :func:`optuna_search`. When using Optuna, pass the search space
        callable via ``search_kwargs={"search_space": fn, "n_trials": N}``
        and supply ``grid={}`` (it's ignored).

    Returns
    -------
    (final_model, search_result)
        ``final_model`` is fitted on ``refit_dataset``. ``search_result``
        carries the per-combo scores and timing for logging.
    """
    sk = dict(search_kwargs or {})
    sk.setdefault("scoring", scoring)
    sk.setdefault("fixed_params", fixed_params)
    sk.setdefault("pass_val_to_fit", pass_val_to_fit)

    if search_fn is grid_search:
        res = search_fn(model_class, grid, train, val, **sk)
    else:
        res = search_fn(model_class, train=train, val=val, **sk)

    refit_params: dict[str, Any] = {**(fixed_params or {}), **res.best_params}

    if refit_with_best_iteration and res.best_model is not None:
        bi = getattr(res.best_model, "best_iteration", None)
        if bi is not None and bi > 0:
            refit_params["n_estimators"] = int(bi) + 1   # 0-indexed → count
            refit_params["early_stopping_rounds"] = None

    final_model = model_class(**refit_params).fit(refit_dataset)
    return final_model, res
