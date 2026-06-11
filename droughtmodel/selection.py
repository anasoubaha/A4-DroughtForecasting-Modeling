"""Feature importance diagnostics (v3 §6).

Per v3 §6, the only "selection" step in this pipeline is embedded in
Lasso / Elastic Net fitting via the L1 penalty (no upstream pre-filter).
Tree-based models use **all** features. This module provides the two
**post-hoc diagnostics** used for interpretability and paper reporting:

  - `permutation_importance_scores`  — model-agnostic permutation importance
                                       (used for Random Forest in §7.1)
  - `tree_shap_importance`           — TreeSHAP mean(|SHAP|) per feature
                                       (used for XGBoost in §7.1)

Both return tidy DataFrames so Phase 11 notebooks can plot directly. SHAP
additionally returns the raw values array for richer visualizations
(summary plot, dependence plot).

Interpretation caveat — multicollinearity. Hydroclimate predictors are
heavily correlated (e.g. SPEI3_lag1, RZSM_lag1, VPD_lag1 all co-move).
Permutation importance under-attributes credit to such features: shuffling
one column barely hurts the score because the model recovers the same
information from its correlated siblings. SHAP (Shapley-value based)
distributes credit more fairly across correlated groups. Phase 11 reports
both side-by-side; large disagreements are the diagnostic signal, not a
bug. For the cleanest SHAP behavior under heavy correlation, switch
`shap.TreeExplainer` to `feature_perturbation="interventional"`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


# Higher score = better. Permutation importance is defined as
# `baseline_score - permuted_score`, so importance > 0 means shuffling
# the feature hurts predictions (i.e. the feature is informative).
SCORING_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "neg_mse": lambda y_true, y_pred: -float(np.mean((y_true - y_pred) ** 2)),
    "neg_mae": lambda y_true, y_pred: -float(np.mean(np.abs(y_true - y_pred))),
    "r2": lambda y_true, y_pred: 1.0 - float(np.sum((y_true - y_pred) ** 2))
        / max(float(np.sum((y_true - y_true.mean()) ** 2)), 1e-12),
}


def permutation_importance_scores(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    n_repeats: int = 10,
    scoring: str | Callable[[np.ndarray, np.ndarray], float] = "neg_mse",
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Model-agnostic permutation importance.

    For each feature column, the column is randomly permuted (``n_repeats``
    independent trials), and the resulting score is compared with the baseline
    score on the original X. The importance is ``baseline − permuted`` so that
    a positive value means shuffling hurts predictions (the feature carries
    information).

    Parameters
    ----------
    predict_fn
        Callable ``(n_samples, n_features) → (n_samples,)``. For a fitted
        sklearn estimator, pass ``model.predict``.
    X, y
        Feature matrix and target. Rows containing NaN in X or y are dropped
        before scoring.
    feature_names
        Names for the columns of X (length must equal X.shape[1]).
    n_repeats
        Number of independent shuffles per feature.
    scoring
        Key into :data:`SCORING_FUNCS` (``'neg_mse'``, ``'neg_mae'``, ``'r2'``)
        or a callable ``(y_true, y_pred) → float`` where higher is better.
    random_state
        Seed for the shuffler.

    Returns
    -------
    DataFrame with columns ``feature``, ``importance_mean``, ``importance_std``
    sorted by ``importance_mean`` descending.

    Notes
    -----
    Pass **out-of-sample** ``(X, y)`` (the fold's test split) for
    generalization-importance reporting. Permutation importance on training
    data measures how much the fit *relied* on each column — inflated by
    overfitting and not what we want in the paper. The pipeline orchestrator
    (Phase 10) is responsible for calling this on test-fold inputs only.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != X.shape[1] ({X.shape[1]})"
        )

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if X.shape[0] == 0:
        raise ValueError("No finite samples after NaN filtering.")

    score_fn = SCORING_FUNCS[scoring] if isinstance(scoring, str) else scoring
    baseline = score_fn(y, predict_fn(X))

    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    importances = np.zeros((n_features, n_repeats))
    for j in range(n_features):
        original_col = X[:, j].copy()
        for r in range(n_repeats):
            X[:, j] = rng.permutation(original_col)
            importances[j, r] = baseline - score_fn(y, predict_fn(X))
        X[:, j] = original_col  # restore

    df = pd.DataFrame({
        "feature": list(feature_names),
        "importance_mean": importances.mean(axis=1),
        "importance_std": importances.std(axis=1),
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def tree_shap_importance(
    tree_model,
    X: np.ndarray,
    feature_names: Sequence[str],
    max_samples: int | None = 5000,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """TreeSHAP mean(|SHAP|) feature importance for tree-based models.

    Uses :class:`shap.TreeExplainer` on (a sample of) X. Returns the
    mean-absolute-SHAP ranking AND the raw SHAP values array so the caller
    can also make summary/dependence plots.

    Parameters
    ----------
    tree_model
        Fitted tree model (XGBoost ``XGBRegressor``/``Booster``, sklearn
        ``RandomForestRegressor``, etc.) supported by ``shap.TreeExplainer``.
    X
        Feature matrix. NaN rows are dropped before sampling.
    feature_names
        Names for the columns of X.
    max_samples
        Cap on the number of rows used for SHAP (random subsample). ``None``
        uses all rows. TreeSHAP is exact but scales with samples × trees.
    random_state
        Subsampling RNG seed.

    Returns
    -------
    importance_df
        DataFrame with columns ``feature``, ``mean_abs_shap``, sorted desc.
    shap_values
        ``(n_sampled, n_features)`` array of raw SHAP values (kept for
        downstream summary / dependence plots).
    """
    import shap  # lazy import (~100 ms)

    X = np.asarray(X)
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != X.shape[1] ({X.shape[1]})"
        )

    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    if X.shape[0] == 0:
        raise ValueError("No finite samples after NaN filtering.")

    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        sel = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[sel]

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]   # single-output regression
    shap_values = np.asarray(shap_values)

    df = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return df, shap_values
