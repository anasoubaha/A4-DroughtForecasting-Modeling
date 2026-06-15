"""Tree-based models (Random Forest, XGBoost) — v3 §7.1.

Both models consume the full Section-3 feature set without any pre-filter
(trees tolerate redundant / collinear features natively, v3 §6). Feature
importance for paper diagnostics is computed *post-hoc* via the helpers in
:mod:`droughtmodel.selection` (permutation importance for RF, TreeSHAP for
XGBoost) — the ``.feature_importance()`` method here returns the model's
*internal* importance (Gini for RF, gain for XGBoost), which is useful for
sanity-checking but not the primary reporting unit.

XGBoost supports **early stopping** on the validation set: when ``val`` is
passed to ``fit()`` and ``early_stopping_rounds`` is set, ``n_estimators``
acts as an upper bound and the actual rounds are determined by the val
score (v3 §8: "n_estimators via early stopping on val").
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from droughtmodel.models._tabular import TabularBaseModel


class TreeBaseModel(TabularBaseModel):
    """Marker base class for tree-based wrappers."""

    name = "tree"


class RandomForestModel(TreeBaseModel):
    """Random Forest regressor (v3 §7.1).

    HP search grid (v3 §8):
        n_estimators     ∈ {200, 500, 1000}
        max_depth        ∈ {None, 5, 10, 20}
        min_samples_leaf ∈ {1, 5, 20}
        max_features     ∈ {"sqrt", 0.5, 1.0}
    → 108 combos.
    """

    name = "rf"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",
        n_jobs: int = -1,
        random_state: int | None = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        super().__init__(
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        )


class XGBoostModel(TreeBaseModel):
    """XGBoost regressor (v3 §7.1).

    HP search grid (v3 §8, reduced):
        max_depth        ∈ {4, 6, 8}
        learning_rate    ∈ {0.05, 0.1}
        subsample        ∈ {0.7, 1.0}
        colsample_bytree ∈ {0.7, 1.0}
        reg_lambda       ∈ {0.1, 1.0, 10.0}
        min_child_weight ∈ {1, 5}
    → 144 combos; ``n_estimators`` set by early stopping on val.

    When ``fit()`` is called with a ``val`` dataset and ``early_stopping_rounds``
    is set, training stops when the val ``rmse`` hasn't improved for that many
    rounds. Without val, the model trains to the full ``n_estimators`` cap.
    """

    name = "xgboost"

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        n_estimators: int = 2000,
        early_stopping_rounds: int | None = 50,
        eval_metric: str = "rmse",
        tree_method: str = "hist",
        n_jobs: int = -1,
        random_state: int | None = 42,
    ):
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.min_child_weight = int(min_child_weight)
        self.n_estimators = int(n_estimators)
        self.early_stopping_rounds = (
            int(early_stopping_rounds) if early_stopping_rounds else None
        )
        self.eval_metric = eval_metric
        self.tree_method = tree_method
        self.n_jobs = int(n_jobs)
        self.random_state = random_state
        super().__init__(
            XGBRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
                n_estimators=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric=self.eval_metric,
                tree_method=self.tree_method,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=0,
            )
        )

    def _fit_estimator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        if X_val is not None and y_val is not None and X_val.shape[0] > 0 and self.early_stopping_rounds:
            self.estimator.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            # No val (or early stopping disabled) → train to the full n_estimators cap.
            # XGBoost requires early_stopping_rounds to be None in this case.
            self.estimator.set_params(early_stopping_rounds=None)
            self.estimator.fit(X, y, verbose=False)

    @property
    def best_iteration(self) -> int | None:
        """Iteration of best val score after early stopping (``None`` if not used)."""
        return getattr(self.estimator, "best_iteration", None)
