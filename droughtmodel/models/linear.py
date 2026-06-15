"""Linear models (OLS / Ridge / Lasso / Elastic Net) — v3 §7.1.

Linear models receive the full Section-3 feature set (all data_vars in the
input dataset except `target`). For **Lasso** and **Elastic Net**, the L1
penalty does the feature selection (v3 §6) — no separate VIF / RFE pre-filter
is applied.

Features are expected to be already standardized by the upstream
`FoldStandardizer` (with the exception list — ENSO, NAO, MO, SPEI3, target
— passing through unchanged). The models themselves perform no internal
scaling, so coefficient values can be interpreted as standardized
contributions directly via `feature_importance()`.
"""

from __future__ import annotations

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from droughtmodel.models._tabular import TabularBaseModel


class LinearModel(TabularBaseModel):
    """Marker base class for OLS / Ridge / Lasso / Elastic Net wrappers.

    All shared fit/predict/feature_importance plumbing lives in
    :class:`TabularBaseModel`; subclasses only configure the sklearn estimator.
    """

    name = "linear"


class LinearRegressionModel(LinearModel):
    """OLS — no regularization (v3 §7.1)."""

    name = "ols"

    def __init__(self):
        super().__init__(LinearRegression())


class RidgeModel(LinearModel):
    """Ridge regression — L2 penalty (v3 §7.1).

    HP search grid (v3 §8): ``alpha ∈ logspace(-3, 3, 13)``.
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        super().__init__(Ridge(alpha=self.alpha, fit_intercept=True))


class LassoModel(LinearModel):
    """Lasso — L1 penalty; performs embedded feature selection (v3 §6, §7.1).

    HP search grid (v3 §8): ``alpha ∈ logspace(-3, 3, 13)``.
    """

    name = "lasso"

    def __init__(self, alpha: float = 0.01, max_iter: int = 10000):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        super().__init__(Lasso(alpha=self.alpha, max_iter=self.max_iter, fit_intercept=True))


class ElasticNetModel(LinearModel):
    """Elastic Net — L1 + L2 penalty (v3 §7.1).

    HP search grid (v3 §8): ``alpha ∈ logspace(-3, 3, 7);
    l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`` → 35 combos.
    """

    name = "elasticnet"

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5, max_iter: int = 10000):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        super().__init__(
            ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                fit_intercept=True,
            )
        )
