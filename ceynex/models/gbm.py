"""Implements SRS 3.1.10 — the gradient-boosted model (SAD Figure 10).

`GradientBoostedModel` is the base class for anything M1 or M3 wants to fit with
features rather than with a pure time-series process — the obvious case being
cinnamon or tea price, where world commodity prices and FX carry information a
univariate model cannot see.

**Three quantile models, not one.** LightGBM's default objective predicts the
conditional mean and gives no interval, and `ForecastPoint` requires one. So
this fits the 10th, 50th and 90th percentiles separately with the `quantile`
objective. That is more honest than a mean prediction with a bootstrapped band
because the band is estimated from the data rather than assumed symmetric — real
export series have occasional large positive shocks and far fewer large negative
ones.

**It models year-on-year changes, not levels.** A regression tree predicts by
averaging training targets, so it can never return a value outside the range it
was trained on. Fitted on export values directly it produces a flat line the
moment the series trends upward past its own history — measured on a 10-year
trending series, MAPE 0.12 and interval coverage 0.00 against a nominal 0.80,
versus 0.02 for the SARIMA model on the same data. Differencing removes the
trend, so the tree predicts a *change* that is genuinely inside its training
range and the level is recovered by cumulating. This is the standard fix and it
is the difference between this class being useful and being decorative.

**Recursive forecasting.** Beyond one step the lag features are unknown, so each
predicted change is fed back in as the next step's lag. Errors compound; the
band widens because the per-step quantile changes accumulate, and the backtest at
horizon > 1 is what checks whether that widening is enough.

The cumulated band is an approximation: a sum of per-step quantiles is not the
quantile of the summed distribution. It errs narrow when errors are positively
correlated, which is why `coverage` is reported and not just MAPE.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ceynex.models.base import MIN_OBSERVATIONS, SeriesForecastModel

log = logging.getLogger(__name__)

DEFAULT_LAGS = (1, 2)
QUANTILES = (0.1, 0.5, 0.9)  # the 80% band `ForecastPoint` documents

# Ten annual observations against LightGBM's defaults (31 leaves, 100 rounds) is
# one leaf per observation and a model that memorises the series. These are the
# smallest settings that still boost.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "min_child_samples": 2,
    "verbose": -1,
}


class GradientBoostedModel(SeriesForecastModel):
    """LightGBM over lag and rolling-mean features, with quantile intervals.

    Usage::

        model = GradientBoostedModel(sector="apparel", item="apparel_knit").fit(df)
        points = model.predict(horizon=2)
    """

    def __init__(
        self,
        *,
        sector: str,
        item: str,
        target: str = "export_value_usd",
        unit: str = "USD",
        lags: tuple[int, ...] = DEFAULT_LAGS,
        params: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(sector=sector, item=item, target=target, unit=unit)
        self.lags = tuple(sorted(lags))
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._models: dict[float, Any] = {}

    def describe_params(self) -> dict[str, Any]:
        return {**super().describe_params(), "lags": list(self.lags), "params": dict(self.params)}

    # --- fitting ---------------------------------------------------------

    def _fit(self, periods: list[int], values: list[float]) -> None:
        features, targets = self._design_matrix(values)
        if len(targets) < MIN_OBSERVATIONS - max(self.lags):
            log.warning(
                "%s: %d training rows after building lag %s features — the model will "
                "be weak and the backtest should say so",
                self.item,
                len(targets),
                self.lags,
            )

        from lightgbm import LGBMRegressor

        self._models = {}
        for quantile in QUANTILES:
            model = LGBMRegressor(objective="quantile", alpha=quantile, **self.params)
            model.fit(features, targets)
            self._models[quantile] = model

    def _design_matrix(self, values: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Lagged *changes* plus a rolling mean change, predicting the next change.

        Every feature is strictly past-dated. A feature computed from the period
        being predicted would leak the future and produce an excellent,
        worthless backtest.
        """
        changes = _differences(values)
        window = max(self.lags)
        rows, targets = [], []
        for index in range(window, len(changes)):
            recent = changes[index - window : index]
            rows.append(
                [changes[index - lag] for lag in self.lags] + [float(np.mean(recent))]
            )
            targets.append(changes[index])
        return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)

    # --- prediction ------------------------------------------------------

    def _predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        window = max(self.lags)
        changes = _differences(self._values)
        level = self._values[-1]
        low_level = high_level = level
        points, lower, upper = [], [], []

        for _ in range(horizon):
            row = np.asarray(
                [[changes[-lag] for lag in self.lags] + [float(np.mean(changes[-window:]))]],
                dtype=float,
            )
            predicted = {q: float(model.predict(row)[0]) for q, model in self._models.items()}

            median = predicted[0.5]
            low, high = predicted[0.1], predicted[0.9]
            # Quantile models are fitted independently, so nothing forces
            # q10 <= q50 <= q90. On short series they do cross, and an interval
            # whose lower bound sits above its upper bound is worse than useless.
            low, high = min(low, median), max(high, median)

            # Cumulate changes back into levels. The band accumulates its own
            # per-step quantiles, so it widens with the horizon.
            level += median
            low_level += low
            high_level += high

            points.append(level)
            lower.append(low_level)
            upper.append(high_level)
            changes.append(median)  # recursive: the predicted change becomes the next lag

        return points, lower, upper


def _differences(values: list[float]) -> list[float]:
    """Year-on-year changes. One shorter than the series it came from."""
    return [values[i] - values[i - 1] for i in range(1, len(values))]


__all__ = ["DEFAULT_LAGS", "DEFAULT_PARAMS", "QUANTILES", "GradientBoostedModel"]
