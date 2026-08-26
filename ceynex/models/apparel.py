"""Apparel value forecasting model (SRS 3.1.3, team plan Day 5's "Apparel value
model", cut-order item `apparel_volume model`).

Naive last-value-carried-forward baseline, not a fitted time-series model —
deliberately, per the risk register's own R3 mitigation: "<40 observations ->
drop to annual + naive/seasonal-naive baseline and report honestly." The real
EDB Apparel sub-category series `apparel_manufacturing.py`'s own Cypher queries
(`_PARTNER_QUERY`, `LIMIT 5`) is 5 annual points — an order of magnitude under
that threshold. Seasonal-naive doesn't apply (annual data has no sub-period to
be seasonal over), so plain naive is the honest choice: fitting anything with
more than one degree of freedom to 5 points overfits the very trend it claims
to extrapolate, and a flat carry-forward makes no claim it can't support.

No published Sri Lankan apparel-export forecasting benchmark exists to beat
(unlike tea/cinnamon — see .claude/commands/backtest.md), so the standard here
is internal: does the reported interval actually contain what happened.

Deliberately self-contained, not routed through a shared model registry or
`eval/backtest.py` — that shared rolling-origin harness (team overview 4.4,
`.claude/commands/backtest.md`) is M2/core-systems scope and doesn't exist yet;
`ForecastModel.backtest()` is a per-model contract method regardless, so this
model can honestly self-report MAPE/RMSE/coverage without waiting on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ceynex.contracts.forecast import ForecastPoint
from ceynex.contracts.protocols import ForecastModel

MIN_OBSERVATIONS_FOR_FORECAST = 3
"""Below this, backtest() has fewer than 2 rolling-origin folds -- not enough
to honestly report a MAPE/coverage, so callers should not forecast at all."""

_INTERVAL_LEVEL = 0.80  # matches ForecastPoint's documented default band
_BOOTSTRAP_DRAWS = 2000
_RNG_SEED = 7  # fixed so predict()/backtest() are reproducible across runs


def _bootstrap_interval(
    point: float, residuals: np.ndarray, horizon: int, rng: np.random.Generator
) -> tuple[float, float]:
    """80% interval for an h-step-ahead naive forecast, by resampling observed
    one-step errors rather than assuming a distribution a 5-point series can't
    support (SRS 3.1.3 requires an interval; contracts/forecast.py requires
    `lower`/`upper`, so "not enough data for one" is not an option).
    """
    if len(residuals) >= 2:
        draws = rng.choice(residuals, size=(_BOOTSTRAP_DRAWS, horizon), replace=True).sum(axis=1)
        lo_pct, hi_pct = (1 - _INTERVAL_LEVEL) / 2 * 100, (1 + _INTERVAL_LEVEL) / 2 * 100
        lower = point + float(np.percentile(draws, lo_pct))
        upper = point + float(np.percentile(draws, hi_pct))
    else:
        # Fewer than 2 residuals to resample: a deliberately wide, clearly
        # arbitrary +-25% band rather than a fabricated statistical one.
        lower, upper = point * 0.75, point * 1.25

    # A mostly-one-directional residual sample (a partner series that has only
    # grown, or only shrunk) can push a multi-step bootstrap sum's 10th/90th
    # percentile past the point estimate itself -- the point is a flat carry
    # forward that deliberately does not extrapolate that trend (class
    # docstring), so the interval built from it may not implicitly do so
    # either. Same crossing guard gbm.py applies to its own independently-fit
    # quantiles.
    lower, upper = min(lower, point), max(upper, point)
    return max(0.0, lower), upper


class NaiveApparelForecastModel(ForecastModel):
    """Last-value-carried-forward baseline with a residual-bootstrap interval.

    `item` should identify the single series being forecast (e.g. a partner
    iso3 like "USA") -- this model forecasts one annual series, it does not
    aggregate across partners itself.
    """

    sector = "apparel"
    target = "export_value_usd"

    def __init__(self, item: str):
        self.item = item
        self._years: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._residuals: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> NaiveApparelForecastModel:
        """`df` needs integer-parseable `period` (year) and float `value` columns,
        one row per year. Order doesn't matter -- sorted here."""
        ordered = df.sort_values("period").reset_index(drop=True)
        self._years = ordered["period"].to_numpy(dtype=int)
        self._values = ordered["value"].to_numpy(dtype=float)
        # In-sample one-step naive residuals (y_t - y_{t-1}) -- the method's
        # own historical error, which is what gets resampled for the interval.
        self._residuals = np.diff(self._values)
        return self

    def predict(self, horizon: int) -> list[ForecastPoint]:
        if self._values is None:
            raise RuntimeError("call fit() before predict()")
        last_value = float(self._values[-1])
        last_year = int(self._years[-1])
        rng = np.random.default_rng(_RNG_SEED)

        points: list[ForecastPoint] = []
        for h in range(1, horizon + 1):
            lower, upper = _bootstrap_interval(last_value, self._residuals, h, rng)
            points.append(
                ForecastPoint(
                    period=str(last_year + h),
                    point=last_value,
                    lower=lower,
                    upper=upper,
                    unit="USD",
                )
            )
        return points

    def backtest(self, folds: int = 3) -> dict[str, float]:
        """Rolling-origin: for each of the last `folds` years, predict it from
        only the years before it (expanding window, never a random split --
        see .claude/commands/backtest.md on why that leaks the future)."""
        if self._values is None:
            raise RuntimeError("call fit() before backtest()")
        n = len(self._values)
        folds = min(folds, n - 1)
        if folds < 1:
            return {
                "mape": float("nan"),
                "rmse": float("nan"),
                "coverage": float("nan"),
                "folds": 0.0,
                "n_obs": float(n),
            }

        rng = np.random.default_rng(_RNG_SEED)
        pct_errors: list[float] = []
        sq_errors: list[float] = []
        covered = 0
        for i in range(n - folds, n):
            train = self._values[:i]
            actual = float(self._values[i])
            pred = float(train[-1])
            lower, upper = _bootstrap_interval(pred, np.diff(train), 1, rng)
            pct_errors.append(abs(actual - pred) / actual if actual else 0.0)
            sq_errors.append((actual - pred) ** 2)
            if lower <= actual <= upper:
                covered += 1

        return {
            "mape": float(np.mean(pct_errors)),
            "rmse": float(np.sqrt(np.mean(sq_errors))),
            "coverage": covered / folds,
            "folds": float(folds),
            "n_obs": float(n),
        }
