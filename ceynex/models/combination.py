"""A forecast combination for short annual series: drift, damped ETS and Theta.

**Why a combination.** The export-value series have about ten annual points, one
of them missing at source (Comtrade has no 2018). On series this short a single
fitted model mostly chases noise. Measured on the live series on 2026-10-01 with
the shared rolling-origin harness (`eval/model_comparison.py` reproduces it):

    mean MAPE over six series     3 folds   5 folds
    combination (this class)        13.5%     13.4%
    SARIMA(1,1,0)                   16.0%     17.0%
    LightGBM on yearly changes      16.0%     17.2%

and on the 29 one-year-ahead forecasts of the 5-fold run it beat SARIMA in 19
(Wilcoxon signed-rank p = 0.011). An equal-weight average of simple methods is
the standard result for short series (the M3 and M4 competitions), so nothing
here is tuned to these six series: the members and the equal weights are fixed.

**Points and intervals.** The point forecast is the mean of the members' points.
The 80% interval is the envelope of the members' intervals (lowest lower bound,
highest upper bound). Averaging the bounds instead gave 0.61 to 0.72 coverage
against the nominal 0.80; the envelope gave 0.83 at both 3 and 5 folds.

**The 2018 gap.** Drift spreads a two-year change evenly over the years it
spans; ETS and Theta see the full annual index with the gap interpolated. None
of the three treats 2017 to 2019 as a single step.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import pandas as pd

from ceynex.models.base import Z_80, SeriesForecastModel

log = logging.getLogger(__name__)

ALPHA_80 = 0.20
MEMBERS = ("drift", "damped_ets", "theta")
MIN_MEMBERS = 2  # a "combination" of one model is that model, under another name


def _full_index(periods: list[int], values: list[float]) -> pd.Series:
    """Values on every year from first to last, missing years interpolated."""
    lookup = dict(zip(periods, values, strict=True))
    years = range(periods[0], periods[-1] + 1)
    series = pd.Series([lookup.get(y, float("nan")) for y in years], dtype=float)
    return series.interpolate().reset_index(drop=True)


def _yearly_steps(periods: list[int], values: list[float]) -> list[float]:
    """Changes per year; a gap's change is shared evenly across its years."""
    steps: list[float] = []
    for (p0, v0), (p1, v1) in zip(zip(periods, values, strict=True), zip(periods[1:], values[1:], strict=True), strict=False):
        span = p1 - p0
        steps.extend([(v1 - v0) / span] * span)
    return steps


class _Drift:
    """Random walk with drift: last value plus the mean yearly change."""

    def fit(self, periods: list[int], values: list[float]) -> _Drift:
        steps = _yearly_steps(periods, values)
        self.last = values[-1]
        self.mean = sum(steps) / len(steps)
        self.sd = math.sqrt(sum((s - self.mean) ** 2 for s in steps) / (len(steps) - 1))
        return self

    def predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        points = [self.last + self.mean * h for h in range(1, horizon + 1)]
        bands = [Z_80 * self.sd * math.sqrt(h) for h in range(1, horizon + 1)]
        return points, [p - b for p, b in zip(points, bands, strict=True)], [p + b for p, b in zip(points, bands, strict=True)]


class _DampedETS:
    """ETS(A,Ad,N): additive error, damped additive trend, no seasonality."""

    def fit(self, periods: list[int], values: list[float]) -> _DampedETS:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        self.result = ETSModel(_full_index(periods, values), error="add", trend="add",
                               damped_trend=True).fit(disp=False)
        return self

    def predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        start = self.result.nobs
        frame = self.result.get_prediction(start=start, end=start + horizon - 1).summary_frame(alpha=ALPHA_80)
        return list(frame["mean"]), list(frame["pi_lower"]), list(frame["pi_upper"])


class _Theta:
    """The Theta method (Assimakopoulos and Nikolopoulos), non-seasonal."""

    def fit(self, periods: list[int], values: list[float]) -> _Theta:
        from statsmodels.tsa.forecasting.theta import ThetaModel

        self.result = ThetaModel(_full_index(periods, values), period=1, deseasonalize=False).fit()
        return self

    def predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        points = [float(v) for v in self.result.forecast(horizon)]
        interval = self.result.prediction_intervals(horizon, alpha=ALPHA_80)
        return points, [float(v) for v in interval["lower"]], [float(v) for v in interval["upper"]]


_BUILDERS = {"drift": _Drift, "damped_ets": _DampedETS, "theta": _Theta}


class CombinationModel(SeriesForecastModel):
    """Equal-weight combination of drift, damped ETS and Theta."""

    def __init__(self, *, members: tuple[str, ...] | list[str] = MEMBERS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        unknown = set(members) - set(_BUILDERS)
        if unknown:
            raise ValueError(f"unknown combination members: {sorted(unknown)}")
        self.members = tuple(members)
        self.fitted_members: tuple[str, ...] = ()
        self.fitted_family = "Combination"

    def describe_params(self) -> dict[str, Any]:
        return {**super().describe_params(), "members": list(self.members)}

    def _fit(self, periods: list[int], values: list[float]) -> None:
        fitted = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels convergence chatter on ten points
            for name in self.members:
                try:
                    fitted[name] = _BUILDERS[name]().fit(periods, values)
                except Exception as exc:  # noqa: BLE001 - a member that cannot fit is left out
                    log.info("%s: combination member %s did not fit (%s)", self.item, name, exc)
        if len(fitted) < MIN_MEMBERS:
            raise ValueError(f"{self.item}: only {len(fitted)} combination member(s) could be fitted")
        self._fitted_models = fitted
        self.fitted_members = tuple(fitted)
        self.fitted_family = "Combination(" + ", ".join(self.fitted_members) + ")"

    def _predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outs = [model.predict(horizon) for model in self._fitted_models.values()]
        points = [sum(o[0][h] for o in outs) / len(outs) for h in range(horizon)]
        lower = [min(o[1][h] for o in outs) for h in range(horizon)]
        upper = [max(o[2][h] for o in outs) for h in range(horizon)]
        return points, lower, upper


__all__ = ["MEMBERS", "CombinationModel"]
