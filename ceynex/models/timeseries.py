"""Implements SRS 3.1.10 — the classical time-series model (SAD Figure 10).

`TimeSeriesModel` is the base class M1 and M3 subclass for anything that is a
plain series of periods and values. It fits SARIMA and falls back to
exponential smoothing when SARIMA will not converge, which on series this short
is common rather than exceptional.

**Why the default order is small.** Sri Lankan annual export series have about
ten observations. A seasonal term needs several full cycles to identify and
there are none in annual data, so seasonality is off by default and the
non-seasonal order stays at `(1, 1, 0)`: one autoregressive term, one difference
for the trend, no moving-average term. Richer orders fit these series better
in-sample and forecast them worse, which is the textbook overfit and also what
the backtest harness exists to catch.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from ceynex.models.base import SeriesForecastModel

log = logging.getLogger(__name__)

DEFAULT_ORDER = (1, 1, 0)
ALPHA_80 = 0.20  # statsmodels takes the complement of the interval level


class TimeSeriesModel(SeriesForecastModel):
    """SARIMA with an exponential-smoothing fallback.

    Usage::

        model = TimeSeriesModel(sector="agriculture", item="cinnamon").fit(df)
        points = model.predict(horizon=3)
        registry.save(model, training_rows=len(df), metrics=model.backtest())
    """

    def __init__(
        self,
        *,
        sector: str,
        item: str,
        target: str = "export_value_usd",
        unit: str = "USD",
        order: tuple[int, int, int] = DEFAULT_ORDER,
        trend: str | None = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(sector=sector, item=item, target=target, unit=unit)
        self.order = tuple(order)
        self.trend = trend
        self.fitted_family: str | None = None
        self._result: Any = None

    def describe_params(self) -> dict[str, Any]:
        return {
            **super().describe_params(),
            "order": list(self.order),
            "trend": self.trend,
            "fitted_family": self.fitted_family,
        }

    def _fit(self, periods: list[int], values: list[float]) -> None:
        # statsmodels is chatty about short series and non-invertible starting
        # values. The warnings are expected here and drown the real logs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = self._fit_sarima(values) or self._fit_ets(values)

        if self._result is None:
            raise RuntimeError(
                f"{self.item}: neither SARIMA{self.order} nor exponential smoothing "
                "converged on this series"
            )

    def _fit_sarima(self, values: list[float]) -> Any:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            result = SARIMAX(
                values,
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            self.fitted_family = f"SARIMA{self.order}"
            return result
        except Exception as exc:  # noqa: BLE001 - falling back is the designed behaviour
            log.info("%s: SARIMA%s did not converge (%s); trying ETS", self.item, self.order, exc)
            return None

    def _fit_ets(self, values: list[float]) -> Any:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            # Additive trend, no seasonality: annual data has no cycle to fit.
            result = ExponentialSmoothing(values, trend="add", seasonal=None).fit()
            self.fitted_family = "ETS(A,A,N)"
            return result
        except Exception as exc:  # noqa: BLE001
            log.warning("%s: exponential smoothing also failed: %s", self.item, exc)
            return None

    def _predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interval = self._sarima_interval(horizon)
            if interval is not None:
                return interval

            # ETS here gives no analytic interval, so the shared residual
            # bootstrap supplies one rather than the caller getting a bare point.
            points = [float(v) for v in self._result.forecast(horizon)]
            residuals = [float(r) for r in getattr(self._result, "resid", [])]
            lower, upper = self._bootstrap_interval(points, residuals)
            return points, lower, upper

    def _sarima_interval(
        self, horizon: int
    ) -> tuple[list[float], list[float], list[float]] | None:
        """SARIMA's own prediction interval, which beats a bootstrap when available."""
        get_forecast = getattr(self._result, "get_forecast", None)
        if not callable(get_forecast):
            return None
        try:
            forecast = get_forecast(steps=horizon)
            confidence = forecast.conf_int(alpha=ALPHA_80)
            points = [float(v) for v in forecast.predicted_mean]
            lower = [float(row[0]) for row in confidence]
            upper = [float(row[1]) for row in confidence]
            return points, lower, upper
        except Exception as exc:  # noqa: BLE001
            log.info("%s: no analytic interval available (%s); bootstrapping", self.item, exc)
            return None


__all__ = ["DEFAULT_ORDER", "TimeSeriesModel"]
