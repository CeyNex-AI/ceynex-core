"""Implements SRS 3.1.3 and 3.1.10 — honest annual short-series baselines."""

from __future__ import annotations

from typing import Literal

from ceynex.models.base import SeriesForecastModel

BaselineStrategy = Literal["naive", "drift"]


class AnnualNaiveModel(SeriesForecastModel):
    """Annual naïve or drift baseline with an 80% residual interval.

    The naïve strategy carries the last observation forward.  Drift extends the
    average historical year-on-year change.  Both are deliberately simple and
    make a suitable reference for series below the 40-observation threshold;
    a more complex model must beat them under rolling-origin validation before
    it can be selected.
    """

    def __init__(
        self,
        *,
        sector: str,
        item: str,
        target: str = "value",
        unit: str = "",
        strategy: BaselineStrategy = "naive",
    ) -> None:
        super().__init__(sector=sector, item=item, target=target, unit=unit)
        if strategy not in {"naive", "drift"}:
            raise ValueError("strategy must be 'naive' or 'drift'")
        self.strategy = strategy
        self._last = 0.0
        self._drift = 0.0
        self._residuals: list[float] = []

    def describe_params(self) -> dict[str, object]:
        return {**super().describe_params(), "strategy": self.strategy}

    def _fit(self, _periods: list[int], values: list[float]) -> None:
        self._last = values[-1]
        self._drift = (values[-1] - values[0]) / (len(values) - 1)
        if self.strategy == "naive":
            self._residuals = [current - previous for previous, current in zip(values, values[1:], strict=False)]
        else:
            self._residuals = [
                current - (previous + self._drift)
                for previous, current in zip(values, values[1:], strict=False)
            ]

    def _predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        if self.strategy == "naive":
            points = [self._last] * horizon
        else:
            points = [self._last + self._drift * step for step in range(1, horizon + 1)]
        residuals = self._residuals
        # A perfectly linear short training series has zero in-sample residuals,
        # not zero future uncertainty.  Give it a conservative 5% level floor
        # so the required 80% band remains a band and widens by horizon.
        if len(residuals) >= 2 and max(residuals) == min(residuals):
            floor = max(abs(self._last) * 0.05, 1e-9)
            residuals = [-floor, floor]
        lower, upper = self._bootstrap_interval(points, residuals)
        return points, lower, upper
