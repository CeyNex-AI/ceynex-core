"""Implements SRS 3.1.3 and 3.1.10 — shared machinery for registered models.

`ForecastModel` (frozen, in `ceynex-contracts`) says what a model must do.
This says how the ones in this project agree to do it, so that M1's cinnamon
model and M3's apparel model produce intervals that mean the same thing and
metrics that can sit in the same table.

Two conventions live here:

**The interval is an 80% band.** `ForecastPoint` requires `lower` and `upper`,
and a model whose library cannot produce them gets them by residual bootstrap
rather than by returning the point three times. A model using a different level
must record it in the registry's `metadata.json`.

**Uncertainty grows with the square root of the horizon.** Independent shocks
accumulate in variance, not in standard deviation, so a flat band over a 3-year
horizon understates exactly the thing the band exists to express. This matches
the drift baseline in `ceynex/agents/forecast.py`, deliberately — two widths for
the same 80% claim would make the forecasts incomparable.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import Any

import pandas as pd

from ceynex.contracts import ForecastModel, ForecastPoint

log = logging.getLogger(__name__)

Z_80 = 1.2816  # two-sided 80% normal quantile
DEFAULT_TARGET = "export_value_usd"
DEFAULT_UNIT = "USD"

# Sri Lankan annual export series run to about ten observations, one of which is
# missing at source (Comtrade has no 2018). Below this there is not enough signal
# to estimate a trend and an error variance, and a model that fits anyway
# produces a confident-looking number with no information in it.
MIN_OBSERVATIONS = 5

_PERIOD_COLUMNS = ("period", "year", "date", "ds", "time")
_VALUE_COLUMNS = ("value", "y", "export_value_usd", "amount")


class NotFittedError(RuntimeError):
    """Raised when `predict` is called before `fit`."""


class InsufficientHistoryError(ValueError):
    """Raised when a series is too short to model honestly."""


class SeriesForecastModel(ForecastModel):
    """Base for the univariate annual models this project registers.

    Subclasses implement `_fit` and `_predict`; everything else — column
    discovery, the fitted guard, interval construction, and the delegation to the
    shared backtest harness — is handled once here.
    """

    def __init__(
        self,
        *,
        sector: str,
        item: str,
        target: str = DEFAULT_TARGET,
        unit: str = DEFAULT_UNIT,
    ) -> None:
        self.sector = sector
        self.item = item
        self.target = target
        self.unit = unit
        self.version: str | None = None  # stamped by the registry on load
        self._periods: list[int] = []
        self._values: list[float] = []
        self._fitted = False

    # --- the contract ----------------------------------------------------

    def fit(self, df: pd.DataFrame) -> SeriesForecastModel:
        periods, values = self._series(df)
        if len(values) < MIN_OBSERVATIONS:
            raise InsufficientHistoryError(
                f"{self.item}: {len(values)} observations, need at least "
                f"{MIN_OBSERVATIONS}. Refusing to fit rather than returning a "
                "model whose interval would be meaningless."
            )
        self._periods, self._values = periods, values
        self._fit(periods, values)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> list[ForecastPoint]:
        if not self._fitted:
            raise NotFittedError(f"{type(self).__name__} has not been fitted")
        if horizon < 1:
            raise ValueError("horizon must be at least 1 period")

        points, lower, upper = self._predict(horizon)
        last = self._periods[-1]
        return [
            ForecastPoint(
                period=str(last + step + 1),
                point=round(float(points[step]), 2),
                # Export value cannot be negative; clamping is more honest than a
                # lower bound the quantity cannot reach.
                lower=round(max(0.0, float(lower[step])), 2),
                upper=round(float(upper[step]), 2),
                unit=self.unit,
            )
            for step in range(horizon)
        ]

    def backtest(self, folds: int = 3) -> dict[str, float]:
        """Rolling-origin backtest, delegated to the one shared harness.

        Imported here rather than at module scope because the harness imports the
        registry, which imports the models. Delegating keeps a single definition
        of MAPE across all three members' models, which is the whole reason the
        harness is shared.
        """
        from eval.backtest import rolling_origin

        if not self._fitted:
            raise NotFittedError("fit before backtesting")
        return rolling_origin(self, self._frame(), folds=folds)

    # --- for subclasses --------------------------------------------------

    @abstractmethod
    def _fit(self, periods: list[int], values: list[float]) -> None:
        """Fit on an ordered annual series."""

    @abstractmethod
    def _predict(self, horizon: int) -> tuple[list[float], list[float], list[float]]:
        """Return `(points, lower, upper)`, each of length `horizon`."""

    def describe_params(self) -> dict[str, Any]:
        """Hyperparameters for `metadata.json`. Override to add model-specific ones."""
        return {"sector": self.sector, "item": self.item, "target": self.target, "unit": self.unit}

    def _bootstrap_interval(
        self, points: list[float], residuals: list[float]
    ) -> tuple[list[float], list[float]]:
        """An 80% band from the spread of in-sample residuals, widened by sqrt(h).

        The fallback for a model whose library gives no interval of its own.
        Honest about being an approximation: it assumes the residuals are roughly
        symmetric and that future errors look like past ones.
        """
        if len(residuals) < 2:
            # No spread to measure. A zero-width band would claim certainty the
            # model does not have, so widen by the level of the series instead.
            spread = abs(points[0]) * 0.25 if points else 0.0
        else:
            mean = sum(residuals) / len(residuals)
            spread = math.sqrt(sum((r - mean) ** 2 for r in residuals) / (len(residuals) - 1))

        lower, upper = [], []
        for step, point in enumerate(points, start=1):
            band = Z_80 * spread * math.sqrt(step)
            lower.append(point - band)
            upper.append(point + band)
        return lower, upper

    # --- data ------------------------------------------------------------

    def _series(self, df: pd.DataFrame) -> tuple[list[int], list[float]]:
        """Pull an ordered annual `(period, value)` series out of a frame.

        Column names are discovered rather than mandated because three members
        build these frames from three different sources, and a rigid contract
        here would be one more thing to coordinate for no analytical gain.
        """
        if df is None or df.empty:
            raise InsufficientHistoryError(f"{self.item}: empty training frame")

        period_col = self._column(df, (*_PERIOD_COLUMNS,), "period")
        value_col = self._column(df, (self.target, *_VALUE_COLUMNS), "value")

        frame = df[[period_col, value_col]].dropna()
        frame = frame.groupby(period_col, as_index=False)[value_col].sum()
        frame[period_col] = frame[period_col].apply(_as_year)
        frame = frame.sort_values(period_col)

        return (
            [int(p) for p in frame[period_col].tolist()],
            [float(v) for v in frame[value_col].tolist()],
        )

    @staticmethod
    def _column(df: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
        lowered = {str(c).lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate and str(candidate).lower() in lowered:
                return lowered[str(candidate).lower()]
        raise InsufficientHistoryError(
            f"no {role} column in {list(df.columns)}; expected one of {candidates}"
        )

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({"period": self._periods, "value": self._values})


def _as_year(value: Any) -> int:
    """Coerce a period label to a year. Accepts 2023, '2023', '2023-01-01'."""
    if isinstance(value, int | float):
        return int(value)
    text = str(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return int(pd.Timestamp(text).year)


__all__ = [
    "DEFAULT_TARGET",
    "DEFAULT_UNIT",
    "MIN_OBSERVATIONS",
    "Z_80",
    "InsufficientHistoryError",
    "NotFittedError",
    "SeriesForecastModel",
]
