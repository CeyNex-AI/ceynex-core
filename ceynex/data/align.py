"""Implements SRS 3.1.8 — frequency alignment across sources of differing cadence.

Comtrade publishes annually, the Central Bank publishes exchange rates daily,
JAAF publishes monthly bulletins. Anything that compares them has to put them on
one time index first, and the *way* it aggregates decides whether the comparison
means anything.

**The aggregation rule depends on what the number is**, which is why this module
exists rather than a bare `df.resample().sum()` at each call site:

| Metric | Rule | Why the others are wrong |
|---|---|---|
| volumes, values | **sum** | Twelve monthly tonnages make one annual tonnage |
| prices | **mean** | Summing twelve monthly USD/kg prices gives a number twelve times too large, in no unit at all |
| exchange rates | **period-end** | An FX rate is a level, not a flow. A shock simulation asks "what is the rate now", and an annual mean of a depreciating currency is a rate that was never observed |

Volume-weighting prices would be better than a plain mean where volumes exist,
and this deliberately does not do it: the weights are frequently missing on the
price series that need aligning, and a rule that silently changes when a column
happens to be present is worse than one that is always the same. It is recorded
as an assumption on the output instead.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

Frequency = Literal["D", "W", "M", "Q", "A"]

# pandas spells these differently and renamed several of them in 2.2. Mapped in
# one place so a pandas upgrade breaks one dict rather than every caller.
_PANDAS_RULE: dict[str, str] = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "A": "YE"}

# Ordered coarsest-last, so "can I go from X to Y" is an index comparison.
_ORDER: list[str] = ["D", "W", "M", "Q", "A"]

SUM_COLUMNS = ("export_volume", "export_value_usd", "volume", "value", "quantity")
MEAN_COLUMNS = ("price", "unit_value", "usd_per_kg")
LAST_COLUMNS = ("fx_usd_lkr", "fx", "exchange_rate", "rate")


class AlignmentError(ValueError):
    """Raised when a series cannot be aligned to the requested frequency."""


def aggregation_for(column: str) -> str:
    """The aggregation rule for a column, by name.

    Name-based rather than dtype-based because every one of these is a float;
    only the name says whether it is a flow, a price, or a level.
    """
    lowered = column.lower()
    if any(token in lowered for token in LAST_COLUMNS):
        return "last"
    if any(token in lowered for token in MEAN_COLUMNS):
        return "mean"
    if any(token in lowered for token in SUM_COLUMNS):
        return "sum"
    return "sum"


def resample(
    frame: pd.DataFrame,
    target: Frequency,
    *,
    period_column: str = "period_start",
    group_by: list[str] | None = None,
    rules: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resample to `target`, aggregating each column by what kind of number it is.

    `group_by` keeps identity columns (item, partner, sector) separate rather
    than collapsing every series into one — resampling a frame that holds twenty
    partners without grouping produces a single meaningless total.

    Upsampling is refused. Turning annual data into monthly data invents eleven
    observations per year that were never measured, and every downstream MAPE
    computed on them would be fiction.
    """
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()
    if target not in _PANDAS_RULE:
        raise AlignmentError(f"{target} is not one of {list(_PANDAS_RULE)}")
    if period_column not in frame.columns:
        raise AlignmentError(f"no `{period_column}` column in {list(frame.columns)}")

    working = frame.copy()
    working[period_column] = pd.to_datetime(working[period_column])

    source = _native_frequency(working)
    if source and _ORDER.index(source) > _ORDER.index(target):
        raise AlignmentError(
            f"refusing to upsample {source} data to {target}: it would invent "
            "observations that were never measured"
        )

    group_by = [c for c in (group_by or []) if c in working.columns]
    aggregations = {
        column: (rules or {}).get(column, aggregation_for(column))
        for column in working.columns
        if column != period_column
        and column not in group_by
        and pd.api.types.is_numeric_dtype(working[column])
    }
    if not aggregations:
        raise AlignmentError("no numeric columns to aggregate")

    grouper = pd.Grouper(key=period_column, freq=_PANDAS_RULE[target])
    keys = [*group_by, grouper] if group_by else [grouper]

    resampled = working.groupby(keys, dropna=False).agg(aggregations).reset_index()
    log.info("resampled %d rows to %d at frequency %s", len(frame), len(resampled), target)
    return resampled


def _native_frequency(frame: pd.DataFrame, period_column: str = "period_start") -> str | None:
    """Infer the frame's own cadence from the gap between consecutive periods.

    Returns None when there is only one period, which is not an error — a
    single-period frame is already aligned to anything.
    """
    periods = frame[period_column].drop_duplicates().sort_values()
    if len(periods) < 2:
        return None

    median_gap = periods.diff().dropna().dt.days.median()
    for frequency, upper in (("D", 3), ("W", 10), ("M", 45), ("Q", 130)):
        if median_gap <= upper:
            return frequency
    return "A"


def to_annual(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    """The common case: everything onto the annual index Comtrade already uses."""
    return resample(frame, "A", **kwargs)  # type: ignore[arg-type]


__all__ = [
    "LAST_COLUMNS",
    "MEAN_COLUMNS",
    "SUM_COLUMNS",
    "AlignmentError",
    "Frequency",
    "aggregation_for",
    "resample",
    "to_annual",
]
