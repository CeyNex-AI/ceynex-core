"""Implements SRS 3.1.3 — forecasts carry an explicit interval, never a bare point.

FROZEN CONTRACT. Changes require 3-way approval (M1, M2, M3).
"""

from typing import TypedDict


class ForecastPoint(TypedDict):
    """A single forecast period with its uncertainty band.

    SRS 3.1.3 forbids presenting a forecast as an unqualified number, so
    `lower` and `upper` are required, not optional. The default band is the
    80% prediction interval; a model using a different level must say so in
    its registry `metadata.json` and in the Evidence it emits.
    """

    period: str  # ISO-8601, e.g. "2026-Q4"
    point: float
    lower: float
    upper: float
    unit: str
