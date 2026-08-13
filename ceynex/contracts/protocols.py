"""Implements SRS 3.1.7, 3.1.8, 3.1.10 — the shared interfaces the three members code against.

FROZEN CONTRACT. Changes require 3-way approval (M1, M2, M3).

These exist so M1 and M3 subclass one shape rather than inventing three. Each
protocol ships with a no-op or fake implementation so no member is ever blocked
waiting on another member's merge.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from ceynex.contracts.forecast import ForecastPoint


@dataclass
class SourceManifest:
    """What a connector reports about the pull it just made."""

    source_id: str
    fetched_at: str  # ISO-8601
    row_count: int
    period_start: str | None = None
    period_end: str | None = None
    frequency: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class DataSourceConnector(ABC):
    """One external data source (SAD Figure 4).

    `refresh_mode` is "scheduled" for sources with programmatic or bulk access
    (Comtrade, WITS, FAOSTAT, CBSL) and "event-driven" for sources that publish
    on their own cadence (JAAF monthly bulletins, EDB annual reports) — the two
    acceptable refresh modes in SRS 3.1.7.

    Implementations must cache raw responses to disk. These get re-run dozens of
    times over a sprint and every source has a rate limit or a slow endpoint.
    """

    source_id: str
    refresh_mode: str  # "scheduled" | "event-driven"

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return raw records. Column names are the source's own, not fact_trade's."""

    @abstractmethod
    def manifest(self) -> SourceManifest:
        """Describe the most recent fetch."""

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map raw columns onto the fact_trade schema (contracts 4.2).

        Default is identity so a connector can be written and tested before its
        mapping exists; override in every real connector.
        """
        return raw


@dataclass
class DQFlag:
    """A cross-source discrepancy. FLAG, never drop (SRS 3.1.8)."""

    item: str
    metric: str
    source_a: str
    value_a: float
    source_b: str
    value_b: float
    pct_diff: float
    severity: str  # "minor" <5% | "material" 5-20% | "severe" >20%
    hs_code: str | None = None
    partner_iso3: str | None = None
    period_start: str | None = None


@runtime_checkable
class CrossValidatorProtocol(Protocol):
    """Owned by M1 (SRS 3.1.8), consumed by M2's UnifiedDatasetWriter.

    Declared here so the writer can ship on Day 3 without waiting for M1's
    Day 3 merge — the writer takes an injected validator and defaults to
    `NullCrossValidator` below.
    """

    def cross_validate(self, records: pd.DataFrame) -> list[DQFlag]: ...


class NullCrossValidator:
    """Default injection: validates nothing, flags nothing.

    Lets the writer be built and tested independently of M1's schedule. Swapped
    for the real `CrossValidator` at the pipeline entrypoint once it lands; the
    writer itself does not change.
    """

    def cross_validate(self, records: pd.DataFrame) -> list[DQFlag]:
        return []


class ForecastModel(ABC):
    """One trained forecasting model (SAD Figure 10).

    Subclassed by M2's `TimeSeriesModel` and `GradientBoostedModel`, and by M1's
    and M3's sector models. `predict` returns `ForecastPoint`s, so a model that
    cannot produce an interval must produce one by residual bootstrap rather
    than returning a bare point (SRS 3.1.3).
    """

    sector: str
    item: str
    target: str

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "ForecastModel": ...

    @abstractmethod
    def predict(self, horizon: int) -> list[ForecastPoint]: ...

    @abstractmethod
    def backtest(self, folds: int = 3) -> dict[str, float]:
        """Rolling-origin backtest. Returns at least mape, rmse, coverage.

        Never a random train/test split — that leaks the future on a time series
        and produces a beautiful, meaningless MAPE.
        """


@runtime_checkable
class LLMReasoningClientProtocol(Protocol):
    """Natural-language explanation (SAD Figure 5).

    Implementations must degrade rather than raise: on provider failure the
    caller sets `degraded=True` and returns figures plus evidence without prose
    (SRS 3.4.3).
    """

    async def generate_explanation(self, context: dict[str, Any]) -> str: ...


@runtime_checkable
class KnowledgeGraphClientProtocol(Protocol):
    """Cypher access to the knowledge graph (SAD Figure 3).

    `run` returns the rows *and* the Cypher text that produced them, so an agent
    can put the literal query straight into `Evidence.detail` — that traceability
    is what makes SRS 3.1.6's "query the graph, not a model" claim checkable.
    """

    async def run(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], str]: ...
