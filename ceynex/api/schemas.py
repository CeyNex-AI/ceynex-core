"""The REST contract from team overview §4.5 — SRS 3.9.3.

`QueryResponse` is what M3's web application renders, so its field names are as
frozen in practice as the Python contracts are. Changing one breaks the frontend
silently, since JSON has no type checker on the wire.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="A question in plain English.")


class EvidenceItem(BaseModel):
    source_id: str
    claim: str
    detail: str
    period: str | None = None
    url: str | None = None


class ForecastPointItem(BaseModel):
    period: str
    point: float
    lower: float
    upper: float
    unit: str


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    confidence_band: str
    agents_used: list[str]
    evidence: list[EvidenceItem]
    forecast: list[ForecastPointItem] | None = None
    degraded: bool
    elapsed_ms: float
    # Not in §4.5, and additive rather than breaking: the routing decision is the
    # project's core claim, so the demo needs it visible.
    route: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    neo4j: bool
    postgres: bool
    llm: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    token: str
    email: str
    role: str


class UserResponse(BaseModel):
    email: str
    role: str


class QueryHistoryItem(BaseModel):
    id: int
    query: str
    answer: str
    confidence: float
    degraded: bool
    asked_at: str
    saved: bool


class QueryHistoryResponse(BaseModel):
    items: list[QueryHistoryItem]


class SaveQueryResponse(BaseModel):
    id: int
    saved: bool


# --- admin (SRS 3.5.4) ------------------------------------------------------


class ModelSummary(BaseModel):
    sector: str
    item: str
    target: str
    version: str
    saved_at: str
    model_class: str
    training_rows: int | None = None
    metrics: dict[str, float] | None = None
    interval_level: float
    notes: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelSummary]


class RetrainRequest(BaseModel):
    sector: str = Field(min_length=1, max_length=50)
    item: str = Field(min_length=1, max_length=100)
    target: str = "export_value_usd"


class IngestRequest(BaseModel):
    # None/omitted = every registered connector, matching `pipeline.py --sources all`.
    sources: list[str] | None = None


class IngestResultItem(BaseModel):
    source_id: str
    status: str
    rows_in: int
    rows_written: int
    dq_flags: int
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    results: list[IngestResultItem]


class PipelineRunItem(BaseModel):
    run_id: int
    source_id: str
    started_at: str
    finished_at: str | None
    status: str
    rows_written: int
    error: str | None = None


class PipelineStatusResponse(BaseModel):
    runs: list[PipelineRunItem]


class DQFlagItem(BaseModel):
    flag_id: int
    item: str | None
    hs_code: str | None
    partner_iso3: str | None
    period_start: str | None
    metric: str | None
    source_a: str | None
    value_a: float | None
    source_b: str | None
    value_b: float | None
    pct_diff: float | None
    severity: str | None
    detected_at: str
    resolved: bool


class DQFlagsResponse(BaseModel):
    flags: list[DQFlagItem]


class ResolveDQFlagResponse(BaseModel):
    flag_id: int
    resolved: bool


class ProviderStatusItem(BaseModel):
    configured: bool
    status: str  # "not_configured" | "cap_reached" | "unknown" | "ok" | "down"
    last_error: str | None
    last_checked_at: str | None  # ISO 8601, converted from the client's epoch seconds


class LLMStatusResponse(BaseModel):
    openai: ProviderStatusItem
    openrouter: ProviderStatusItem
