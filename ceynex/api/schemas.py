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
