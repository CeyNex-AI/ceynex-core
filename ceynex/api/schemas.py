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


# --- the drawable graph (SRS 3.1.4, 3.1.6) ----------------------------------
#
# `EvidenceItem.detail` already carries the Cypher that produced each figure,
# which makes "this was answered from the graph" checkable by anyone who reads
# Cypher. These carry the same claim as a picture. Built by `ceynex/kg/subgraph.py`
# from explicitly-projected labels and relationship types — see that module on
# why `record.data()` means the shape has to be asked for rather than returned.


class GraphNode(BaseModel):
    #: "Country:USA" — the label plus its schema.cypher uniqueness key, not
    #: Neo4j's elementId, which changes across a reload and so could not survive
    #: the round trip back to /api/graph/expand.
    id: str
    label: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    focus: bool = False


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    #: 0..1 within its own relationship type, for stroke width. None where the
    #: relationship has no magnitude (CLASSIFIED_AS).
    weight: float | None = None
    #: The figure, formatted ("$412M"). `weight` is unit-less by the time it
    #: arrives, so this is the only thing that can be printed on the edge.
    label: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class AnswerGraph(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    focus_id: str | None = None
    #: The Cypher behind this picture, so the drawing is as auditable as the
    #: figures beside it.
    queries: list[str] = Field(default_factory=list)
    #: There is more graph than is drawn. In the response rather than inferred
    #: from a node count, same reasoning as `degraded` — a partial view that
    #: does not say so reads as a complete one.
    truncated: bool = False


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
    # Additive in the same way. None — and so no panel at all — whenever the
    # answer did not come from the graph: a diagram beside an answer the graph
    # did not produce would claim a provenance that isn't there.
    graph: AnswerGraph | None = None
    #: What the answer cost (D15). Additive, and optional so an older client is
    #: unaffected. The ledger has recorded this since the observability layer
    #: shipped; this is the first surface that shows it.
    usage: dict[str, Any] | None = None
    #: Every term in the SRS 3.1.4 confidence formula — weighted, staleness, dq,
    #: coverage, final — so "why this confidence?" is answerable from the answer.
    confidence_breakdown: dict[str, float] | None = None


class GraphFragment(BaseModel):
    """One hop out from a clicked node — the /api/graph/expand response.

    Deliberately not `AnswerGraph`: a fragment is merged into a canvas that
    already has a focus and its own queries, and reusing the answer shape would
    invite a caller to replace the drawing with it rather than add to it.
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    truncated: bool = False


# --- news sidecar (docs/ARCHITECTURE_DELTA.md D11) --------------------------
#
# Deliberately NOT `EvidenceItem`. News is not evidence — it has no `source_id`
# from the frozen `SourceId` vocabulary, no `claim`, and no `detail` naming the
# query that produced it, because none of those would be true of a headline
# nobody verified. A separate shape is what stops the two ever being merged by
# a future convenience.


class NewsArticleItem(BaseModel):
    url: str
    title: str
    domain: str
    source_country: str
    #: When GDELT first *saw* the article, not when it was published. The UI says
    #: "seen" for that reason.
    seen_at: str | None = None
    #: The raw cross-encoder logit, shipped but not rendered. It travels so the
    #: floor in config/news.yaml can be tuned without a frontend redeploy.
    relevance_score: float | None = None
    #: "strong" | "related" | "loose" | "unscored" — what the UI actually shows.
    relevance: str = "unscored"


class NewsSearchResponse(BaseModel):
    query: str
    articles: list[NewsArticleItem]
    #: "gdelt" (live) | "cache" (indexed headlines) | "unavailable" (neither).
    #: In the response on purpose: the panel says which it is showing rather than
    #: degrading silently, same reasoning as `QueryResponse.degraded`.
    source: str
    elapsed_ms: float


class TrendingArticleItem(BaseModel):
    url: str
    title: str
    domain: str
    seen_at: str | None = None


class TrendingTopicItem(BaseModel):
    topic_id: str
    label: str
    #: The question that goes in the query box when the chip is clicked — not the
    #: label, which would produce a question the router cannot use.
    prompt: str
    scope: str
    articles_24h: int
    baseline_24h: float
    delta_pct: float
    direction: str
    top_articles: list[TrendingArticleItem] = Field(default_factory=list)


class TrendingResponse(BaseModel):
    #: "warming" (never computed) | "ready" | "stale" (older than two intervals)
    #: | "unavailable" (the sidecar is off). Always HTTP 200 — an empty trending
    #: panel must not make the page look broken.
    status: str
    computed_at: str | None = None
    partial: bool = False
    scopes: dict[str, list[TrendingTopicItem]] = Field(default_factory=dict)


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


# --- account: notification preferences + API keys ---------------------------


class NotificationPreferences(BaseModel):
    dq_flag_alerts: bool
    forecast_updates: bool
    weekly_digest: bool


class ApiKeyItem(BaseModel):
    id: int
    label: str
    key_prefix: str
    created_at: str
    last_used_at: str | None
    revoked: bool


class ApiKeyListResponse(BaseModel):
    keys: list[ApiKeyItem]


class CreateApiKeyRequest(BaseModel):
    label: str = Field(min_length=1, max_length=100)


class CreateApiKeyResponse(BaseModel):
    id: int
    label: str
    key: str
    key_prefix: str
    created_at: str


class RevokeApiKeyResponse(BaseModel):
    id: int
    revoked: bool


# --- the conversational layer (deviation D13) --------------------------------
#
# Separate from `QueryResponse` on purpose. That shape is what the existing
# one-shot Query page binds to and is frozen in practice; a conversation is a new
# surface, and folding turns into the old shape would couple the two so that
# neither could move.


class ConversationSummary(BaseModel):
    """One row in the past-chats sidebar."""

    id: int
    title: str | None
    created_at: str
    updated_at: str
    pinned: bool
    archived: bool
    message_count: int


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=80)


class ConversationPatchRequest(BaseModel):
    """All three optional: a PATCH sets only what it names."""

    title: str | None = Field(default=None, max_length=80)
    pinned: bool | None = None
    archived: bool | None = None


class UsageSummary(BaseModel):
    """What one turn spent. A cache hit is 0 tokens and $0 — see
    `ceynex/observability/ledger.py` on why that is correct rather than missing."""

    calls: int = 0
    cache_hits: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class ChatMessageItem(BaseModel):
    """One turn. A user turn carries `content` and nothing else; an assistant
    turn carries the whole answer payload so reopening a conversation redisplays
    the evidence panel, the forecast and the graph rather than just the prose."""

    #: The row id, so a reader can rate this answer (§5). `seq` orders the
    #: transcript; it does not identify the row.
    id: int | None = None
    seq: int
    role: str
    content: str
    created_at: str | None = None
    mode: str | None = None
    request_id: str | None = None
    confidence: float | None = None
    confidence_band: str | None = None
    degraded: bool | None = None
    agents_used: list[str] = Field(default_factory=list)
    route: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    forecast: list[ForecastPointItem] | None = None
    graph: AnswerGraph | None = None
    elapsed_ms: float | None = None
    usage: UsageSummary | None = None
    #: Cross-link to the existing `query_history` row, so the chat UI's save
    #: button calls the untouched `/api/history/{id}/save` rather than a parallel one.
    query_history_id: int | None = None
    #: That row's `saved` flag, read through the join rather than stored twice.
    saved: bool = False
    #: SRS 3.1.4's working behind `confidence`, when it was computed.
    confidence_breakdown: dict[str, float] | None = None
    #: A `discuss` turn only: False when its prose was withheld as ungrounded.
    grounded: bool | None = None
    #: A user turn only: the query actually run, when it differs from `content`.
    effective_query: str | None = None
    #: A regenerated answer: the id of the version it replaces.
    regenerated_from: int | None = None


class ConversationDetail(BaseModel):
    conversation: ConversationSummary
    messages: list[ChatMessageItem]


class TraceEventItem(BaseModel):
    """One step of a stored reasoning trace, replayed when a chat is reopened.

    `payload` is open rather than typed per kind: the kinds are a taxonomy that
    will grow (a web-search step, a clarification step), and freezing the shape
    here would mean a contract change every time a call site learns to report
    something new. The frontend renders per `kind` and ignores what it does not
    recognise.
    """

    seq: int
    kind: str
    node: str | None = None
    ts: float
    payload: dict[str, Any] = Field(default_factory=dict)


class ChatStreamRequest(BaseModel):
    """A turn. `conversation_id` is optional so the streaming transport still
    works for a stateless question — which is what makes it demonstrable before
    any account exists."""

    query: str = Field(min_length=1, max_length=2000, description="A question in plain English.")
    conversation_id: int | None = None


class ClarifyAnswerRequest(BaseModel):
    """The reader's reply to a clarifying question (D13).

    `skip` is the "just answer it" escape, and it is not the same as sending no
    answers: it says the reader looked at the question and decided the original
    wording was what they meant.
    """

    answers: list[str] = Field(default_factory=list, max_length=8)
    skip: bool = False


# --- usage and cost (docs/ARCHITECTURE_DELTA.md D15) -------------------------


class UsageRollupItem(BaseModel):
    """One grouped row — a day, or a role/model pair. `key` says which."""

    key: str
    calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: float


class UsageSummaryResponse(BaseModel):
    days: int
    #: "user" (the caller's own) or "all" (admin-only, everyone's).
    scope: str
    by_day: list[UsageRollupItem] = Field(default_factory=list)
    by_role: list[UsageRollupItem] = Field(default_factory=list)
    total_cost_usd: float
    total_calls: int
    total_tokens_in: int
    total_tokens_out: int


class UsageLimitsResponse(BaseModel):
    """SRS 3.4.6's disclosure: the restrictions, said out loud."""

    limits: list[UsageRollupItem] = Field(default_factory=list)
    daily_spend_cap_usd: float
    spent_today_usd: float
    #: True, and deliberately in the response rather than hidden: the cap is
    #: enforced per uvicorn worker, so real spend can reach `worker_count` times
    #: it. Showing the cap as exact would be the silent enforcement the
    #: requirement forbids.
    cap_is_per_worker: bool = True
    worker_count: int = 2


class UserInstructionResponse(BaseModel):
    content: str
    enabled: bool
    max_chars: int


class UserInstructionRequest(BaseModel):
    content: str = Field(default="", max_length=2000)
    enabled: bool = True


class DataFreshnessResponse(BaseModel):
    """How current the trade record is. Always HTTP 200 — see the route."""

    available: bool
    #: The most recent period the dataset actually holds, not "now".
    latest_observation: str | None = None
    observations: int = 0
    #: The last ingest that *finished successfully*; a failed run says nothing
    #: about how current the data is.
    last_ingest_at: str | None = None


# --- answer feedback and shared conversations (execution plan §5) ------------


class FeedbackRequest(BaseModel):
    #: 1 for 👍, -1 for 👎. Not a 5-point scale: a rating nobody can interpret
    #: consistently is not eval data.
    rating: int = Field(ge=-1, le=1)
    reason: str = Field(default="", max_length=2000)


class FeedbackResponse(BaseModel):
    message_id: int
    rating: int


class ShareRequest(BaseModel):
    shared: bool


class ShareResponse(BaseModel):
    shared: bool
    #: None when sharing was turned off — the link is revoked, not hidden.
    token: str | None = None


class SharedConversationResponse(BaseModel):
    """A read-only transcript. Carries no `user_email`, by design."""

    title: str | None = None
    created_at: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
