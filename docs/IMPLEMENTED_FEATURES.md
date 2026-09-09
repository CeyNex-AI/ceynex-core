# CeyNex — Implemented Features

Multi-agent decision-intelligence platform for Sri Lanka's national export economy
(agriculture: tea, cinnamon, rubber, coconut; apparel: HS 61/62).
Group 07, Project P16, CS3501, University of Moratuwa.

*Snapshot: 2026-09-09, against `main` of all four repos. Live host: `https://34.47.150.194`.*

Repos: `ceynex-contracts` (frozen interfaces) · `ceynex-core` (engine + API) ·
`ceynex-web` (React frontend) · `ceynex-infra` (GCP deployment).

---

## 1. Multi-agent orchestration

- **LangGraph orchestrator** (`orchestrator/graph.py`) — route → parallel fan-out → merge, with a per-node timeout so one slow/hanging agent cannot stall the request.
- **5 of 5 domain agents implemented:**
  | Agent | Covers |
  |---|---|
  | `export_analytics` | export value/volume, market concentration (HHI), partner shares, growth/CAGR |
  | `agriculture_commodity` | tea/cinnamon/rubber/coconut price & volume trends, district/production questions, substitution |
  | `apparel_manufacturing` | HS 61/62 apparel, EDB + JAAF sources, knit vs woven |
  | `trade_economics` | tariff/policy-shock simulation, trade-agreement effects, policy-document retrieval |
  | `forecast` | short-horizon export-value forecasts with 80% intervals |
- **Two routers:**
  - `keyword_route()` — deterministic keyword/sector routing, always available (works with no LLM key).
  - `llm_route()` — LLM-based routing for ambiguous / cross-sector questions, with keyword routing as the fallback.
- **Missing-agent stand-in** — a route to an absent agent returns a `failed_output` naming the gap instead of crashing; the answer degrades to a partial result.
- **Intent parsing** (`agents/common.py::parse_intent`) — extracts item, trade partner, region/district, forecast target (export value / export volume / producer price), horizon, and requested frequency (year/quarter/month) from plain English.
- **Region- / partner-scoped queries** — a named country or region constrains every agent's Cypher; a wrong region is treated as a wrong answer, not a missing one.

## 2. Answer synthesis, evidence & confidence

- **Finding-oriented merge** (`orchestrator/merger.py`) — combines agent outputs into one answer organised by finding, never "agent A says… agent B says…".
- **Conflict surfacing** — directional or large-magnitude disagreements between agents are shown, never averaged.
- **Evidence panel** — every figure traces to an `Evidence` entry: `source_id`, the human claim, and the literal Cypher / model id / policy passage in `detail`. Deduplicated across agents, attribution preserved.
- **Answer grounding (SRS 3.1.3)** — every figure in the generated prose must appear in something the merge LLM was given; if not, the prose is discarded and a deterministic composition is served, with the rejected figures logged.
- **Confidence score (SRS 3.1.4)** — relevance-weighted mean of agent confidences minus three penalties: staleness (age of data, ≤0.20), data-quality (cross-source disagreement, ≤0.25), coverage (a needed agent failed, 0.15). Clamped to [0.05, 0.95]. No hardcoded scores.
- **Confidence bands** — Very low / Low / Moderate / High, shown beside the percentage.
- **Honest declines** — an agent's SAD §4.1 refusal ("no sourced series for X") is reported as a stated gap, not a competing finding, and is suppressed when another finding already covered the question. *(PR #59: a decline no longer leaks its low confidence or its data-gap evidence into a real answer.)*
- **`unanswered[]`** — the parts of a question that could not be answered are returned explicitly.

## 3. Forecasting

- **Model registry** (`models/registry.py`) — per-item versioned models under `models/<sector>/<item>/<target>/<version>` with `load_best` (lowest backtest error first) / `load_latest`.
- **Registered models live:** tea, cinnamon, rubber (agriculture) + apparel_knit, apparel_woven, apparel, apparel_edb — SARIMA(1,1,0) time-series models, target `export_value_usd`, 80% prediction intervals, rolling-origin MAPE per item. (`models/timeseries.py`; a gradient-boosted LightGBM model class, `models/gbm.py`, is also implemented.)
- **Drift-baseline fallback** — when no model is registered for an item, `forecast.py` computes a drift forecast (mean YoY change + residual-bootstrap 80% interval) live from Neo4j history and says so in its evidence/assumptions. Doubles as the "did the real model beat drift" benchmark.
- **Backtesting harness** (`eval/backtest.py`, `make backtest`) — rolling-origin CV, writes MAPE / interval-coverage into model metadata.
- **Admin retrain** — refit a registered model's class on the latest data from the UI.

## 4. Knowledge graph (Neo4j)

- **Single access path** — `kg/client.py::KnowledgeGraphClient`, async pooled driver; every query is parameterised Cypher (`$param`), with a debug tripwire that flags any mutating query missing a bind parameter.
- **`run()` returns `(rows, cypher_text)`** so agents can put the exact query into `Evidence.detail`.
- **Loaders** (`kg/loaders/`): trade flows, trade agreements, agriculture, apparel, policy documents.
- **Graph model** — `Country` / `Commodity` / `ApparelCategory` nodes, `EXPORTS_TO` edges (year, value, volume), `TradeAgreement` nodes and coverage.
- **Subgraph expansion** (`kg/subgraph.py`) — neighbourhood fetch behind the `/api/graph/expand` endpoint for the KG viz panel.
- **Resilience** — one retry on transient errors, then `KnowledgeGraphUnavailableError`, caught by every agent → degraded partial result.

## 5. Data pipeline (Postgres / Parquet)

- **Connectors** (`data/connectors/`): UN Comtrade, EDB, JAAF, FAOSTAT, Tea Board, World Bank Pink Sheet, cinnamon, apparel-source bundle. Each declares a `source_id` used in every evidence citation.
- **Ingestion pipeline** (`data/pipeline.py`, `make ingest`) — fetch → clean → crosswalk/align → idempotent upsert.
- **Single writer / single reader** — `data/writer.py` (idempotent upsert on `fact_trade_upsert_key`), `data/reader.py`. No agent opens a DB connection itself.
- **Cleaning & cross-validation** (`data/cleaning/`) — per-source cleaners plus a cross-source validator that raises data-quality flags.
- **Crosswalks & reference tables** (`data/reference/`): countries, HS codes, item vocabulary, partner aggregates & aliases, regions, trade agreements + coverage, policy documents.
- **System of record** — `fact_trade` (reporter/partner/HS/period/volume/value) + `dim_country`, `dim_hs`, `dq_flag`, `ingest_run`; ~12,100 rows live, data span 2015–2025.
- **Ingest-run tracking** — every pipeline run recorded and surfaced in Admin.

## 6. Policy-document retrieval (Qdrant — optional, deviation D10)

- **Vector search over trade-policy documents** from Sri Lanka's largest export markets (US, UK, …), chunked into passages.
- **Graph-anchored** — eligible documents are chosen in Cypher first (`kg/queries.py::policy_documents_for`), so a search cannot return the wrong country's policy.
- **Reranking** with a relevance floor (`retrieval/client.py::_rerank`) — returns nothing rather than the least-bad passage.
- **Provenance** — passages become `Evidence` with `source_id: POLICY` and a link to the source page; the filter it ran under is recorded (a vector search has no query text to cite).
- **2-second hard budget** on embed → search → rerank; past it the agent gets nothing and says so.
- **Fully optional** — Qdrant down / not installed / `CEYNEX_POLICY_RETRIEVAL=off` → the system answers exactly as it did before retrieval existed.

## 7. News sidecar (GDELT — deviation D11)

- **GDELT DOC 2.0 client** (`news/gdelt.py`) — article search + 7-day volume timeline, `httpx.MockTransport` seam for tests.
- **Outbound throttle** (`news/throttle.py`) — GDELT enforces an unpublished rate limit; a 6-second gate + back-off on HTTP 429.
- **Degradation** — a 200 carrying a plain-text complaint is caught, re-raised as `GdeltUnavailableError`, and the raw body is written to a forensic file.
- **Relevance filtering & snapshots** (`news/relevance.py`, `news/snapshot.py`, `news/store.py`) — on-disk cache with TTL.
- **Endpoints:** `GET /api/news/search` (coverage for a question), `GET /api/news/trending`.
- **UI:** `NewsPanel` (per-answer coverage) and `TrendingNews`.

## 8. Backend API (FastAPI)

| Area | Endpoints |
|---|---|
| **Query** | `POST /api/query` — the full pipeline; returns answer, confidence + band, evidence, forecast, agents_used, route, sectors, unanswered, KG graph payload, degraded flag |
| **Auth** | `POST /api/auth/login`, `GET /api/auth/me` — signed JWT, `admin` / non-admin roles |
| **History** | `GET /api/history`, `POST /api/history/{id}/save`, `POST /api/history/{id}/unsave` — per-user query history + saved analyses |
| **Account** | `GET`/`PUT /api/account/preferences` (notification prefs), `GET`/`POST /api/account/api-keys`, `POST /api/account/api-keys/{id}/revoke` — API-key create / list / one-time-copy / revoke |
| **Admin** | `GET /llm/status`, `GET /models`, `POST /retrain`, `POST /pipeline/ingest`, `GET /pipeline/status`, `GET /dq-flags`, `POST /dq-flags/{id}/resolve`, `GET /audit-log` |
| **Graph** | `GET /api/graph/expand?node=…` — KG neighbourhood for the viz panel |
| **News** | `GET /api/news/search`, `GET /api/news/trending` |
| **Site** | `GET`/`POST /theme` — site-wide appearance setting |
| **Health** | `GET /health` — neo4j / postgres / llm reachability, `fact_trade_rows`, reasoning availability |

- **Rate limiting (SRS 3.4.6)** — 30 queries/min per signed-in user (or per client address when anonymous), `429` + `Retry-After`; Redis-backed so it holds across `uvicorn --workers 2`; fails open if Redis is unreachable. Separate limiters on `/api/graph/expand` and `/api/news/*`.
- **Audit log** — admin actions recorded and viewable (`GET /audit-log`).
- **Field-name translation** documented for the frontend (`answer` → `final_answer`, etc.).

## 9. LLM integration

- **Provider client** (`llm/client.py`) — used for routing, per-agent explanation prose, and the final merge.
- **OpenRouter free-tier failsafe** behind the primary provider — a single failed call no longer means degraded mode; the key must be genuinely absent.
- **Degraded mode (SRS 3.4.3)** — no key / LLM failure → keyword routing + figures + evidence with no prose, `degraded: true`. A designed, demonstrable path (`demo.py --no-llm`).
- **Per-agent grounded explanations** — each agent's own prose is grounded against its own findings, not only the merge LLM's output (PR #58).
- **Admin LLM status** — last real call to each provider (model, latency, ok/error), not a live probe.
- **Prompt-hash cache** (`.cache/`).

## 10. Frontend (`ceynex-web` — React / Vite / Tailwind)

- **Pages:** Login, Query, Account, Admin, Help, 404.
- **Auth:** JWT stored client-side, `RequireAuth` route guard, role-aware nav (Admin hidden for non-admins; also enforced server-side).
- **Query page:** question box → answer with
  - `ConfidenceBadge` (percentage + band),
  - `EvidencePanel` (expandable, per-source claims + underlying query),
  - `ForecastChart` (points + 80% interval band),
  - `KnowledgeGraphPanel` (Cytoscape graph, click a node → `/api/graph/expand`),
  - `NewsPanel` (related coverage), `TrendingNews`.
- **History & saved analyses** — list past queries, save/unsave.
- **Account page** — notification preferences; API-key create / copy-once / revoke.
- **Admin page** — System status, LLM providers, Models (+ retrain), Data pipeline (+ trigger ingest, run history), Data-quality review (resolve DQ flags), Appearance (site theme).
- **Help page** — usage guidance, example questions.
- **Branding** — real CeyNex logo (favicon + login lockup).
- **Theming** — classic default with an opt-in "Signal Deck" site-wide theme, switchable from Admin. *(On `main`; live VM still serves the earlier build.)*

## 11. Evaluation & quality harness

- **Orchestrator eval** (`eval/harness.py`, `make eval`) — 30 graded questions in `eval/questions.yaml` (single-sector, cross-sector, simulation, 3 deliberately unanswerable), plus `eval-degraded`.
- **Policy-retrieval eval** — `eval/policy_questions.yaml`, `make eval-policy` and `eval-policy-baseline` (measures the pre-retrieval system).
- **Coherence check** (`eval/coherence.py`, `make coherence`).
- **Backtest** (`eval/backtest.py`).
- **Test suite** — ~1,220 unit tests (`make test-unit`, no Docker needed); integration tests behind an `integration` marker.
- **Lint/format** — `ruff` (`make lint` / `make fmt`).
- **Docs conversion** — `make docs` renders SRS/SAD/plan to greppable text; standing overview docs in `docs/` (SYSTEM_OVERVIEW, DATA_SOURCES, DATA_PIPELINE, POLICY_RETRIEVAL, EVALUATION, ANSWERABLE_QUESTIONS, …).

## 12. Deployment (`ceynex-infra`)

- **3-tier GCP deployment**, one VPC (`asia-south1-a`): frontend (only public VM, nginx :80/:443) → backend (FastAPI + LangGraph) → database (Postgres 18, Neo4j 5.26 + APOC, Redis 8, Qdrant).
- **Tag-based firewall** — backend/database reachable only from the tier above by instance tag, not address range.
- **HTTPS** at the proxy (self-signed cert, HTTP→HTTPS redirect), JWT secret shared via env.
- **Dockerised** — backend image builds from `ceynex-contracts` + `ceynex-core` as sibling checkouts; `docker compose` per tier; documented `DEPLOY_ORDER.md` (database first).
- **Ops scripts** — Docker install, firewall setup, cert generation.
- **Model registration on deploy** — deploy flow registers forecast models on the backend VM.
- **Schema management** — applied by `make db-init` / `make kg-load` (no initdb hook; the DB is updated, not recreated).
- **Migration-ready** — a fresh-rebuild runbook exists; DB moves via `pg_dump -Fc` / `pg_restore`.

---

## Known gaps / caveats (not features, but part of an honest picture)

- **No CI** — `.github/workflows/` does not exist; the test suite is only ever run locally. *(PR #60 fixes two never-committed / unguarded tests this let through.)*
- **Tea Board / FAOSTAT never ingested** — those raw files are absent from the VM and the workspace; tea-volume and cinnamon-price series show 0 observations. UN Comtrade has usable volume/price data in `fact_trade` as a possible fallback.
- **No district-level data** in the graph.
- **Forecast training samples are small** — ~9–10 rows, 3 CV folds per model (expected for a term project).
- **Live VM ≠ `main`** — the site-theme frontend build and PRs #59/#60 are on `main` but not yet deployed.
- **Rubber forecast error** is materially higher than the other items (~22–30% MAPE).
