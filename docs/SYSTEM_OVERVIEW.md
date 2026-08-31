# System overview

How CeyNex answers a question, end to end, and where things stand. Written for
demo prep — not a substitute for the SRS/SAD, just a map of what's actually in
the repos right now.

## What it is

A question-answering system over Sri Lanka's export economy (agriculture: tea,
cinnamon, rubber, coconut; apparel: HS 61/62). A user asks a plain-English
question and gets back an answer with figures, evidence, a confidence score,
and sometimes a forecast.

## The three repos

| Repo | Role |
|---|---|
| `ceynex-contracts` | Frozen interfaces — `AgentState`, `AgentOutput`, `Evidence`, `schema.sql`, `schema.cypher`. |
| `ceynex-core` | The engine — data pipeline, Neo4j/Postgres clients, the 5 agents, the LangGraph orchestrator, the FastAPI backend, eval harness. |
| `ceynex-infra` | GCP deployment — 3 VMs (frontend / backend / database), one VPC, only frontend has a public IP. |
| `ceynex-web` *(4th repo, M3's)* | React/Vite frontend — Login, Query page, Admin, Help. |

## Request path

```
browser -> ceynex-web (frontend VM, :80)
        -> nginx proxies /api -> backend VM :8000 (FastAPI)
        -> POST /api/query  (ceynex/api/routes/query.py)
        -> LangGraph graph.ainvoke(state)
             route node  -> keyword_route() or llm_route() picks agents
             fan-out (parallel) -> export_analytics, agriculture_commodity,
                                    apparel_manufacturing, trade_economics, forecast
                 each agent: parse_intent -> query Neo4j/Postgres -> build Evidence
                           -> derive confidence -> optional LLM prose
             merge node  -> combines outputs, aggregates confidence, builds final_answer
        -> QueryResponse { answer, confidence, evidence, forecast, agents_used, degraded }
        -> rendered in Query.tsx with EvidencePanel + ForecastChart
```

Backend only talks to the database VM (Postgres, Neo4j, Redis, Qdrant). Never the
reverse.

## Agents — 5 of 5 implemented

| Agent | File | Status |
|---|---|---|
| `export_analytics` | `ceynex/agents/export_analytics.py` | implemented |
| `trade_economics` | `ceynex/agents/trade_economics.py` | implemented |
| `forecast` | `ceynex/agents/forecast.py` | implemented |
| `apparel_manufacturing` | `ceynex/agents/apparel_manufacturing.py` | implemented |
| `agriculture_commodity` | `ceynex/agents/agriculture_commodity.py` | implemented (M1's, landed 2026-08-26) |

`graph.py`'s `_missing_agent_node` stand-in still exists and still returns a
`failed_output` naming the gap rather than crashing, so a route to an absent
agent degrades to a partial result. Nothing reaches it today.

**This matters for the numbers below and in `EVALUATION.md`:** both sector
agents were stubs when that evaluation was run, so its routing, grounding and
coherence figures describe a system two agents smaller than this one.

## Frontend connection

Real, not mocked. `ceynex-web/src/lib/queryApi.ts` calls `POST /api/query` and
translates the backend's field names (`answer`/`confidence`/`evidence`) into
the frontend's shape (`final_answer`/`final_confidence`/`merged_evidence`) —
that mapping exists because the deployed response diverged from what was
originally assumed. Check `queryApi.ts` first if the UI ever shows blank or
wrong fields.

## Confidence score

`ceynex/orchestrator/confidence.py`

Think of the confidence score as a starting trust level that gets knocked down
by three things that make the answer less reliable.

**Step 1 — start with a weighted average of the agents.**
Each agent that worked on the answer gives its own confidence (0 to 1). Agents
that were more relevant to the question are weighted more heavily. The result is
a single starting number.

**Step 2 — subtract penalties:**

| Penalty | What it means | How much |
|---|---|---|
| **Staleness** | The data is old. Every month since the latest data point costs 0.02, up to a max of 0.20 (reached at ~10 months). | 0 – 0.20 |
| **Data quality** | Two sources disagreed on a number the answer uses. Each "material" disagreement costs 0.05; each "severe" one costs 0.10. Capped at 0.25. | 0 – 0.25 |
| **Coverage** | At least one agent that was needed failed or couldn't give a full answer. Flat penalty. | 0 or 0.15 |

**Step 3 — clamp to [0.05, 0.95].**
The score never goes to 0 (the system did return something) and never reaches 1
(no trade forecast from historical data can be certain).

The UI converts the final number to a label: **Very low** (<0.30) /
**Low** (0.30–0.50) / **Moderate** (0.50–0.75) / **High** (≥0.75).

## Querying the knowledge graph

Only through `ceynex/kg/client.py::KnowledgeGraphClient` — an async, pooled
Neo4j driver. Every query is parameterized Cypher (`$param`, never an
f-string); a debug-log tripwire flags any query with `WHERE`/`SET`/`MERGE`/
`CREATE` but no `$`. `run()` returns `(rows, cypher_text)` together so agents
can put the literal Cypher into `Evidence.detail`, which is what makes the
evidence panel's claims verifiable instead of asserted. One retry on transient
errors, then it raises `KnowledgeGraphUnavailableError`, caught by every agent
and turned into a degraded partial result.

## What Postgres is for

Holds `fact_trade` (raw trade figures — reporter/partner/HS code/period/
volume/value) plus `dim_country`, `dim_hs`, `dq_flag`, `ingest_run`. It's the
system of record for raw statistics; Neo4j holds the relationships derived
from it (Country/Commodity/ApparelCategory nodes, `EXPORTS_TO` edges, trade
agreements). `ceynex/data/writer.py` is the only writer (idempotent upsert on
`fact_trade_upsert_key`), `ceynex/data/reader.py` is the only reader. No agent
opens a Postgres connection itself.

## What Qdrant is for

The newest datastore (deviation D10), and the only one that is optional. It
holds trade-policy documents from Sri Lanka's biggest export markets, cut into
searchable passages, so the system can say what the US or the UK actually does
rather than only what Sri Lanka ships them.

Reached only through `ceynex/retrieval/client.py`, and only by
`trade_economics`. Which documents are searchable is decided in Cypher first
(`kg/queries.py::policy_documents_for`), so a search cannot return the wrong
country's policy. Passages become `Evidence` entries with `source_id: POLICY`
and a link to the source page.

If Qdrant is down, not installed, or switched off with
`CEYNEX_POLICY_RETRIEVAL=off`, the system answers exactly as it did before it
existed. Plain guide: [POLICY_RETRIEVAL.md](POLICY_RETRIEVAL.md).

## Forecasting models

Five registered models under `models/`, all `GradientBoostedModel` (LightGBM,
`ceynex/models/gbm.py`), target `export_value_usd`, each with 2 saved
versions:

| Item | MAPE | 80% interval coverage |
|---|---|---|
| cinnamon | 12.2% | 100% |
| apparel_knit | 15.8% | 67% |
| tea | 5.3–5.7% | 67–100% |
| apparel_woven | 14.1–18.5% | 67–100% |
| rubber | 21.9–30.3% | 67% |

(Ranges cover the 2 saved versions per item — `load_best` serves whichever
scored lowest error, not necessarily the newest. Rubber's error is notably
higher than the rest, worth knowing if asked about it live.)

Served by the `forecast` agent via `ceynex/models/registry.py::load_best/
load_latest`. If no model is registered for an item, `forecast.py` falls back
to a drift baseline (mean year-on-year change + residual-bootstrap interval)
computed live from Neo4j history, and says so explicitly in its evidence and
assumptions. That baseline also serves as the accuracy benchmark ("did the
real model beat drift").

Note for Q&A: only 9 training rows and 3 CV folds per model — small sample,
expected for a term project. That's the honest answer if a marker asks about
MAPE reliability, not something to gloss over.

## Degraded mode (no LLM)

If `OPENAI_API_KEY` is empty or the LLM call fails, `keyword_route` still
routes correctly and agents return figures + evidence without prose
(`degraded: true`). This is a designed SRS 3.4.3 requirement, not a bug —
worth demonstrating deliberately with `--no-llm`.

An OpenRouter free-tier failsafe now sits behind the primary provider, so
"degraded" needs a key to be *absent*, not merely for one call to fail.

## Answer grounding (SRS 3.1.3)

`ceynex/orchestrator/grounding.py`

Every figure in the composed prose must appear in something the merge LLM was
given — the question, a finding's summary, its figures, its assumptions, or the
evidence. If one does not, the prose is **discarded** and the deterministic
composition is served instead; `MergeResult.ungrounded` lists what was rejected
and a warning is logged.

This was free while the system always ran degraded, because the deterministic
composer can only restate what agents wrote. With prose generation on it is the
only thing standing between a fabricated figure and the reader.

## Rate limiting (SRS 3.4.6)

`ceynex/api/rate_limit.py` — 30 queries/minute per signed-in user, or per client
address for anonymous callers, `429` with a `Retry-After` header past that.
Tuneable in `config/api.yaml`. Backed by Redis when `REDIS_URL` is set, because
the deployed image runs `uvicorn --workers 2` and a per-process counter would
allow double the configured limit. Fails **open** if Redis is unreachable.

## Running it

```bash
cd ceynex-core
python -m ceynex.orchestrator.demo "Which country takes the largest share of Sri Lanka's tea exports?"
python -m ceynex.orchestrator.demo --no-llm "..."   # force degraded mode
python -m ceynex.orchestrator.demo --json "..."
```

Needs `make up` (Postgres + Neo4j + Redis + Qdrant via docker-compose) locally, or
point `ceynex/settings.py`'s connection env vars at the GCP database VM.

Good demo questions are in `eval/questions.yaml` (30 pre-written, graded
questions: single-sector, cross-sector, simulation, and 3 deliberately
unanswerable ones).
