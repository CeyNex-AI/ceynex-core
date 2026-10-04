# ceynex-core

The engine of **CeyNex**, a multi-agent decision intelligence platform for Sri Lanka's export economy. It covers tea, cinnamon, rubber and coconut on the agriculture side and knitted and woven apparel (HS 61/62) on the manufacturing side.

A user asks a question in plain English. CeyNex answers it with figures computed from one integrated trade dataset and a knowledge graph. Each answer carries an evidence panel, a confidence score and, for forecasts, an 80% prediction interval. The language model only routes the question and phrases the result. A grounding guard throws away any generated sentence that states a figure the agents did not supply.

Group 07, Project P16, CS3501 Data Science and Engineering Project, University of Moratuwa.
Live system: <https://ceynex.cc> (sign-in required).

---

## Contents

1. [Where this repo fits](#where-this-repo-fits)
2. [How a question is answered](#how-a-question-is-answered)
3. [Repository layout](#repository-layout)
4. [Data](#data)
5. [Agents](#agents)
6. [Forecasting](#forecasting)
7. [Confidence score](#confidence-score)
8. [Grounding and degraded mode](#grounding-and-degraded-mode)
9. [Conversational layer, news and web search](#conversational-layer-news-and-web-search)
10. [API](#api)
11. [Accounts and security](#accounts-and-security)
12. [Getting started](#getting-started)
13. [Configuration](#configuration)
14. [Testing and CI](#testing-and-ci)
15. [Evaluation](#evaluation)
16. [Documentation index](#documentation-index)
17. [Team](#team)

---

## Where this repo fits

CeyNex is split over four code repositories in the [CeyNex-AI](https://github.com/CeyNex-AI) organisation:

| Repo | Role |
|---|---|
| [`ceynex-contracts`](https://github.com/CeyNex-AI/ceynex-contracts) | Frozen interfaces: `AgentState`, `AgentOutput`, `Evidence`, `ForecastPoint`, connector and model protocols, and the PostgreSQL and Neo4j schemas |
| **`ceynex-core`** (this repo) | Data pipeline, knowledge graph, agents, LangGraph orchestrator, forecasting models, FastAPI backend and evaluation harness |
| [`ceynex-web`](https://github.com/CeyNex-AI/ceynex-web) | React front end: query workspace, chat, scenario workbench, admin console |
| [`ceynex-infra`](https://github.com/CeyNex-AI/ceynex-infra) | Docker Compose, nginx, GCP setup, backups, monthly refresh and monitoring |

Two supporting repositories hold work that feeds this one: [`trade-data-pipeline`](https://github.com/CeyNex-AI/trade-data-pipeline) builds the policy-document vector index, and [`data-analysis`](https://github.com/CeyNex-AI/data-analysis) holds one exploratory notebook per data source.

`ceynex` is an implicit namespace package. `ceynex.contracts` is installed from `ceynex-contracts`, and everything else comes from this repo. Neither repo ships a `ceynex/__init__.py`, and adding one breaks every import.

## How a question is answered

```
browser ──HTTPS──▶ nginx (ceynex-web) ──/api──▶ FastAPI  (ceynex/api)
                                                 │  sign-in, rate limit, audit
                                                 ▼
                                  LangGraph orchestrator (ceynex/orchestrator)
                                                 │
                         route ──▶ router.py: LLM router, keyword fallback,
                                   clarification gate
                                                 │  fan-out, in parallel
             ┌──────────────┬───────────────┬────┴──────────┬──────────────┐
             ▼              ▼               ▼               ▼              ▼
     export_analytics  agriculture_   apparel_         trade_          forecast
                       commodity      manufacturing    economics
             │              │               │               │              │
             └──── Neo4j graph · PostgreSQL fact_trade · Qdrant policy passages ────┘
                                                 │
                         merge ──▶ merger.py + grounding.py + confidence.py
                                                 ▼
          { answer, confidence, evidence[], forecast[], agents_used, degraded }
```

1. **Route.** `orchestrator/router.py` picks the sectors, markets and agents. A low-cost LLM (gpt-4o-mini) does the routing, with a keyword router as a fallback. A rule keeps questions about past years away from the forecast agent. If a question is too ambiguous, the chat layer asks one clarifying question.
2. **Fan out.** The chosen agents run in parallel as LangGraph nodes. Each one queries the knowledge graph or the fact table, computes its figures and returns an `AgentOutput` with `Evidence` entries. Every evidence entry includes the literal Cypher or SQL that produced it.
3. **Merge.** `orchestrator/merger.py` asks the merge model to write one coherent answer from the agents' findings, and the model is not allowed to simply concatenate them. `grounding.py` then checks every figure in the prose against what the model was given, and `confidence.py` computes the score.
4. **Respond.** `POST /api/query` returns the whole answer at once. `POST /api/chat/stream` streams the same pipeline as Server-Sent Events with a live reasoning trace.

## Repository layout

```
ceynex/
  agents/            the five domain agents + shared helpers (common.py)
  orchestrator/      graph.py (LangGraph), router, planner, merger, grounding,
                     confidence, answer_stream (streamed answers)
  data/
    connectors/      one connector per source (see Data)
    cleaning/        cleaner + cross-validator (flags sources that disagree by 5%+)
    reference/       committed CSVs: countries, HS codes, item vocabulary,
                     partner aliases, regions, trade agreements, policy documents
    pipeline.py      fetch -> clean -> cross-validate -> write
    writer.py        the only writer to PostgreSQL (idempotent upsert)
    reader.py        the only reader agents use
    bootstrap.py     applies schema.sql and seeds dim_country / dim_hs
    freshness.py     per-source staleness for /health and the admin card
  kg/
    client.py        async Neo4j client, parameterised Cypher only
    loaders/         trade flows, agreements, apparel, agriculture, policy documents
    queries.py       named Cypher queries used by agents
    load.py          CLI: python -m ceynex.kg.load --schema --agreements ...
  models/            SARIMA (timeseries.py), LightGBM (gbm.py), combination
                     (combination.py), apparel model, shocks, registry
  retrieval/         Qdrant client for policy passages (hybrid dense + BM25)
  news/              GDELT news sidecar: refresh, relevance, trending, store
  websearch/         optional Tavily web search behind a provider protocol
  chat/              conversations, turn classifier, clarification, titles
  llm/client.py      OpenAI client with OpenRouter failsafe, spend cap, prompt cache
  observability/     request context, reasoning trace, usage ledger, spend
  api/               FastAPI app (main.py), auth, RBAC, rate limits, routes/
  settings.py        environment-driven settings
config/              llm.yaml, api.yaml, sources.yaml, news.yaml, elasticities.yaml
eval/                evaluation harnesses, question sets, back-testing, load test
eval_runs/           committed results of the repeated and load runs
migrations/          one-off SQL migrations
tests/               about 2,600 pytest tests (unit + integration)
tools/               docs_to_text.py, agriculture workbook audit
docs/                design notes, evaluation record, handoffs (see the index)
```

## Data

All trade figures land in one PostgreSQL table, `fact_trade`, keyed so a source can be re-ingested without creating duplicates. Production holds **13,132 observations from 8 sources, 1960 to 2026**.

| Source id | Connector | What it gives | Refresh |
|---|---|---|---|
| `UN_COMTRADE` | `comtrade.py` | Sri Lanka's exports by partner for all five products | monthly cron |
| `PINK_SHEET` | `pinksheet.py` | World Bank monthly Colombo tea auction price | monthly cron |
| `WB_FX` | `fx.py` | World Bank LKR/USD exchange rate | monthly cron |
| `FAOSTAT` | `faostat.py` | Producer prices and production | by hand, annual |
| `EDB` | `edb.py` | Export Development Board apparel exports | by hand |
| `JAAF` | `jaaf.py` | Joint Apparel Association Forum statistics | by hand |
| `TEA_BOARD` | `teaboard.py` | Curated annual tea export workbook | by hand |
| `CINNAMON` | `cinnamon.py` | Curated annual cinnamon export workbook | by hand |

- **Cleaning.** Country codes are harmonised to ISO 3166 and UN M49, and units are converted to US dollars and kilograms. `cleaning/cross_validator.py` raises a data-quality flag whenever two sources disagree on the same observation by 5% or more. Admins review and resolve those flags in the web console.
- **Provenance.** `fact_provenance` records the workbook, sheet and row that each curated agriculture fact came from.
- **Knowledge graph.** Neo4j holds `Country`, `Commodity`, `ApparelCategory`, `HSCode`, `TradeAgreement` and `PolicyDocument` nodes. They are linked by `EXPORTS_TO` edges carrying year, volume and value, plus agreement and policy relationships. The schema lives in `ceynex-contracts`.
- **Policy passages.** About 544 passages from destination-market trade policy documents sit in Qdrant (`ceynex_policy`). A Cypher query first restricts retrieval to documents issued by the market the question names. The index is built by `trade-data-pipeline`.

More detail: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md), [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md), [`docs/INSPECTING_THE_DATA.md`](docs/INSPECTING_THE_DATA.md), [`docs/FACT_PROVENANCE_HANDOFF.md`](docs/FACT_PROVENANCE_HANDOFF.md).

## Agents

Each agent is an asynchronous LangGraph node. It reads `AgentState` and returns an `AgentOutput` with a summary, figures, assumptions, a confidence and `Evidence[]`. If its datastore is down, it returns a degraded partial result instead of failing.

| Agent | File | Answers |
|---|---|---|
| `export_analytics` | `agents/export_analytics.py` | Values, growth, shares, rankings and top partners for any product and year |
| `agriculture_commodity` | `agents/agriculture_commodity.py` | Tea, cinnamon, rubber and coconut volumes, prices and destinations |
| `apparel_manufacturing` | `agents/apparel_manufacturing.py` | Knitted and woven apparel exports by market and category (EDB, JAAF, Comtrade) |
| `trade_economics` | `agents/trade_economics.py` | Trade agreements, tariff and FX shocks, elasticity-based scenarios, destination-market policy passages |
| `forecast` | `agents/forecast.py` | Forecasts with 80% intervals from the model registry, falling back to a drift baseline |

## Forecasting

`ceynex/models/` implements three model families behind the `ForecastModel` protocol from `ceynex-contracts`:

- **SARIMA(1,1,0)** (`timeseries.py`);
- **LightGBM** on year-on-year changes (`gbm.py`);
- **an equal-weight combination** of simple methods (`combination.py`).

Models are chosen by rolling-origin back-testing on annual export value (`eval/backtest.py`, `eval/model_comparison.py`). `models/registry.py::load_best` serves the version with the lowest MAPE. LightGBM is served only if it beats the best simpler model by at least 5%. Retrains are back-tested before they are saved.

- **Result:** the combination model was best on four of five series. It reached **5.2% MAPE on tea** and beat SARIMA in 19 of 29 one-year-ahead forecasts (Wilcoxon p = 0.011).
- **In production:** the combination is served for five of six series and SARIMA for cinnamon.

Every forecast carries an 80% interval. `ForecastPoint.lower` and `.upper` are required by the contract.

## Confidence score

`orchestrator/confidence.py` starts from a relevance-weighted average of the agents' own confidences, then subtracts three penalties:

| Penalty | Rule | Range |
|---|---|---|
| Staleness | 0.02 per month since the latest data point | 0 to 0.20 |
| Data quality | 0.05 per material and 0.10 per severe source disagreement | 0 to 0.25 |
| Coverage | a needed agent failed, or the answer is about years other than those asked | 0 or 0.15 |

The result is clamped to [0.05, 0.95]. The web front end shows it as Very low, Low, Moderate or High, along with a breakdown of the penalties.

## Grounding and degraded mode

- **Grounding** (`orchestrator/grounding.py`, `CEYNEX_GROUNDING=direction`). Every number in the generated prose must appear in something the merge model was given: the question, an agent's figures, its assumptions or the evidence. If one does not, the prose is discarded and a deterministic composition is served instead.
- **Degraded mode.** With no `OPENAI_API_KEY`, or when both OpenAI and the OpenRouter failsafe fail, the keyword router still routes and the agents still return figures and evidence. The response is marked `degraded: true`. This is a required behaviour of the system, not a fault.

## Conversational layer, news and web search

Each feature has a switch that can turn it off. When a switch is off, the system behaves exactly as it did before that feature existed:

| Switch | Feature |
|---|---|
| `CEYNEX_CHAT` | Multi-turn chat with follow-ups, conversation history, sharing and feedback (`ceynex/chat`, `/api/chat/*`) |
| `CEYNEX_CLARIFY` | A clarifying question when a question is ambiguous |
| `CEYNEX_SCENARIO` | The scenario workbench (`/api/scenario/run`) for tariff, FX and demand shocks |
| `CEYNEX_CITATIONS` | Inline `[n]` citations that link sentences to evidence |
| `CEYNEX_POLICY_RETRIEVAL` | Destination-market policy passages from Qdrant |
| `CEYNEX_NEWS` / `CEYNEX_NEWS_REFRESH` | GDELT news sidecar, refreshed hourly into its own Qdrant collection |
| `CEYNEX_WEB_SEARCH` | Tavily web search. It also needs `TAVILY_API_KEY`, and without one the feature stays off |

## API

The FastAPI app is `ceynex.api.main:app`. Interactive docs are at `/docs` when the server runs locally.

| Area | Endpoints |
|---|---|
| Health | `GET /health` (datastores, LLM, row count, stale sources) |
| Auth | `POST /api/auth/signup`, `POST /api/auth/login`, `GET /api/auth/me` |
| Query | `POST /api/query` |
| Chat | `POST /api/chat/stream`, `GET /api/chat/turns/{id}/events`, `POST /api/chat/turns/{id}/cancel`, conversations CRUD, trace, feedback, share, `GET /api/chat/shared/{token}` |
| Scenario | `POST /api/scenario/run` |
| Graph | `GET /api/graph/expand` (the knowledge-graph panel) |
| News | `GET /api/news/search`, `GET /api/news/trending` |
| Data | `GET /api/data/freshness` |
| History | `GET /api/history`, save and unsave |
| Account | password, e-mail, delete, notification preferences, API keys, custom instructions |
| Usage | `GET /api/usage/summary`, `/api/usage/limits`, `/api/usage/all` (admin) |
| Admin | `/api/admin/*`: users and roles, model registry and retrain, ingest and pipeline status, freshness, data-quality flags, audit log, LLM status |
| Site | `GET/POST /api/site/theme` |

## Accounts and security

- **Roles:** `admin`, `researcher`, `exporter` and `policymaker`. Sign-up never grants `admin`. The first admin comes from `CEYNEX_BOOTSTRAP_ADMIN=email:password`.
- **Sessions:** JWT with a per-user `token_epoch`, so a password change or an admin action revokes existing sessions. Passwords are hashed with bcrypt.
- **Sign-in required:** query, chat and news all need a signed-in user.
- **Rate limits:** 30 queries per minute per user (`config/api.yaml`), with a separate limit on auth endpoints. The counter is shared through Redis across the two API workers.
- **Spend cap:** a daily LLM spend cap (`config/llm.yaml`) plus a per-role usage ledger.
- **Audit log:** every admin action is written to an audit log, and the action fails if the log write fails.
- **CI security scanning:** bandit (SAST, against `bandit-baseline.json`), pip-audit and gitleaks.
- **Live testing:** an authenticated OWASP ZAP scan was run against the live site, and its findings were fixed.

## Getting started

Prerequisites: Python 3.11 or 3.12, Docker with Compose, and `ceynex-contracts` checked out next to this repo.

```bash
git clone https://github.com/CeyNex-AI/ceynex-contracts.git
git clone https://github.com/CeyNex-AI/ceynex-core.git
cd ceynex-core
python -m venv .venv && . .venv/bin/activate
cp .env.example .env            # add OPENAI_API_KEY and COMTRADE_API_KEY if you have them

make install      # ceynex-contracts from ../ceynex-contracts, then this package + pre-commit
make up           # postgres, neo4j, qdrant, redis, adminer; waits until healthy
make db-init      # apply schema.sql, seed dim_country / dim_hs
make ingest       # run every connector (manual sources need their files in data/raw/)
make kg-load      # schema.cypher, agreements, apparel, trade flows, policy documents
make news-refresh # optional: one GDELT refresh

CEYNEX_BOOTSTRAP_ADMIN=admin@example.com:change-me \
  uvicorn ceynex.api.main:app --port 8000
```

Then start `ceynex-web` with `npm run dev`. It proxies `/api` to port 8000.

Raw files for the manual sources (EDB, JAAF, Tea Board, cinnamon, FAOSTAT) are not committed. `data/raw/<source>/` shows where each one goes, and `data.md` explains how to obtain them.

## Configuration

| File | Holds |
|---|---|
| `.env` (from `.env.example`) | Datastore credentials and ports, API keys (OpenAI, OpenRouter, Comtrade, Tavily), feature switches |
| `config/llm.yaml` | Model per role (all production roles on gpt-4o-mini), token limits, prices, daily spend cap, OpenRouter failsafe |
| `config/api.yaml` | Rate limits and request limits |
| `config/sources.yaml` | Source cadence and which sources the monthly refresh re-fetches |
| `config/news.yaml` | GDELT queries and refresh interval |
| `config/elasticities.yaml` | Sourced trade elasticities used by scenarios |

## Testing and CI

```bash
make test-unit    # pytest -m "not integration", no Docker needed
make test         # everything, including integration tests against the Docker stack
make coverage     # unit suite with coverage (CI floor: 79%)
make lint         # ruff
make security     # bandit + pip-audit
```

There are about 2,600 tests, including 81 integration tests against real PostgreSQL, Neo4j and Qdrant. They cover:

- the agents and the orchestrator;
- the schema contract;
- an authorisation matrix over every endpoint;
- the apparel knowledge-graph spot checks;
- regression tests for bugs found in live audits.

GitHub Actions (`.github/workflows/`) runs:

- **ci:** ruff; unit tests on Python 3.11 and 3.12 with the coverage floor; integration tests against service containers.
- **security:** bandit, pip-audit and gitleaks.

## Evaluation

The evaluation set is 30 pre-registered questions (`eval/questions.yaml`), each run five times. Every claim in the answers was checked against the dataset by a GPT-4o judge, and two people audited the judge (Cohen's kappa 0.89 between the annotators, 0.92 between the annotators and the judge).

| Measure | GPT-4o alone | CeyNex |
|---|---|---|
| Figures not in the dataset | 67.9% | 0.0% |
| Claims contradicted by the data | 15.7% | 0.4% |
| Unanswerable questions refused | 6.7% | 100% |

- **Latency:** single-sector p95 is 8.2 s with the merge role on gpt-4o-mini.
- **Load:** on the production server, 50 concurrent users received all 150 answers with no failures.

| Command | What it runs |
|---|---|
| `make eval` / `make eval-degraded` | The 30 questions, with and without an LLM |
| `make eval-repeat` / `make eval-repeat-cited` | Repeated cold runs, without and with citations |
| `make eval-chat` | The multi-turn conversation set |
| `make eval-policy` / `make eval-policy-baseline` | The 15-question policy-retrieval set, with and without retrieval |
| `make coherence` | Blind merge-coherence sheets for three human raters |
| `make backtest SECTOR=agriculture ITEM=tea` | Rolling-origin forecast back-test |
| `make load-test` | 50 concurrent users against a running server |

Full method and history: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Documentation index

| Document | About |
|---|---|
| [`SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) | End-to-end map of a request |
| [`ARCHITECTURE_DELTA.md`](docs/ARCHITECTURE_DELTA.md) | Every deviation from the original SRS/SAD, with reasons (D1 onward) |
| [`EVALUATION.md`](docs/EVALUATION.md) | Evaluation method, runs and results |
| [`DATA_SOURCES.md`](docs/DATA_SOURCES.md), [`DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) | Sources, connectors and the ingest pipeline |
| [`INSPECTING_THE_DATA.md`](docs/INSPECTING_THE_DATA.md) | Queries for looking at the data yourself |
| [`POLICY_RETRIEVAL.md`](docs/POLICY_RETRIEVAL.md) | How policy passages are retrieved and cited |
| [`ANSWERABLE_QUESTIONS.md`](docs/ANSWERABLE_QUESTIONS.md) | What the system can and cannot answer |
| [`ELASTICITY_SOURCES.md`](docs/ELASTICITY_SOURCES.md) | Where each scenario elasticity comes from |
| [`DEFERRED.md`](docs/DEFERRED.md) | Known gaps and deferred work |
| `AGRICULTURE_*.md` | Agriculture pipeline, demo runbook, user guide and exporter feedback |

## Team

| Member | Main areas |
|---|---|
| Senindu Dinapura (230151T) | Agriculture data and agent, evaluation, workbook audit and fact provenance |
| Thisen Ekanayake (230170B) | Core systems, orchestrator, policy retrieval, news and web search, infrastructure |
| Dhinanjaya Fernando (230181J) | Apparel data and agent, forecasting models, evaluation and judge, front end, RBAC, security and load testing |

Supervisor: Dr. Chathuranga Hettiarachchi, University of Moratuwa.
