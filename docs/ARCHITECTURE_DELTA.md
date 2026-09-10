# Architecture Delta

Where the build diverges from `CeyNex-SAD-v1.0` and the team plan, and why.
Deviations with reasons are engineering; deviations without are drift.

---

## D1 — `AgentState` carries LangGraph reducers

**Agreed:** Day 1, 3-way, at the contract review.
**Spec touched:** team overview §4.1, SRS 3.6.4.

Team overview §4.1 declares `agent_outputs: dict[AgentName, AgentOutput]` and
`errors: list[str]` as plain `TypedDict` keys. LangGraph raises
`InvalidUpdateError` when two nodes running in the same superstep write the same
state key without a reducer telling it how to combine the writes. A cross-sector
query (SRS 3.1.2) fans out to 2–3 agent nodes in parallel and every one of them
writes `agent_outputs`, so the contract as written cannot express the routing
behaviour the SRS requires.

`agent_outputs`, `errors` and `degraded` are therefore `Annotated` with
`merge_agent_outputs`, `operator.add` and `operator.or_` respectively. The field
names, value types, and every other key are unchanged, so no member's code needs
to move. `tests/contracts/test_state.py::test_parallel_state_keys_carry_reducers`
guards it.

**Also added:** `relevance: dict[AgentName, float]` — the router's per-agent
weight. It is written once before fan-out (no reducer needed) and is the `w_i`
term in the confidence formula (SRS 3.1.4), which otherwise has no principled
way to weight a cross-sector answer.

---

## D2 — One Neo4j container per member, not one instance with per-member databases

**Agreed:** Day 1, 3-way.
**Spec touched:** team overview §7, risk R8.

R8's stated mitigation is "each member gets a separate Neo4j database name
locally." Multi-database is a Neo4j **Enterprise** feature; Community 5.x serves
a single user database (`neo4j`) plus `system`, so the mitigation is not
available on the edition the project uses.

Replaced with: each member runs their own compose stack, with
`NEO4J_BOLT_PORT` / `NEO4J_HTTP_PORT` / `POSTGRES_PORT` read from `.env`. Only
`main`'s loader writes the shared demo instance. Same isolation guarantee,
actually implementable.

---

## D3 — `UnifiedDatasetWriter` takes an injected cross-validator

**Agreed:** Day 1, 3-way.
**Spec touched:** SAD Figure 4, SRS 3.1.8.

SAD Figure 4 shows records flowing `DataCleaner → CrossValidator →
UnifiedDatasetWriter`. The plan schedules M1's `CrossValidator` and M2's writer
for the same Day 3 merge window, which makes each block the other.

The writer therefore depends on `CrossValidatorProtocol`
(`ceynex/contracts/protocols.py`) rather than on M1's concrete class, and
defaults to `NullCrossValidator`, which flags nothing. The real validator is
injected at the pipeline entrypoint once it lands; the writer does not change.
The runtime data flow is exactly the one in Figure 4 — only the binding is late.

---

## D4 — Comtrade partner exclusions cover World as well as the EU aggregate

**Spec touched:** SRS 3.1.7, team plan Day 2.

The M2 plan names the EU aggregate row (`partner = 97`) as the double-counting
trap. Comtrade also emits **World** as `partner = 0`, which aggregates every
partner and so double-counts far more aggressively. Both are excluded at the
connector, and both exclusions are asserted in `tests/data/test_crosswalk.py`.
Any market-share figure computed with either row present is wrong.

---

## D5 — a second unique index on `fact_trade` for the writer to upsert against

**Spec touched:** team overview §4.2, SRS 3.10.2. **No contract file edited.**

The frozen DDL declares

```sql
UNIQUE (source_id, item, hs_code, reporter_iso3, partner_iso3, period_start, frequency)
```

and both `hs_code` and `partner_iso3` are nullable. Postgres treats NULLs as
distinct inside a unique constraint, so two rows that differ in nothing but a
NULL `partner_iso3` do not conflict. The world-partner rows would therefore never
match `ON CONFLICT`, and every re-ingest would insert duplicates rather than
update — silently, since nothing raises.

`ceynex/data/bootstrap.py` adds

```sql
CREATE UNIQUE INDEX IF NOT EXISTS fact_trade_upsert_key
    ON fact_trade (source_id, item, hs_code, reporter_iso3,
                   partner_iso3, period_start, frequency)
    NULLS NOT DISTINCT
```

and `UnifiedDatasetWriter` upserts against `fact_trade_upsert_key` by name. The
contract's own constraint is untouched, so this is an addition rather than a
contract change. `NULLS NOT DISTINCT` requires Postgres 15+; the dev stack and
the deployed VM both run 18.

---

## D6 — schemas are applied from code, not from a docker initdb mount

**Spec touched:** SAD §7 (deployment), SRS 3.10.

The dev stack mounted `schema.sql` into `/docker-entrypoint-initdb.d/`. That hook
only ever fires on a first-boot empty volume, so it cannot apply anything to a
database that already exists — and the deployed database VM's compose had no
initdb hook at all, meaning the schema was never applied there by any mechanism.

`schema.sql` and `schema.cypher` now ship as package data inside
`ceynex-contracts` and are applied by `ceynex.data.bootstrap` (`make db-init`)
and `ceynex.kg.load` (`make kg-load`). One code path serves a developer's local
stack and the VM, and both are idempotent, so re-running them against a database
two teammates are already loading into is free.

---

## D7 — `ceynex` split across two distributions as a namespace package

**Spec touched:** none — this is a packaging decision, not an architectural one.

The frozen contracts moved to their own repository, `ceynex-contracts`, so that
the 3-way approval rule is enforced by pull-request review rather than by
everyone remembering it. `ceynex` is therefore an implicit namespace package:
`ceynex-contracts` supplies `ceynex.contracts`, `ceynex-core` supplies
`ceynex.data`, `.kg`, `.models`, `.agents`, `.orchestrator`, `.llm` and `.api`.

Import paths are unchanged — `from ceynex.contracts import AgentState` still
works — so no teammate code needed editing. Neither repo ships a
`ceynex/__init__.py`; adding one back shadows the other distribution.
`tests/test_layout.py` guards it.

---

## D8 — Comtrade variant partner codes are aliased, not treated as unknown

**Spec touched:** SRS 3.1.7, 3.6.1. **Found by running the connector, not by reading the spec.**

The M2 plan names the EU aggregate (`partner = 97`) as *the* Comtrade partner
trap, and D4 added World (`partner = 0`). Both are about rows that must be
**excluded**. There is a third case, and it fails in the opposite direction.

Comtrade does not use ISO 3166-1 numeric codes for territories it reports
together with their dependencies. It reports the USA as **842** (ISO: 840),
France as **251** (250), India as **699** (356), Switzerland as **757** (756) and
Norway as **579** (578). None of these resolve against a plain M49 table, so a
crosswalk built only from ISO 3166-1 drops them as unknown partners.

Measured on a 2021–2023 pull of HS 0902/0906/4001/61/62:

| Code | Country | Rows | Export value dropped |
|---|---|---:|---:|
| 842 | USA | 15 | USD 6,658,410,898 |
| 251 | France | 15 | USD 342,438,422 |
| 699 | India | 15 | USD 198,045,891 |
| 757 | Switzerland | 12 | USD 23,042,363 |
| 579 | Norway | 12 | USD 12,846,506 |
| | **total** | | **USD 7,234,784,080** |

The USA is Sri Lanka's largest apparel market. Losing it silently would have made
every market-share, top-partner and CAGR figure in the system wrong, with nothing
raising — the same failure mode as the EU aggregate, in the opposite direction.

`ceynex/data/reference/partner_aliases.csv` maps these onto their ISO-3 codes and
`to_iso3()` consults it. `tests/data/test_comtrade_connector.py` asserts all
three of USA, India and France survive the mapping, and a further test asserts
the committed fixture still contains the traps, so regenerating the fixture
cannot quietly disarm them.

**Sanity check, human-verified once:** the 2023 pull sums to USD 1.27bn of tea
exports against a published figure of roughly USD 1.3bn, and USA/UK/Italy/Germany
come out as the top four apparel destinations, which is the expected ordering.

---

## D9 — `WITSConnector` is not built

**Decided 2026-08-19, M2, under schedule pressure.**
**Spec touched:** SAD Figure 4 and §5.1, SRS 3.1.7, 3.1.8.

SAD Figure 4 names six concrete `DataSourceConnector` subclasses. Five are being
built — `ComtradeConnector` (M2), `FAOSTATConnector` and `CBSLConnector` (M1),
`JAAFConnector` and `EDBConnector` (M3). `WITSConnector` is not, and no static
tariff table replaces it either.

**Why.** The M2 plan names WITS as the first thing to cut and the 30-question
evaluation as something that may never be cut. With the schedule roughly two
weeks behind, that trade is the plan's own instruction, taken deliberately
rather than by omission.

**Scope of the damage.** One thing, narrowly: the MFN tariff rate re-imposed in
a preference-loss simulation is a documented constant in
`config/elasticities.yaml` rather than a queried schedule. FX shocks and
user-specified tariff shocks are unaffected, and whether GSP+ *covers* a given
HS code is still answered from the knowledge graph. Full reasoning, including
what the agent does when coverage is absent, is in
[DEFERRED.md](DEFERRED.md#wits-tariff-ingestion--cut).

**Reversal cost is low by construction:** add the connector and register it in
`CONNECTORS` in `ceynex/data/pipeline.py`. The writer, schema and agents do not
change — which is the point of the `DataSourceConnector` ABC.

---

## D10 — a fourth datastore: Qdrant, anchored by the graph

**Decided 2026-08-28, M2.**
**Spec touched:** SAD §8 (layer rules), SRS 3.1.4, 3.1.5, 3.1.9.

The SAD's Knowledge Layer is Neo4j and its Data Layer is Postgres plus Parquet.
A vector store is neither, and adding one is a real deviation rather than an
implementation detail.

**Why.** Every answer the system gave was derived from Sri Lanka's own trade
flows. Nothing in it knew what a *destination market* does — the US tariff
schedule, the UK's post-DCTS preferences, EU non-tariff measures on spices — so
any question about them had no grounding at all, and the preference-loss
simulations D9 left resting on a literature constant had no second source to
check against.

**What was added.** A Qdrant collection (`ceynex_policy`) holding chunked
trade-policy documents for Sri Lanka's top 10 export destinations, reached only
through `ceynex/retrieval/client.py::PolicyRetriever` — the same single-entry
rule the SAD's layer rules impose on Neo4j and Postgres, for the same reason.

**The direction of the dependency is the design.** Qdrant does not answer
questions; it answers *passages*. Which documents are eligible is decided in
Cypher first, by `kg/queries.py::policy_documents_for()`, and the resulting
`doc_id` list is passed to Qdrant as a filter. Trade-policy documents read alike
by construction — objectives, market access, competitiveness — so an unanchored
similarity search returns the right topic from the wrong country. Measured on
the first live run: a question about US tariffs on knitwear returned the
corpus's ABBREVIATIONS page, which matched only because it contains the words
"United States dollars".

**The graph stores a pointer, not the text.** `:PolicyDocument` carries
`doc_id`, provenance, `sha256` and `qdrant_collection`; the passages live only in
Qdrant. Two copies of the text would be two things that can disagree about what
the corpus contains.

**Why not put the chunks in Neo4j and skip the deviation.** It would work at
this scale — 544 chunks — and it was considered. It was rejected because the two
stores are being asked different questions: Cypher answers "which documents are
about the United States and HS 61" exactly, and a vector index answers "which
passage is about *this*" approximately. Neo4j Community has no vector index, so
the approximate half would become a full scan with a cosine computed in Cypher.

**Cost, stated plainly.** A fourth service in `docker-compose.yml`, an optional
`[policy]` dependency extra, and roughly 300–700 ms added to an agreement or
tariff simulation — inside the 2 s ceiling `RETRIEVAL_TIMEOUT_S` enforces, but
spent on a path whose p95 already breaches SRS 3.4.1 (EVALUATION.md §1). FX
shocks skip retrieval entirely, which is most simulation traffic.

**Reversal cost is low.** Unset `QDRANT_URL`, or set
`CEYNEX_POLICY_RETRIEVAL=off`. `PolicyRetriever.from_settings()` returns None and
`trade_economics` answers exactly as it did before — the path
`make eval-policy-baseline` measures, and the one
`test_no_retriever_configured_behaves_exactly_as_before` holds.

**The contract change this needs is not applied.** `:PolicyDocument` requires a
constraint in `ceynex-contracts`' frozen `schema.cypher`, which is a three-way
approval. The proposed diff is in
[CONTRACT_PROPOSAL_POLICY_DOCUMENT.md](CONTRACT_PROPOSAL_POLICY_DOCUMENT.md);
the loader works without it, so the PR is not on the critical path.

---

## D11 — a news sidecar over the same Qdrant, deliberately not evidence

**What the SAD says.** Nothing. There is no news source in the SRS, no current-
events requirement, and no scheduled component in the SAD's deployment view.
This is an addition, not a deviation from a stated design, and it is recorded
here because an addition nobody wrote down is drift.

**What was built.** `GET /api/news/search` runs a GDELT DOC 2.0 query alongside
each submitted question; `GET /api/news/trending` serves a precomputed panel of
fourteen watchlist topics ranked by 24-hour article volume against their own
7-day baseline. An hourly in-process task refreshes the panel and indexes the
headlines into a Qdrant collection.

**The load-bearing constraint: news is never evidence.** SRS 3.1.4 says every
claim beside an answer must be checkable against a verified source. A headline
from an outlet nobody vetted is not one, so the guarantee is enforced in four
independent places rather than by convention:

1. **A separate collection.** `ceynex_news`, never `ceynex_policy`.
   `NewsStore.__init__` raises `ValueError` if the two names match. Sharing one
   collection with a `doc_type` discriminator was considered and rejected:
   `retrieval/client.py` *widens its own filters* when a goods-filtered search
   returns nothing, so "remember to exclude news" would have to survive a code
   path designed to relax constraints — and forgetting once is silent.
2. **A separate response type.** `NewsArticleItem` has no `source_id`, no
   `claim` and no `detail`. `NewsArticle` has no `.citation` and no
   `.to_evidence()`; `tests/news/test_schema.py` asserts their absence.
3. **The frozen contracts.** `SourceId` in `ceynex-web/src/types/contracts.ts`
   is a closed union and `ceynex-contracts` sits behind a three-reviewer PR, so
   emitting news as `Evidence` is *structurally* impossible without that PR.
   The gate normally read as friction is, here, the enforcement mechanism.
4. **On screen.** The panel carries the sentence "Recent coverage from GDELT.
   Not used to produce the answer above", and is `print:hidden` — the printed
   report is the evidence-backed answer.

**Headlines only, no article bodies.** GDELT returns metadata and a title; it
does not return text. Fetching each article would mean scraping several hundred
arbitrary outlets — slow, fragile, and squarely into publishers' copyright.
GDELT's own terms permit non-commercial use with attribution, which the
`DATA_SOURCES` footer entry provides. The cost is that a dense vector is built
from 5–12 words, which is why the relevance floor is not retrieval's.

**Not folded into `POST /api/query`.** That path's p95 is 14.6 s against SRS
3.4.1's 10 s budget (EVALUATION.md §1). Adding an outbound HTTP call to it would
make a documented problem worse for every query, including those nobody wanted
news for, and would change `QueryResponse`, whose field names the frontend
depends on. The browser fires the two in parallel instead — a property of the
*frontend*, so `routes/news.py` carries a comment saying so.

**An in-process task instead of a scheduler.** No Celery, no APScheduler, no
cron exists anywhere in this project, and adding one for 28 HTTP calls an hour
would be a component the SAD does not have. The cost is a Redis lock, because
`ceynex-infra/backend/Dockerfile` runs `uvicorn --workers 2` and both workers
run the same lifespan. Without `REDIS_URL` the lock is a no-op, which is correct
for one worker and logged rather than assumed.

**A latent bug fixed on the way.** `Runtime.warmup()` was defined with a
docstring explaining it must run before the first request, and `main.py`'s
lifespan never called it. Deployed, the first policy query paid several seconds
of ONNX loading against a 2 s budget, degraded, and never reproduced once warm.
It is now started as a task — not awaited, because on a cold `fastembed_cache`
volume it downloads several hundred megabytes and the container healthcheck
allows roughly 95 s before it starts killing the process.

**Cost, stated plainly.** A second Qdrant collection (~30 days' retention, a few
hundred thousand points), one outbound call per user query and 28 per hour from
the refresher, and a `news_trending_snapshot` table in Postgres. No new
dependency, no new container, no new volume — `.cache/news/` and
`data/raw/gdelt/` sit inside volumes the backend already mounts.

**Egress is the deployment risk.** The backend VM has no external IP and
`ceynex-infra/gcp/` configures ingress only, so reaching `api.gdeltproject.org`
depends on the same NAT path the OpenAI client already uses. If the deployed LLM
works, this works; the compose file carries the one-line check.

**Reversal cost is low.** `CEYNEX_NEWS=off`. Both endpoints stay up and report
`unavailable`, the refresher returns immediately, and the query flow is
untouched — it never depended on either.

---

## D12 — an SSE transport beside the request/response endpoint

**Decided 2026-09-10, M2.**
**Spec touched:** SRS 3.4.1, 3.4.3, 3.9.1; SAD §8 (layer rules).

The SRS says *"no persistent socket based or streaming protocol is required for
standard query submission and response."* Not required is not forbidden, but it
means a stream is an addition beyond the written design, in the same class as
D10 and D11, and recorded the same way.

**Why.** `EVALUATION.md` §1 records single-sector p95 at 14.6 s against SRS
3.4.1's 10 s budget, with a 29.0 s cold tail. Streaming makes nothing faster. It
moves time-to-first-paint from the whole wait to the first frame and turns a
documented budget breach into progressive disclosure. Everything it shows was
already being produced and thrown away: `kg/client.py::run` has always returned
`(rows, cypher_text)` so the query could be cited, the policy retriever has
always returned its filter description, and the LLM client has always read
`prompt_tokens`/`completion_tokens` off every response.

**Nothing in the stream is invented.** Every event is emitted from a call site
that actually ran, carrying its real payload and real duration. An earlier draft
reserved one concession — holding a completed step on screen for a ~250 ms
minimum so a 40 ms Cypher query did not flash past. It was built and then
removed: rows are appended and never removed, so nothing flashes past, and
keeping the constant would have meant a docstring describing a delay the code did
not apply. `tests/observability/test_trace_is_truthful.py` asserts it in both
directions — every traced query ran, and every query that ran was traced.

**A context-scoped bus, not `astream_events`.** LangGraph's own event stream sees
node boundaries; the interesting facts happen three layers below one. So
`ceynex/observability/` holds a `ContextVar` set before `ainvoke()`, and `emit()`
is a no-op when it is unset — which is why `demo.py`, `eval/harness.py` and every
pre-existing test are untouched. This rests on LangGraph copying the caller's
context into each parallel node task, true in 1.2 via the private
`pregel/_executor.py`. `langgraph` is therefore pinned `>=1.2,<1.3` and
`tests/observability/test_context_propagation.py` is the canary to re-run before
widening it.

**SSE, not WebSocket, is an infrastructure fact.**
`ceynex-infra/frontend/nginx.conf.template`'s `location /api/` sets no
`Upgrade`/`Connection` headers, so a WebSocket upgrade cannot cross the proxy
today. Two nginx defaults would each have broken SSE silently, and both are
handled in code rather than left to a deploy: `proxy_buffering` is on by default,
defeated with `X-Accel-Buffering: no`; and `proxy_read_timeout` is 60 s, kept
alive by a 15 s heartbeat comment with `STREAM_BUDGET_S = 45.0` finishing first so
our own `error` frame is what a pathological request produces rather than the
proxy dropping the socket. Verified against the *unmodified* production block in
a real nginx container: first frame at 3 ms, last at 1.85 s, a 1.41 s gap between
— buffering would have delivered all 18 together — and nginx consumed the header,
which is how it signals it acted on it. **No infrastructure change is needed to
deploy this.**

**One orchestration path, two transports.** `submit_query()`'s body was lifted
into `ceynex/api/query_runner.py::run_query()`, which both routes now call. The
extraction is behaviour-preserving, so `tests/api/test_query.py`'s fixtures pass
unmodified, and
`test_the_done_frame_carries_the_same_answer_as_the_json_endpoint` is the guard
that the two never diverge. It also applied a timeout that was declared and never
used: `REQUEST_TIMEOUT_S = 25.0` was never passed to anything, so the real ceiling
was nginx's 60 s — which is why `EVALUATION.md` records a 29 s tail rather than a
25 s failure.

**A deliberate departure from `history.py`'s own pattern.** That module calls
`psycopg.connect()` synchronously inside an async handler, which is fine for a
bounded one-shot request and not fine here: a blocking call stalls the event loop
while a stream is meant to be emitting heartbeats, and with `--workers 2` it
stalls every other request on the process. So the live trace never touches the
database — events stream from an in-memory queue and the whole trace is persisted
once per turn in a single multi-row insert, best-effort — and every new write path
goes through `asyncio.to_thread`. One driver, one pattern, no new dependency.

**Cost, stated plainly.** Six tables in Postgres, no new service, no new
dependency. The planner is the only added LLM call on this path and it is
**skipped entirely when nothing is listening** — see D15's note on why that
matters more than it sounds.

**Reversal cost is low.** `CEYNEX_CHAT=off` removes the conversational surface
and `POST /api/query` answers exactly as before, which is asserted rather than
assumed.

---

## D13 — conversation state in Postgres, and a pre-routing clarification gate

**Decided 2026-09-10, M2.**
**Spec touched:** SRS 3.1.11, 3.4.6, 3.5.2, 3.10.

One question answered once is not how anyone works. The answer to "now do the
same for rubber" was to retype the whole question.

**Classify before you re-run.** Most follow-ups do not need a five-agent fan-out.
One cheap `turn_classify` call splits them: a `discuss` turn ("what does HHI
mean?", "summarise that in three bullets") is answered from the turn's existing
answer, evidence and figures with no graph re-run at all; an `analyse` turn is
rewritten as a standalone query and streamed as normal. Measured: a `discuss`
turn is **3 SSE frames against 22**, with no fan-out. That one decision governs
the cost and latency profile of the whole feature. Grounding still applies —
`orchestrator/grounding.py::ungrounded_figures` runs against the conversation's
own evidence, so a chat reply cannot introduce a figure the analysis never
produced. The separate `condense` role is configured and unused: the classifier
already returns the rewrite, and a second call to reword what the first just read
would double the cost of every re-analysis for nothing.

**A gate before routing, not LangGraph `interrupt()`.** Native HITL needs
`.compile(checkpointer=...)`, a `thread_id` per invocation and a resume command.
That is the right answer for pausing mid-graph, and we do not need mid-graph
pausing: the ambiguity we can actually detect — two commodities named, a
simulation with no resolvable destination — is knowable *before* routing, from
`parse_intent` gaps and the `RouteDecision`. Three further reasons the
checkpointer is the wrong tool here: `langgraph-checkpoint-postgres` is not
installed; `build_graph()` compiles one graph once at startup and shares it across
every request, so enabling checkpointing there makes *every* query write
checkpoint rows after every superstep, the wrong cost shape for a feature required
to be rare; and the natural call site would entangle clarification into
`graph.py`, the one module whose docstring is most insistent about staying exactly
`route → fan-out → merge → END`. The gate runs before `ainvoke()` is ever called,
so `graph.py` is not touched at all.

**The one-round cap is structural, not a counter.** Resume composes the answered
query and runs it without the gate on that path, so it cannot drift out of sync
across the two uvicorn workers the way an in-memory counter would. The single
condition that would flip this decision is an agent discovering ambiguity *after*
the graph has started, which genuinely cannot be a pre-check.

**Conversations require authentication; `/api/query` does not.** SRS 3.1.11
requires an account before query submission, and the frontend honours it on both
surfaces. The backend's anonymous support on `/api/query` is a documented
deferred-scope decision for that endpoint, not a UI affordance — a chat page
offering an anonymous path was written and removed for exactly this reason. A
conversation is stateful and has a `BIGSERIAL` id, so an anonymous caller
supplying someone else's `conversation_id` would be a plain IDOR; every store
function filters `WHERE id = %s AND user_email = %s` and returns 404 without
distinguishing "not found" from "not yours", the same posture as
`history.set_saved()`.

**Rate limiting is not inherited.** `enforce_rate_limit` was wired to
`/api/query` alone, and the streaming route invokes the identical fan-out, so
omitting it would have been a bypass around the most expensive call in the
system. `/api/chat/*` has its own `chat:`-namespaced allowance in
`config/api.yaml`, applied in the same commit that created the route.

**`query_history` is untouched.** Every chat turn that runs the graph writes its
existing `query_history` row exactly as before, so `GET /api/history`, the saved
star and the History panel need no change; `chat_message.query_history_id`
cross-links a turn to its row so a chat "save" calls the existing endpoint. A
`discuss` turn writes none — it is not a new analysis.

**A real schema defect, found by a leaking test purge.** `chat_trace_event` was
created without a foreign key, so deleting a conversation left every Cypher query,
token count and timing behind for good — which is precisely what the delete
endpoint's own docstring says it does not do. Fixed with an idempotent `DO $$`
block, because `CREATE TABLE IF NOT EXISTS` is a no-op against a table that
already exists and Postgres has no `ADD CONSTRAINT IF NOT EXISTS`; verified
against a table that already existed without it. Cited here as **SRS 3.10's
referential-integrity requirement** rather than as a data-subject erasure right —
3.10 is Database Requirements and states no such right, and the promise being kept
is the endpoint's own.

**Cost, stated plainly.** Three tables, one cheap classifier call per follow-up
turn, and one title call per conversation. Reversal is `CEYNEX_CHAT=off` for the
surface and `CEYNEX_CLARIFY=off` for the gate alone — separable on purpose, so a
demo can have conversation without ever being interrupted by a question, and a
reviewer comparing answers against `queries.md` can turn the gate off without
losing chat.

---

## D14 — general web search as enrichment, never as an agent

**Decided 2026-09-10, M2.**
**Spec touched:** SRS 3.1.2, 3.1.9, 3.6.4; SAD §4.1.

Everything CeyNex knows is Sri Lanka's own trade record plus a fixed policy
corpus. A question whose answer moved last week has no grounding at all, and the
news sidecar (D11) is deliberately not evidence and deliberately headline-only.

**Not a sixth agent, and this is enforced by construction rather than by
convention.** SRS 3.6.4 fixes the agent count at five and
`test_five_agents_exactly` asserts it. The provider therefore lives on `Runtime`,
never on `AgentDeps` — the same structural guarantee `Runtime` already applies to
`gdelt`/`news`, whose comment says it outright: *"no agent may reach news, and the
way to guarantee that is for it never to be handed to one."* No `AgentState` key
is added and `ALL_AGENTS` is untouched.

**Web evidence is appended after `merge()` has already returned.** That ordering
is the whole safety argument, and it buys three guarantees without a single check
having to be written:

- it **never reaches the merge LLM**, so the prose cannot cite a web figure as
  though it came from the graph;
- it **never enters `grounding.py::ungrounded_figures()`**, which is a pure
  digit-string presence test with no notion of source. Blending corpora would let
  a number in a scraped page *launder* a KG-attributed claim — defeating the one
  check the SRS cares most about;
- it **cannot move `confidence.py::aggregate_confidence()`**, which only iterates
  `Mapping[AgentName, AgentOutput]`, and web search is not an `AgentName`.

This supersedes an earlier plan to apply a confidence *penalty* to WEB findings.
Structural exclusion is strictly better: there is nothing to penalise if web
figures never enter the arithmetic in the first place.

**One hole this ordering does not close on its own.** `chat/turn.py::_corpus`
builds a `discuss` turn's grounding corpus from the stored message's entire
evidence list. Once a WEB entry is persisted on a message, a follow-up could
ground a figure on a scraped page — laundering by the back door, one turn later.
`_corpus` therefore filters `source_id == "WEB"` explicitly, and a test asserts a
web figure in a discuss reply is rejected.

**Recency is gated before any call is made.** `_wants_current_context()` is a
cheap keyword test in the same shape as the existing `FORECAST_WORDS`, so a purely
historical question makes no outbound request at all. That bounds cost and
injection surface in one decision.

**Untrusted content, contained by not feeding it to a model.** Web text reaches no
LLM in v1 — snippets only, no full-page fetch, which also matches this project's
recorded reason for not scraping news articles. The second layer is a frontend
rule: `source_id="WEB"` text renders as plain text, never HTML, and is given a
visually distinct chip, because teal means "verified source" everywhere else in
this UI and an unvetted web result must not borrow it.

**No keyless fallback, and the earlier note promising one is corrected.**
`settings.web_search_enabled()`'s docstring said an absent key falls back to a
keyless provider; the design note said an absent key means the system answers
exactly as it does today. Both could not be true. A scraped keyless provider is
fragile and widens the injection surface for little gain, so `from_settings()`
returns `None` without a key — the same three-way `None` as
`PolicyRetriever.from_settings()` and `NewsStore.from_settings()`.

**Cost, stated plainly.** No new dependency — Tavily is reached over `httpx`,
which is already required, rather than by adding `tavily-python`. One outbound
call on recency-flavoured queries only, hard-capped by its own timeout and
throttled through the Redis window the rate limiter already provides.

**Reversal cost is zero.** `CEYNEX_WEB_SEARCH=off`, or simply no
`TAVILY_API_KEY`. That the answers are then byte-identical to the pre-web-search
system is a checkable claim, and it is checked.

---

## D15 — per-user token and cost metering in Postgres

**Decided 2026-09-10, M2.**
**Spec touched:** SRS 3.4.6, 3.4.7, 3.5.4.

`LLMReasoningClient` has always read `prompt_tokens`/`completion_tokens` off every
response to compute a cost, and has always thrown both away afterwards. What
survived was a single process-global counter.

**One row per LLM *call*, not per request.** It survives a mid-request crash —
partial spend is still recorded, which is the thing a cost ledger exists for — and
it makes every rollup a plain `GROUP BY` rather than a nested-JSON blob to parse.
`llm_usage` carries role, model, provider, cache hit, fallback, failure, tokens in
and out, cost and elapsed time.

**The cap's real weakness, named rather than papered over.**
`LLMReasoningClient._cap_reached()` reads the per-process `usage.cost_usd`, and
`ceynex-infra/backend/Dockerfile` runs `uvicorn --workers 2` — each worker builds
its own client. **True daily spend can therefore reach 2× `daily_spend_cap_usd`.**
`ledger.spent_today()` is cross-worker accurate and is deliberately *reporting
only*: wiring it into enforcement would put a database round trip in front of
every LLM call. The real fix is a Redis-backed shared counter shaped exactly like
`rate_limit.py::RedisWindow` — a clean follow-up, and its own delta entry when it
lands.

**A cache hit is two different numbers, and only one of them is a gap.** For
*accounting*, a cache hit cost 0 tokens and $0, because no API call happened;
recording that is correct, not a shortfall. For *display* — "what would this have
cost" — the figure was genuinely absent, so `PromptCache.put()` now stores
optional original token and cost counts. The TTL is 168 h, so entries written by
the old `put()` are read by the new `get()` for a full week after any deploy;
every such field is read with `.get()`, never `[]`, or the first deploy would
`KeyError` on week-old cache entries.

**`conversation_id` is `ON DELETE SET NULL`, not `CASCADE`** — deliberately the
opposite choice from `chat_trace_event` (D13). A trace is *about* a conversation
and dies with it; a spend record is about money, and should survive the deletion
of the thing it was spent on. The link goes, the row stays.

**This is where SRS 3.4.6's disclosure requirement lands.** The requirement is not
merely that rate limits exist — it is that *"any usage restrictions … must be
disclosed to the user within the application rather than enforced silently."* The
usage page is that disclosure, so the feature closes a requirement rather than
decorating one.

**What it does not close.** SRS 3.4.7 wants an audit log of user queries *and
administrative actions*. The persisted trace and this ledger deliver most of an
actor/timestamp trail for query activity; admin actions are still unrecorded, and
`DEFERRED.md` remains the honest statement of that. An audit log that misses some
actions is worse than none, because it invites the reader to trust a record that
is not complete.

**Cost, stated plainly.** One table, one batched insert per request off the event
loop, four indexes. No new dependency and no new service.
