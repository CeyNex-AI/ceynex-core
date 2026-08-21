# Deferred

Work that is consciously **not** in this repo, with what depends on it. The
point of writing this down is that a marker or a teammate can tell the
difference between a gap someone decided on and a gap nobody noticed.

Deviations from the SAD that *were* built live in
[ARCHITECTURE_DELTA.md](ARCHITECTURE_DELTA.md). This file is the other half:
things not built at all.

---

## Owned by M3, deliberately not pre-empted

`ceynex/api/` was seeded by M2 so the orchestrator is reachable for the
mid-evaluation demo, and is M3's from then on. Five routes exist — `GET
/health`, `POST /api/query`, `POST /api/auth/login`, `GET /api/auth/me`
(`ceynex/api/routes/auth.py`), and `GET /api/history`
(`ceynex/api/routes/history.py`) — and nothing else beyond that. Specifically
**not** built, because building them would mean guessing at M3's design and
then arguing about it:

| Deferred | Spec | Consequence today |
|---|---|---|
| Admin routes (retrain, ingest triggers, DQ review) | SRS 3.5.4 | Retraining is CLI-only (`make backtest`, the registry's `retrain()` hook). The hook exists so M3's endpoint is a thin wrapper, not a rewrite |
| Help and guidance content | SRS 3.5.5 | — |
| `web/` frontend | SAD §6 | The frontend VM is intentionally empty |

**Saved / bookmarked queries (SRS 3.5.2, second half) are built**: `saved` is
a column on `query_history` (`ceynex/api/history.py`), not a second table — a
saved query is a history entry, just flagged, and every query that could ever
be saved already has a row there from the moment it was asked. `POST
/api/history/{id}/save` and `.../unsave` toggle it, scoped to the caller's own
`user_email` in the same `UPDATE` (never a separate ownership check, same
reasoning as `auth.authenticate` never distinguishing "no such user" from
"wrong password") — a 404 covers both "no such entry" and "not yours", not
just the first. `GET /api/history?saved=true` filters the existing list route
rather than adding a second one.

**Login exists now; `POST /api/query` itself still does not require a token.**
`require_user` (`ceynex/api/routes/auth.py`) is ready for any route that needs
one, but wiring it onto `/api/query` was a deliberate choice left for later:
the frontend's four demo roles don't currently gate *what* a query can see,
only which UI pages render, so requiring a token there today would add a login
wall without changing any behaviour behind it. `POST /api/query` instead takes
an *optional* token (`get_optional_user`) — signed in or not, a query still
answers; being signed in only additionally attributes it to that user for
history. Revisit the hard requirement once a route actually needs to tell
users apart to change *what* it returns (the admin routes above are the first
candidate).

**Query history (SRS 3.5.2, first half) is built**: `ceynex/api/history.py`
records every authenticated query into a `query_history` table (additive to
the frozen contracts schema, not part of it — see that module's docstring for
why), and `GET /api/history` lists a signed-in user's own past queries. An
anonymous query, or one with an invalid/expired token, still answers
normally — it simply is not recorded, silently, by design.

## WITS tariff ingestion — cut

**Decided 2026-08-19. Spec touched: SRS 3.1.7, 3.1.8.**

The M2 plan lists WITS as the first thing to cut under schedule pressure, with a
static GSP+ table as the documented fallback. It is now cut outright: no
`ceynex/data/connectors/wits.py`, and no committed tariff table either.

**Why.** The schedule is roughly two weeks behind the written plan, and the
three things the plan says may never be cut — the orchestrator, the
single-graph constraint, the 30-question evaluation — are worth more marks than
a second connector. The orchestrator and the graph constraint are done; the
evaluation is not, and it is the headline deliverable. WITS is what pays for it.

**What actually degrades.** Less than it sounds, and precisely one thing:

- **FX shocks** are unaffected — no tariff rate is involved.
- **Explicit tariff scenarios** ("if the EU raises tariffs on tea by 10%") are
  unaffected: the rate comes from the question, not from WITS.
- **Preference-loss scenarios** ("if Sri Lanka loses GSP+") need an MFN rate to
  re-impose, and that is the number WITS would have supplied. It is currently a
  documented constant in `config/elasticities.yaml`
  (`agreement_loss_mfn_tariff`, default 9.5%), surfaced in the agent's
  `assumptions` list on every run rather than hidden as a literal.

So the honest description is: **preference-loss magnitudes rest on a literature
constant rather than a queried tariff schedule, and say so.** Whether GSP+
*covers* a given HS code is still resolved from the knowledge graph and is not
affected by this cut.

**What does not happen.** The agent does not silently invent coverage. With no
`TradeAgreement` coverage in the graph, `_simulate_agreement_loss` returns no
outcome and the agent reports that the simulation cannot be completed, citing
the Cypher that found nothing — SAD §4.1. Four tests in
`tests/agents/test_trade_economics.py` hold that line, including one asserting
the refusal cites the coverage query rather than some other query that happened
to be in scope.

**To undo the cut,** add a `WITSConnector(DataSourceConnector)` and register it
in `CONNECTORS` in `ceynex/data/pipeline.py`; nothing else changes shape.

## Data owned by teammates

`fact_trade` and the sector nodes are populated by M1 (agriculture) and M3
(apparel) through the same `UnifiedDatasetWriter` and `MERGE`-only loaders, so
their loads add to mine rather than colliding with them. M2 seeds only
`dim_country`, `dim_hs`, the `TradeAgreement` nodes, and the Comtrade extract.

## Measured but not measured under load

**SRS 3.4.2 wants 50 concurrent users. Nothing has tested that.** The latency
figures in [EVALUATION.md](EVALUATION.md) are single-user, sequential by design
so the numbers mean something per query. They pass their budgets with 5–15x
headroom, and that headroom is partly because no LLM key is configured, so the
system never pays for a model call.

Two separate things are therefore unverified: throughput at 50 concurrent users,
and latency with prose generation switched on. Both need re-measuring before any
claim about SRS 3.4.1/3.4.2 is made in the final report.

## Merge coherence not yet rated

`eval/coherence.py` builds the blind rating sheets and scores them. It needs
three human raters and the session has not happened, so SRS 3.1.2's "one
coherent answer, not a list of per-agent responses" is currently supported by
the merger's design and its unit tests, not by a measurement.

## Operational

- **`OPENAI_API_KEY` and `COMTRADE_API_KEY` are unset.** The system runs
  degraded by design (SRS 3.4.3): figures and evidence, no generated prose.
  Comtrade uses the keyless public preview endpoint, which returns real data at
  a lower rate limit.
- **SSH (22) and RDP (3389) are open to `0.0.0.0/0`** on the `ceynex-dev` VPC.
  Not closed unilaterally because restricting SSH to a single address could lock
  out two teammates. RDP serves no purpose on these Linux VMs and can be deleted
  safely.
- **Data-tier credentials have not been rotated** since being committed as
  `.env.example` values in `DevOps/`.
