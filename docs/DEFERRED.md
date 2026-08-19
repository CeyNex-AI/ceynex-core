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
mid-evaluation demo, and is M3's from then on. Two routes exist — `GET /health`
and `POST /api/query` — and nothing else. Specifically **not** built, because
building them would mean guessing at M3's design and then arguing about it:

| Deferred | Spec | Consequence today |
|---|---|---|
| Authentication and authorization | SRS 3.1.11 | **`POST /api/query` is unauthenticated.** Acceptable only because the backend is VPC-internal with no public route to port 8000; it must not stay true if the API is ever exposed |
| Admin routes (retrain, ingest triggers, DQ review) | SRS 3.5.4 | Retraining is CLI-only (`make backtest`, the registry's `retrain()` hook). The hook exists so M3's endpoint is a thin wrapper, not a rewrite |
| Query history and saved queries | SRS 3.5.2 | Each request is independent; nothing is persisted per user |
| Help and guidance content | SRS 3.5.5 | — |
| `web/` frontend | SAD §6 | The frontend VM is intentionally empty |

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
