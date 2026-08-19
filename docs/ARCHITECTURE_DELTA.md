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
