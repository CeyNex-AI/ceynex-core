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
