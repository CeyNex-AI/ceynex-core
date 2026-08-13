---
description: Scaffold a LangGraph agent node conforming to the AgentOutput contract
argument-hint: <agent_name> (one of the five AgentName literals)
---

Scaffold the agent node `$1`.

`$1` must already be one of the five literals in `AgentName`
(`ceynex/contracts/state.py`). SRS 3.6.4 fixes the agent set at five — if `$1`
is not among them, stop and say so rather than widening the contract.

Read `ceynex/contracts/state.py` and the "Agent node contract" section of the
root `CLAUDE.md` before writing anything.

Create:

1. `ceynex/agents/$1.py` with `async def ${1}_node(state: AgentState) -> AgentState`:
   - module docstring naming the SRS section it implements
   - writes exactly one key into `agent_outputs`, keyed `"$1"`
   - **never raises**: wrap the body in try/except, and on failure return
     `{"agent_outputs": {"$1": failed_output("$1", str(exc))}, "errors": [...]}`
   - returns **≥2 `Evidence`** entries naming the real source and period. For a
     KG-backed agent, `Evidence.detail` carries the literal Cypher string
   - derives `confidence` via `ceynex/orchestrator/confidence.py` — never a
     hardcoded constant — and documents the derivation in the docstring
   - degraded path: LLM unavailable → figures + evidence + templated summary,
     `degraded=True` (SRS 3.4.3)
   - returns only the keys it changed, so LangGraph's reducers merge it cleanly
     under parallel fan-out

2. `tests/agents/test_$1.py` covering:
   - a happy path asserting the output validates against `AgentOutput`
   - the LLM client monkeypatched to raise → `degraded is True`, node still returns
   - the data/KG dependency monkeypatched to raise → `error` set, `confidence == 0.0`,
     **node does not raise**
   - `len(output["evidence"]) >= 2`

3. Register the node in `ceynex/orchestrator/graph.py` if it is not already
   wired — one graph, always (SRS 3.6.4).

Then run `make lint && pytest tests/agents/test_$1.py`.
