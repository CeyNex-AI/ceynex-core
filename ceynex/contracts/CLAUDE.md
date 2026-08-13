# ceynex/contracts/ — FROZEN

**Do not modify anything in this package without asking.** A change here is a PR
that all three members approve. It is the only thing in the repo with that rule,
and it is what lets three people build in parallel without waiting on each other.

If you think a contract needs to change, say so and stop. Do not edit and
mention it afterwards.

## What lives here

| File | Contents | Spec |
|---|---|---|
| `state.py` | `AgentState`, `AgentOutput`, `AgentName`, `Sector`, reducers | Team overview §4.1, SRS 3.1.2 / 3.6.4 |
| `evidence.py` | `Evidence` | SRS 3.1.4 |
| `forecast.py` | `ForecastPoint` | SRS 3.1.3 |
| `protocols.py` | `DataSourceConnector`, `CrossValidatorProtocol`, `ForecastModel`, KG + LLM client protocols | SRS 3.1.7 / 3.1.8 / 3.1.10, SAD Figs. 4 & 10 |

## The two things most likely to be broken by accident

1. **The `Annotated[...]` reducers on `agent_outputs`, `errors`, `degraded`.**
   They are load-bearing. LangGraph raises `InvalidUpdateError` when parallel
   branches write a state key with no reducer, and cross-sector queries fan out
   to 2–3 agents in parallel. `tests/contracts/test_state.py` guards this.

2. **`ForecastPoint.lower` / `.upper` being required.** SRS 3.1.3 forbids a
   forecast presented as an unqualified number. A model that cannot produce an
   interval gets one by residual bootstrap; it does not get to omit the field.
