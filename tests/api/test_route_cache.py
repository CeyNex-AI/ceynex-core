"""An answer that carried no evidence does not leave its route in the prompt cache.

The S07 case (docs/DEFERRED.md): the LLM router dropped an agent, the answer came
back with no evidence, and the prompt cache replayed that route for every later
ask of the same question for 168 hours. `run_query` is the one place that knows
both the question and whether its answer carried evidence, so it is where the
route is distrusted (`orchestrator/router.py::distrust_route`).
"""

from __future__ import annotations

import json

from ceynex.api import deps as deps_module
from ceynex.api.query_runner import run_query
from ceynex.llm import FakeLLMClient, LLMReasoningClient
from ceynex.llm.client import _CallOutcome
from ceynex.observability.spend import InProcessSpendCounter
from ceynex.orchestrator.router import ROUTER_SYSTEM, llm_route
from tests.api.test_query import ANSWERED, FakeGraph, FakeKG

QUESTION = "Which markets buy the most Sri Lankan knitted apparel?"
NO_EVIDENCE = {**ANSWERED, "merged_evidence": []}


def _runtime(final, llm):
    return deps_module.Runtime(kg=FakeKG(), llm=llm, deps=None, graph=FakeGraph(final))


async def test_an_answer_with_no_evidence_forgets_its_route():
    llm = FakeLLMClient()
    await run_query(_runtime(NO_EVIDENCE, llm), QUESTION, record_history=False)
    assert llm.forgotten == [("router", ROUTER_SYSTEM, QUESTION)]


async def test_an_answer_with_evidence_keeps_its_route():
    llm = FakeLLMClient()
    await run_query(_runtime(ANSWERED, llm), QUESTION, record_history=False)
    assert llm.forgotten == []


async def test_the_route_is_really_gone_from_the_cache(tmp_path, monkeypatch):
    """With the real client and on-disk cache: the router's cached response to an
    evidence-free answer is gone, so the next ask pays for routing again."""
    config = json.loads(json.dumps({
        "provider": "openai",
        "models": {"router": {"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 300,
                              "cost_per_1k_input_tokens": 0.0, "cost_per_1k_output_tokens": 0.0}},
        "limits": {"request_timeout_s": 8.0, "max_retries": 1, "daily_spend_cap_usd": 5.0},
        "cache": {"enabled": True, "path": str(tmp_path / "cache"), "ttl_hours": 1},
    }))
    llm = LLMReasoningClient(config=config, api_key="sk-test", spend=InProcessSpendCounter())
    calls = []

    async def answer(*args, **kwargs):
        calls.append(1)
        return _CallOutcome('{"route": ["apparel_manufacturing"], "sectors": ["apparel"]}', 0.0, 0, 0)

    monkeypatch.setattr(llm, "_call", answer)

    await llm_route(QUESTION, llm)
    await llm_route(QUESTION, llm)
    assert len(calls) == 1, "premise: a usable route is cached"

    await run_query(_runtime(NO_EVIDENCE, llm), QUESTION, record_history=False)

    await llm_route(QUESTION, llm)
    assert len(calls) == 2, "the route behind an evidence-free answer was still replayed"
