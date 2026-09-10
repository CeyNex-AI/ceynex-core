"""Assertions for the plan shown before the fan-out (SRS 3.1.2, SRS 3.4.3).

The plan is a claim about what the system is about to do, so the tests that
matter are the ones about honesty: it must not name work that is not routed, it
must not state a conclusion it cannot have yet, and a model that returns
something malformed must produce the deterministic plan rather than half of a
generated one.
"""

from __future__ import annotations

import json

from ceynex.llm import FakeLLMClient
from ceynex.orchestrator.planner import (
    MAX_STEPS,
    MIN_STEPS,
    deterministic_plan,
    plan,
)


class ScriptedLLM:
    """Returns a canned payload for the planner role. Never touches the network."""

    available = True

    def __init__(self, payload):
        self.payload = payload
        self.roles: list[str] = []

    async def generate(self, role, system, user, *, json_mode=False):
        self.roles.append(role)
        return self.payload


# --- the deterministic plan (the degraded path, and the default safety net) ---


def test_the_deterministic_plan_names_the_agents_that_will_actually_run():
    """A step for work that is not routed is the same defect as a trace event for
    a query that never ran."""
    from ceynex.orchestrator.router import keyword_route

    query = "what is the price trend for cinnamon"
    steps = deterministic_plan(query)
    routed = keyword_route(query).route

    body = " ".join(steps).lower()
    if "agriculture_commodity" in routed:
        assert "agriculture" in body
    if "forecast" not in routed:
        assert "forecast model" not in body


def test_the_deterministic_plan_mentions_the_subject_of_the_question():
    steps = deterministic_plan("cinnamon exports to Germany in 2025")
    assert "cinnamon" in steps[0].lower()
    assert "germany" in steps[0].lower()


def test_the_deterministic_plan_always_ends_by_naming_the_merge():
    """Confidence and citation are the last thing that happens and the part a
    reader most needs to expect."""
    steps = deterministic_plan("cinnamon export trend")
    assert "confidence" in steps[-1].lower()


def test_the_deterministic_plan_stays_within_the_step_bounds():
    for query in (
        "cinnamon export trend",
        "how would losing GSP+ affect apparel revenue and tea prices in the EU",
        "hello",
    ):
        steps = deterministic_plan(query)
        assert 1 <= len(steps) <= MAX_STEPS, f"{query}: {steps}"


def test_a_plan_never_states_a_figure():
    """It describes checks, not findings — it runs before any data is read."""
    steps = deterministic_plan("cinnamon exports in 2025")
    for step in steps:
        assert "%" not in step
        assert "usd" not in step.lower()


# --- degraded mode --------------------------------------------------------


async def test_no_llm_key_still_produces_a_real_plan():
    """SRS 3.4.3. The reasoning display works with no provider at all."""
    steps, method = await plan("cinnamon export trend", FakeLLMClient(available=False))
    assert method == "deterministic"
    assert steps == deterministic_plan("cinnamon export trend")


async def test_a_missing_llm_entirely_is_handled():
    steps, method = await plan("cinnamon export trend", None)
    assert method == "deterministic"
    assert steps


# --- the LLM plan ---------------------------------------------------------


async def test_a_well_formed_llm_plan_is_used():
    llm = ScriptedLLM(json.dumps({"steps": [
        "Find which markets took Sri Lankan cinnamon",
        "Compare the last two years of export value",
        "Merge the findings and score confidence",
    ]}))
    steps, method = await plan("cinnamon export trend", llm)

    assert method == "llm"
    assert len(steps) == 3
    assert llm.roles == ["planner"], "must use its own role so its cost is separable"


async def test_trailing_full_stops_are_stripped():
    llm = ScriptedLLM(json.dumps({"steps": ["Check the graph.", "Merge findings.", "Score it."]}))
    steps, _ = await plan("cinnamon", llm)
    assert all(not step.endswith(".") for step in steps)


async def test_unparseable_json_falls_back_rather_than_repairing():
    """A half-parsed plan is worse than the deterministic one — it is shown with
    the same authority and describes work nobody chose."""
    steps, method = await plan("cinnamon export trend", ScriptedLLM("not json at all"))
    assert method == "deterministic"
    assert steps == deterministic_plan("cinnamon export trend")


async def test_a_plan_with_too_many_steps_is_rejected():
    llm = ScriptedLLM(json.dumps({"steps": [f"step {i}" for i in range(MAX_STEPS + 3)]}))
    _, method = await plan("cinnamon", llm)
    assert method == "deterministic"


async def test_a_plan_with_too_few_steps_is_rejected():
    llm = ScriptedLLM(json.dumps({"steps": ["only one"]}))
    _, method = await plan("cinnamon", llm)
    assert method == "deterministic"
    assert MIN_STEPS > 1


async def test_a_null_response_falls_back():
    steps, method = await plan("cinnamon", ScriptedLLM(None))
    assert method == "deterministic"
    assert steps


async def test_a_raising_llm_never_fails_the_query():
    class Exploding:
        available = True

        async def generate(self, *args, **kwargs):
            raise RuntimeError("provider on fire")

    steps, method = await plan("cinnamon export trend", Exploding())
    assert method == "deterministic"
    assert steps


async def test_the_method_is_reported_so_the_ui_can_say_which_it_was():
    """A reader who cannot tell a model's plan from a generated one cannot judge
    either. Which it was is a fact about the answer, not an internal detail."""
    llm_steps, llm_method = await plan(
        "cinnamon", ScriptedLLM(json.dumps({"steps": ["a", "b", "c"]}))
    )
    det_steps, det_method = await plan("cinnamon", FakeLLMClient(available=False))

    assert llm_method == "llm"
    assert det_method == "deterministic"
    assert llm_steps != det_steps
