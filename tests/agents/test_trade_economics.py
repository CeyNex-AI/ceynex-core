"""Assertions for SRS 3.1.5 — the simulation agent, and its refusal path.

The refusal path is the one worth testing hardest. SAD §4.1 says that when the
graph cannot support a simulation the agent reports that rather than producing a
number, and a number produced from a missing premise is the single worst output
this system could give: it looks exactly like a real answer.
"""

import asyncio

import pytest

from ceynex.agents.common import AgentDeps
from ceynex.agents.trade_economics import AGENT, trade_economics_node
from ceynex.contracts import new_state
from ceynex.llm import FakeLLMClient

BASELINE_USD = 1_000_000.0


class KG:
    """A graph with baseline trade data and configurable agreement coverage."""

    def __init__(self, *, coverage: list[dict] | None = None, raises=None):
        self._coverage = coverage if coverage is not None else []
        self._raises = raises

    async def run(self, cypher, params=None):
        if self._raises:
            raise self._raises
        if "latest_year" in cypher:
            return [{"latest_year": 2024}], cypher
        if "COVERED_BY" in cypher:
            return list(self._coverage), cypher
        return [{"total_export_value_usd": BASELINE_USD}], cypher


GSP_PLUS = [
    {
        "agreement": "GSP+",
        "agreement_type": "unilateral_preference",
        "matched_on": "61",
        "agreement_verified": "verified",
    }
]


async def run(query: str, kg: KG):
    patch = await trade_economics_node(new_state(query, "test"), AgentDeps(kg=kg, llm=FakeLLMClient(available=False)))
    return patch["agent_outputs"][AGENT]


# --- the refusal path (SAD §4.1) -----------------------------------------


def test_missing_coverage_refuses_rather_than_inventing_a_number():
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))

    assert out["figures"] == {}, "a refusal that still reports an impact figure is not a refusal"
    assert any("cannot be simulated" in a for a in out["assumptions"])


def test_the_refusal_cites_the_query_that_found_no_coverage():
    """Evidence has to support its own claim, or grounding is theatre."""
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))

    coverage_claims = [e for e in out["evidence"] if "No trade-agreement coverage" in e["claim"]]
    assert coverage_claims, "the refusal did not say why it refused"
    assert "COVERED_BY" in coverage_claims[0]["detail"], (
        "the refusal cited some other query as evidence that coverage is missing"
    )


def test_a_refusal_still_meets_the_two_evidence_floor():
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))
    assert len(out["evidence"]) >= 2


def test_coverage_that_is_not_a_preference_is_also_a_refusal():
    """An FTA is not GSP+. Losing a preference you never had is not a shock."""
    fta = [dict(GSP_PLUS[0], agreement="ISFTA", agreement_type="bilateral_fta")]
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=fta)))
    assert out["figures"] == {}


def test_present_coverage_produces_a_simulation():
    """The mirror of the refusal tests: with coverage, a number is expected."""
    out = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=GSP_PLUS)))

    assert out["figures"], "coverage was present and the simulation still produced nothing"
    assert out["figures"]["apparel_impact_usd"] < 0, "losing a preference cannot raise revenue"


# --- the agent node contract ---------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "How would a 5% rupee depreciation affect apparel exports?",
        "What if the EU raises tariffs on tea by 10%?",
        "What happens to agriculture if Sri Lanka loses GSP+?",
    ],
)
def test_every_simulation_states_its_assumptions(query):
    """SRS 3.1.5 requires the assumptions, not just the number."""
    out = asyncio.run(run(query, KG(coverage=GSP_PLUS)))
    assert out["assumptions"], "a simulation with no stated assumptions is an unfalsifiable claim"


def test_the_node_writes_exactly_one_output_key():
    state = new_state("How would a 5% rupee depreciation affect apparel exports?", "test")
    deps = AgentDeps(kg=KG(coverage=GSP_PLUS), llm=FakeLLMClient(available=False))
    patch = asyncio.run(trade_economics_node(state, deps))
    assert set(patch["agent_outputs"]) == {AGENT}


def test_the_node_never_raises_when_the_graph_is_down():
    """SAD §4.1 partial-result guarantee."""
    out = asyncio.run(run("How would a 5% depreciation affect apparel?", KG(raises=RuntimeError("neo4j down"))))
    assert out["agent"] == AGENT
    assert out["confidence"] == 0.0 or out["figures"] == {}


def test_confidence_is_derived_rather_than_hardcoded():
    """Two runs with materially different evidence must not score identically."""
    refused = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=[])))
    answered = asyncio.run(run("What if Sri Lanka loses GSP+ for apparel?", KG(coverage=GSP_PLUS)))
    assert refused["confidence"] != answered["confidence"]


# --- policy retrieval (deviation D10) ------------------------------------
#
# The four tests above hold the D9 refusal line and must keep passing unchanged:
# adding a second source of tariff rates must not turn a refusal into a guess.


class PolicyKG(KG):
    """`KG`, plus the policy-document lookup the retrieval path runs first."""

    def __init__(self, *, documents=("USA-USTR-TARIFF-ACTIONS",), **kwargs):
        super().__init__(**kwargs)
        self._documents = documents

    async def run(self, cypher, params=None):
        if "PolicyDocument" in cypher:
            return [{"doc_id": doc_id} for doc_id in self._documents], cypher
        return await super().run(cypher, params)


class SpyRetriever:
    """Records what it was asked for and returns a fixed set of chunks."""

    def __init__(self, chunks=(), raises=None):
        self._chunks = list(chunks)
        self._raises = raises
        self.calls: list = []

    async def search(self, query, *, filters=None, limit=5):
        self.calls.append(filters)
        if self._raises:
            raise self._raises
        return list(self._chunks), filters.describe() if filters else ""


def policy_chunk(text: str):
    from ceynex.retrieval.schema import PolicyChunk

    return PolicyChunk(
        doc_id="USA-USTR-TARIFF-ACTIONS",
        chunk_index=0,
        text=text,
        title="Presidential Tariff Actions",
        publisher="USTR",
        url="https://example.invalid/tariff",
        page=7,
    )


MFN_TEXT = "Knitted apparel under HS 61 faces an MFN duty of 16.5% when preferences do not apply."


async def run_with(query: str, kg: KG, retriever):
    deps = AgentDeps(kg=kg, llm=FakeLLMClient(available=False), extras={"policy": retriever})
    patch = await trade_economics_node(new_state(query, "test"), deps)
    return patch["agent_outputs"][AGENT]


def test_no_retriever_configured_behaves_exactly_as_before():
    """`None` is the pre-retrieval system, and it has to stay byte-identical.

    `make eval-policy-baseline` measures against this path, so any drift here
    makes the before/after comparison meaningless.
    """
    query = "What happens to apparel exports if Sri Lanka loses GSP+?"
    without = asyncio.run(run("What happens to apparel exports if Sri Lanka loses GSP+?", KG(coverage=GSP_PLUS)))
    explicit_none = asyncio.run(run_with(query, KG(coverage=GSP_PLUS), None))

    assert without["figures"] == explicit_none["figures"]
    assert without["summary"] == explicit_none["summary"]


def test_an_fx_shock_never_calls_the_retriever():
    """A currency question needs no tariff schedule.

    Skipping it there is what keeps the commonest simulation on the latency it
    has today, which matters because single-sector p95 already breaches SRS
    3.4.1 (EVALUATION.md §1).
    """
    spy = SpyRetriever()
    asyncio.run(run_with("How would a 5% rupee depreciation affect apparel?", PolicyKG(coverage=GSP_PLUS), spy))

    assert spy.calls == []


def test_an_agreement_shock_anchors_the_search_on_the_graphs_entities():
    spy = SpyRetriever()
    asyncio.run(
        run_with(
            "What happens to apparel exports to the United States if Sri Lanka loses GSP+?",
            PolicyKG(coverage=GSP_PLUS),
            spy,
        )
    )

    assert spy.calls, "an agreement question must consult the policy corpus"
    filters = spy.calls[0]
    assert filters.iso3 == ("USA",), "the partner resolved from the query must scope the search"
    assert "61" in filters.hs_prefixes
    assert filters.doc_ids == ("USA-USTR-TARIFF-ACTIONS",), "Cypher must supply the allow-list"
    assert filters.is_anchored


def test_a_sourced_rate_replaces_the_constant_and_appears_in_evidence():
    """EVALUATION.md §1 grounding class 1 names this agent for figures that
    appear in no evidence. A new figure without a new evidence entry would
    repeat the bug the same run found."""
    out = asyncio.run(
        run_with(
            "What happens to apparel exports to the United States if Sri Lanka loses GSP+?",
            PolicyKG(coverage=GSP_PLUS),
            SpyRetriever([policy_chunk(MFN_TEXT)]),
        )
    )

    assert out["figures"]["apparel_mfn_tariff_pct"] == 16.5
    assert any(e["source_id"] == "POLICY" for e in out["evidence"])
    assert any("16.5" in e["claim"] for e in out["evidence"]), "the rate must be restated in evidence"
    assert any(e.get("url") for e in out["evidence"] if e["source_id"] == "POLICY")


def test_the_assumptions_say_which_rate_was_used():
    """Whichever source supplied the magnitude, the answer names it.

    A constant disappearing silently behind a sourced figure is the same defect
    as inventing one — the reader cannot tell how much to trust the number.
    """
    sourced = asyncio.run(
        run_with(
            "What happens to apparel exports to the United States if Sri Lanka loses GSP+?",
            PolicyKG(coverage=GSP_PLUS),
            SpyRetriever([policy_chunk(MFN_TEXT)]),
        )
    )
    fallback = asyncio.run(
        run_with(
            "What happens to apparel exports to the United States if Sri Lanka loses GSP+?",
            PolicyKG(coverage=GSP_PLUS),
            SpyRetriever([]),
        )
    )

    assert any("USTR" in a for a in sourced["assumptions"])
    assert any("literature constant" in a for a in fallback["assumptions"])
    assert sourced["figures"]["apparel_impact_usd"] != fallback["figures"]["apparel_impact_usd"]


def test_a_retriever_that_fails_does_not_fail_the_query():
    """Retrieval is an enhancement to an agent that already works without it."""
    out = asyncio.run(
        run_with(
            "What happens to apparel exports if Sri Lanka loses GSP+?",
            PolicyKG(coverage=GSP_PLUS),
            SpyRetriever(raises=RuntimeError("qdrant unreachable")),
        )
    )

    assert out["figures"], "a retrieval outage must not empty the simulation"
    assert not out.get("error")


def test_retrieval_never_rescues_a_refusal_into_a_number():
    """The D9 line, retested with the new source present.

    Policy text may be cited for context when the graph records no coverage, but
    it must not become the missing premise: a preference that the graph does not
    record is still not a preference that can be lost.
    """
    out = asyncio.run(
        run_with(
            "What happens to apparel exports to the United States if Sri Lanka loses GSP+?",
            PolicyKG(coverage=[]),
            SpyRetriever([policy_chunk(MFN_TEXT)]),
        )
    )

    assert out["figures"] == {}, "a refusal that reports an impact figure is not a refusal"
    assert any("cannot be simulated" in a for a in out["assumptions"])


# --- descriptive policy questions (D10, routing fix) ---------------------
#
# The branch exists because routing policy questions here *without* it made the
# system worse: `_classify_shock` fell through to `fx`, so "What does India's
# Foreign Trade Policy say about imports from Sri Lanka?" was answered with a 5%
# rupee depreciation and a figure of USD -8,240,802.


@pytest.mark.parametrize(
    "query",
    [
        "What does India's Foreign Trade Policy say about imports from Sri Lanka?",
        "What non-tariff measures does the European Union apply to imported spices?",
        "Does the Netherlands' foreign trade policy identify Sri Lanka as a priority market?",
        "Does the United Kingdom's trade strategy keep preferential access for Sri Lankan tea?",
        "Which trade agreement gives Sri Lankan cinnamon preferential access to the European Union?",
    ],
)
def test_a_question_about_what_a_policy_says_is_not_a_shock(query):
    from ceynex.agents.trade_economics import _classify_shock

    assert _classify_shock(query) == "policy"


@pytest.mark.parametrize(
    "query",
    [
        "What happens to apparel export revenue if Sri Lanka loses GSP+?",
        "What if the European Union raised tariffs on Sri Lankan tea by 10%?",
        "How would a 5% depreciation of the Sri Lankan rupee affect apparel exports?",
        "If the United States withdrew duty-free access for Sri Lankan knitted apparel, what tariff would apply?",
        "How much would apparel export revenue fall if the United Kingdom ended DCTS preferences?",
    ],
)
def test_a_question_that_posits_a_change_is_still_a_simulation(query):
    """The descriptive check must not swallow the simulations the agent exists for."""
    from ceynex.agents.trade_economics import _classify_shock

    assert _classify_shock(query) != "policy"


def test_a_price_driver_question_is_not_a_policy_lookup():
    """S06 in the 30-question set belongs to the agriculture agent.

    A bare "what is" marker classified it as a policy question; the markers are
    specific to policy instruments for this reason.
    """
    from ceynex.agents.trade_economics import _classify_shock

    assert _classify_shock("What is driving the recent movement in cinnamon prices?") != "policy"


def test_a_descriptive_question_reports_no_impact_figure():
    """The whole point of the branch. Nothing was shocked, so nothing moved."""
    out = asyncio.run(
        run_with(
            "What does India's Foreign Trade Policy say about imports from Sri Lanka?",
            PolicyKG(coverage=GSP_PLUS, documents=("IND-DGFT-FTP-2023",)),
            SpyRetriever([policy_chunk("India's FTP sets out import licensing for agricultural goods.")]),
        )
    )

    assert out["figures"] == {}, "a question about what a document says has no impact figure"
    assert not any("depreciation" in a.lower() for a in out["assumptions"])
    assert any(e["source_id"] == "POLICY" for e in out["evidence"])


def test_a_descriptive_question_anchors_on_the_destination_not_sri_lanka():
    """`parse_intent` resolves the longest country name, and "Sri Lanka" is long.

    Measured before the fix: this question anchored on LKA and answered about
    Indian policy with four passages from Sri Lanka's own export strategy.
    """
    spy = SpyRetriever()
    asyncio.run(
        run_with(
            "What does India's Foreign Trade Policy say about imports from Sri Lanka?",
            PolicyKG(coverage=GSP_PLUS, documents=("IND-DGFT-FTP-2023",)),
            spy,
        )
    )

    assert spy.calls
    assert spy.calls[0].iso3 == ("IND",), "the destination is India, never the reporter"


def test_an_empty_corpus_for_a_country_is_stated_with_its_cypher():
    """A gap the reader can check beats an answer from the wrong country."""
    out = asyncio.run(
        run_with(
            "Does the Netherlands' foreign trade policy identify Sri Lanka as a priority market?",
            PolicyKG(coverage=GSP_PLUS, documents=()),
            SpyRetriever([]),
        )
    )

    assert out["figures"] == {}
    assert "cannot be answered" in out["summary"]
    assert any(e["source_id"] == "KG" for e in out["evidence"]), (
        "the Cypher that found no document is the evidence for saying so"
    )
