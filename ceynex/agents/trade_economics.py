"""Implements SRS 3.1.5 — exchange-rate, tariff and agreement-loss simulation.

Answers "how would a 5% depreciation of the rupee affect apparel exports" with a
number, a direction, and — non-negotiably — the assumptions behind it. SRS 3.1.5
requires the assumptions to be stated, and `AgentOutput.assumptions` must be
non-empty for this agent specifically.

**KG-grounded, and it refuses rather than guesses.** Before simulating an
agreement loss it resolves coverage through `agreement_coverage()`. If the graph
does not know whether GSP+ covers the code in question, it reports that the
simulation cannot be completed rather than producing a number (SAD §4.1). A
confident figure resting on a coverage assumption nobody checked is worse than no
figure.

**The model is linear and says so.** Elasticities live in
`config/elasticities.yaml` — a documented table with `value`, `basis` and
`source` per entry — not buried in this file. Every one is currently marked as a
literature range rather than a fitted estimate, and the agent surfaces that in
its assumptions rather than implying a precision it does not have.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ceynex.agents.common import (
    AgentDeps,
    Intent,
    evidence_from_policy,
    evidence_from_query,
    finish,
    parse_intent,
)
from ceynex.contracts import AgentState, Evidence, failed_output
from ceynex.kg import queries as q
from ceynex.kg.client import KnowledgeGraphUnavailableError
from ceynex.kg.queries import hs_hierarchy
from ceynex.retrieval.rates import SourcedRate, extract_tariff_rate
from ceynex.retrieval.schema import SIMULATION_MEASURES, PolicyChunk, RetrievalFilter
from ceynex.retrieval.tagging import HS_FOR_ITEM, countries_in
from ceynex.settings import elasticity_config

log = logging.getLogger(__name__)

AGENT = "trade_economics"

SECTOR_OF_ITEM = {
    "tea": "agriculture",
    "cinnamon": "agriculture",
    "rubber": "agriculture",
    "coconut": "agriculture",
    "apparel_knit": "apparel",
    "apparel_woven": "apparel",
}

# Which shock the question is about. Checked in this order: an agreement question
# that also mentions a currency is still an agreement question.
SHOCK_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # "losing access to the United States market" is a preference loss, not a
    # currency move. Without these, X09 fell through to the `fx` default and was
    # answered with a 5% rupee depreciation — a figure about a shock the question
    # never mentioned, and one the grounding check duly flagged as unsupported.
    ("agreement", ("gsp", "gsp+", "agreement", "fta", "preference", "duty-free",
                   "duty free", "access to", "market access", "losing access")),
    ("tariff", ("tariff", "duty", "customs", "import tax")),
    ("fx", ("deprecia", "apprecia", "exchange rate", "rupee", "lkr", "currency", "devalu")),
)

# Words that posit a change. Their presence makes a question a simulation — the
# reader is asking what *would happen*, not what the rules currently are.
CHANGE_MARKERS = (
    "what if", "if the", "if sri lanka", "if a ", "if every", "were to", "would",
    "happens to", "happened", "impact of", "effect of", "suppose", "simulat",
    "raise", "raised", "raises", "withdrew", "withdraw", "withdrawn", "lose",
    "loses", "losing", "lost", "end ", "ends", "ended", "ending", "suspend",
    "remove", "increase", "cut ", "fell", "falls", "fall ", "change",
)

# Words that mark a question about what a policy *says*. Answered from the
# document corpus, not by simulating anything.
#
# This distinction exists because the alternative is much worse than a missing
# answer. `_classify_shock` used to fall through to `fx` for anything it did not
# recognise, so "What does India's Foreign Trade Policy say about imports from
# Sri Lanka?" was answered with a 5% rupee-depreciation simulation and a figure
# of USD -8,240,802 — a confident number about a currency move nobody mentioned,
# in reply to a question about a document. Routing such questions here without
# this branch would have made the system worse, not better.
# Deliberately specific. A bare "what is"/"what are" was tried and removed: it
# classified "What is driving the recent movement in cinnamon prices?" (S06, a
# question for the agriculture agent about price drivers) as a policy lookup.
# Every marker here names a policy instrument or asks what a document states.
DESCRIPTIVE_MARKERS = (
    "what does", "say about", "says about", "state about", "identify",
    "priority market", "priorities", "trade strategy", "trade policy",
    "foreign trade policy", "export strategy", "non-tariff", "measures does",
    "measures do", "rules of origin", "provisions", "does the", "do the ",
    "which trade agreement", "market access", "preferential access", "eligible",
    "what tariff", "which tariff", "what rate", "what duty",
    # Comparative forms. "How do Japan's tariffs on Sri Lankan tea compare with
    # Germany's?" asks what two schedules say; it proposes no change to either.
    # The CHANGE_MARKERS veto keeps the comparative *simulations* out -- M01
    # ("how would a 5% depreciation … compared to agriculture") and X09 ("hurt
    # more by losing access") both carry one.
    "compare", "how do ", "how does ",
)

#: What kind of measure a descriptive question is about, so retrieval can be
#: filtered to it. Checked in order; the first match wins.
DESCRIPTIVE_MEASURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ntm", ("non-tariff", "sanitary", "phytosanitary", "standards", "certification",
             "quota", "licensing", "technical barrier", "traceability")),
    ("tariff", ("tariff", "duty", "duties", "customs", "mfn")),
    ("fta", ("agreement", "gsp", "dcts", "fta", "preference", "preferential",
             "rules of origin", "duty-free", "duty free")),
    ("export_promotion", ("export promotion", "export development", "priority market",
                          "market access", "priorities", "strategy")),
    ("investment", ("investment", "fdi")),
)


async def trade_economics_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    try:
        return await _simulate(state, deps)
    except KnowledgeGraphUnavailableError as exc:
        log.warning("%s: knowledge graph unavailable: %s", AGENT, exc)
        return {
            "agent_outputs": {AGENT: failed_output(AGENT, f"knowledge graph unavailable: {exc}")},
            "errors": [f"{AGENT}: {exc}"],
            "degraded": True,
        }
    except Exception as exc:  # noqa: BLE001 - an agent that raises breaks partial results
        log.exception("%s failed", AGENT)
        return {
            "agent_outputs": {AGENT: failed_output(AGENT, str(exc))},
            "errors": [f"{AGENT}: {exc}"],
            "degraded": True,
        }


async def _simulate(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    config = elasticity_config()
    intent = parse_intent(state["query"])
    shock = _classify_shock(state["query"])
    magnitude = intent.pct_change if intent.pct_change is not None else 0.05

    # A tariff question that supplies no rate and posits no change is asking what
    # the tariffs *are*. Simulating it means inventing the input: `magnitude`
    # above falls back to 5%, and the answer reports the revenue effect of a
    # tariff move nobody proposed. Measured live 2026-09-03 on P08 ("which of Sri
    # Lanka's largest apparel markets has the most restrictive import tariffs?")
    # and P10 ("how do Japan's tariffs on Sri Lankan tea compare with
    # Germany's?") -- both got an unrequested 5% simulation, P08 at confidence
    # 0.8. The rate-carrying questions (P14's "by 15%", M03's "by 10%") and the
    # ones that posit a change without a rate are untouched: those have something
    # to simulate.
    #
    # `agreement` is deliberately not included. Losing a preference re-imposes a
    # rate the question does not have to supply -- `_simulate_agreement_loss`
    # sources it, and says so when it falls back to the D9 literature constant.
    if shock == "tariff" and intent.pct_change is None and not _posits_a_change(state["query"]):
        log.info("tariff question with no rate and no proposed change; describing policy instead")
        shock = "policy"

    # A question about what a policy *says* is not a shock, and answering it with
    # one produces a figure nobody asked for. See `_describe_policy`.
    if shock == "policy":
        return await _describe_policy(state, deps, intent)

    # Which sectors the question touches. Both, unless it named one.
    sectors = _sectors_for(state, intent.item)

    figures: dict[str, float] = {}
    evidence: list[Evidence] = []
    assumptions = _base_assumptions(config, shock, magnitude)
    lines: list[str] = []

    for sector in sectors:
        item = _representative_item(sector, intent.item)

        # Policy retrieval, per sector, and only where policy text can change the
        # answer. An FX question needs no tariff schedule — the shock comes
        # entirely from the currency move — so skipping it there keeps the
        # commonest simulation on exactly the latency it has today, which matters
        # because single-sector p95 already breaches SRS 3.4.1 (EVALUATION.md §1).
        chunks: list[PolicyChunk] = []
        retrieval_detail = "not attempted for an fx shock"
        sourced: SourcedRate | None = None
        hs_prefixes = tuple(hs_hierarchy(_hs_for_item(item)))

        if shock in ("agreement", "tariff"):
            context = await _retrieve_policy(
                deps,
                state,
                partner=_destination(state["query"], intent),
                hs_prefixes=hs_prefixes,
            )
            chunks, retrieval_detail = context.chunks, context.detail
            sourced = extract_tariff_rate(chunks, hs_prefixes) if chunks else None

        baseline, baseline_year, baseline_cypher = await _baseline_value(deps, item)

        if baseline is None:
            assumptions.append(
                f"No baseline export value for {sector} in the knowledge graph, so its "
                "impact could not be simulated."
            )
            continue

        # The Cypher that justifies a refusal is the coverage lookup, not the
        # baseline — citing the baseline query under a claim about coverage is
        # evidence that does not support its own claim.
        refusal_cypher = baseline_cypher
        if shock == "agreement":
            outcome, refusal_cypher = await _simulate_agreement_loss(
                deps, sector, item, baseline, config, sourced
            )
        elif shock == "tariff":
            # The magnitude of an explicit tariff question comes from the
            # question ("raise tariffs by 10%"), so a retrieved rate must not
            # override it — that would answer a different question than the one
            # asked. Retrieved text is context here, never the input.
            outcome = _simulate_tariff(sector, baseline, magnitude, config)
        else:
            outcome = _simulate_fx(sector, baseline, magnitude, config)

        if outcome is None:
            assumptions.append(
                f"The knowledge graph does not record preference coverage for {sector}, so the "
                "effect of losing it cannot be simulated. Reporting this rather than a number."
            )
            evidence.append(
                evidence_from_query(
                    claim=(
                        f"No trade-agreement coverage is recorded for {sector} in the knowledge "
                        "graph, so an agreement-loss simulation would rest on an unchecked "
                        "assumption."
                    ),
                    cypher=refusal_cypher,
                    period=str(baseline_year),
                )
            )
            # A refusal still has to meet the two-evidence floor. The baseline is
            # the half of the picture that *is* known, and stating it is what
            # makes the refusal specific rather than a shrug.
            evidence.append(
                evidence_from_query(
                    claim=(
                        f"{sector.title()} exports of {item} were USD {baseline:,.0f} in "
                        f"{baseline_year}; the baseline is known, only the preference "
                        "coverage needed to shock it is missing."
                    ),
                    cypher=baseline_cypher,
                    period=str(baseline_year),
                )
            )
            # A refusal that can still say what the destination's own policy
            # states is a better refusal. It does not become an answer — no
            # figure is produced and none is implied — but "the graph records no
            # coverage, and here is what the UK strategy says about preferences"
            # tells the reader something, where a bare refusal tells them nothing.
            evidence.extend(_policy_evidence(chunks, retrieval_detail, limit=2))
            if chunks:
                assumptions.append(
                    f"No simulation was run. {len(chunks)} passage(s) of destination-market "
                    "policy are cited for context only; no figure is derived from them."
                )
            continue

        delta, pct, detail = outcome
        figures[f"{sector}_baseline_usd"] = round(baseline, 2)
        figures[f"{sector}_impact_usd"] = round(delta, 2)
        figures[f"{sector}_impact_pct"] = round(pct, 4)

        evidence.append(
            evidence_from_query(
                claim=(
                    f"{sector.title()} export revenue of USD {baseline:,.0f} in {baseline_year} "
                    f"is the baseline the simulation moves by {pct * 100:+.1f}%, "
                    f"USD {delta:+,.0f}."
                ),
                cypher=baseline_cypher,
                period=str(baseline_year),
            )
        )
        # The coverage query is the other half of an agreement-loss claim, and it
        # was being run and then cited only on the refusal path. That left a
        # successful single-sector simulation carrying one evidence entry against
        # the agent contract's floor of two, and — more to the point — the rate
        # that produced the figure appeared in no evidence at all, which is
        # EVALUATION.md §1's grounding class 1 with this agent named in it.
        if shock == "agreement" and refusal_cypher != baseline_cypher:
            evidence.append(
                evidence_from_query(
                    claim=(
                        f"Preference coverage for {sector} ({item}, HS {hs_prefixes[0]}) is "
                        f"recorded in the knowledge graph, and losing it is modelled as an MFN "
                        f"tariff of {(sourced.rate if sourced else _value(config, 'agreement_loss_mfn_tariff', sector, 0.095)) * 100:.1f}%."
                    ),
                    cypher=refusal_cypher,
                    period=str(baseline_year),
                )
            )

        # The sourced rate is a figure this agent computed with, so it gets its
        # own evidence entry naming the sentence it came from. EVALUATION.md §1
        # grounding class 1 is precisely this agent producing figures that appear
        # in no evidence; adding a new number without a new entry would repeat
        # the bug the same run found.
        if sourced is not None:
            figures[f"{sector}_mfn_tariff_pct"] = round(sourced.rate * 100, 2)
            evidence.append(
                evidence_from_policy(
                    claim=sourced.claim,
                    detail=f"{sourced.chunk.citation}; {retrieval_detail}",
                    url=sourced.chunk.url,
                )
            )
        evidence.extend(_policy_evidence(chunks, retrieval_detail, limit=2, exclude=sourced))

        assumptions.append(detail)
        lines.append(
            f"{sector.title()} export revenue would move by roughly USD {delta:+,.0f} "
            f"({pct * 100:+.1f}%) from a {baseline_year} base of USD {baseline:,.0f}."
        )

    if len(sectors) == 2 and all(f"{s}_impact_pct" in figures for s in sectors):
        first, second = sectors
        gap = figures[f"{first}_impact_pct"] - figures[f"{second}_impact_pct"]
        figures["impact_gap_pct"] = round(gap, 4)
        harder, softer = (first, second) if abs(figures[f"{first}_impact_pct"]) > abs(
            figures[f"{second}_impact_pct"]
        ) else (second, first)
        lines.append(
            f"The effect is larger on {harder} than on {softer}, by "
            f"{abs(gap) * 100:.1f} percentage points."
        )

    summary = " ".join(lines) or (
        "The simulation could not be completed: the knowledge graph is missing the "
        "baseline or coverage data it depends on."
    )

    return await finish(
        agent=AGENT,
        state=state,
        deps=deps,
        summary=summary,
        figures=figures,
        evidence=evidence,
        assumptions=assumptions,
    )


# --- the shocks ----------------------------------------------------------


def _simulate_fx(
    sector: str, baseline: float, depreciation: float, config: dict[str, Any]
) -> tuple[float, float, str]:
    """A rupee depreciation makes exports cheaper abroad, so volume rises.

    Two parameters, both from the config table:

    - **pass-through**: how much of the currency move reaches the foreign-currency
      price. Contract-priced apparel passes less through than spot-traded
      agricultural commodities.
    - **export demand elasticity**: how much volume responds to that price change.

    USD revenue change ≈ pass_through × depreciation × (−elasticity − 1).
    The −1 is the price effect: each unit earns fewer dollars, which offsets part
    of the volume gain. Omitting it is the classic error that reports a
    depreciation as pure upside.
    """
    pass_through = _value(config, "fx_pass_through", sector, 0.5)
    elasticity = _value(config, "export_demand_elasticity", sector, -1.0)

    price_change = -pass_through * depreciation  # foreign price falls
    volume_change = elasticity * price_change  # demand rises as price falls
    revenue_change = volume_change + price_change

    detail = (
        f"{sector}: FX pass-through {pass_through:.2f} and export demand elasticity "
        f"{elasticity:.2f} applied linearly to a {depreciation * 100:.1f}% depreciation. "
        f"Volume effect {volume_change * 100:+.1f}%, price effect {price_change * 100:+.1f}%."
    )
    return baseline * revenue_change, revenue_change, detail


def _simulate_tariff(
    sector: str, baseline: float, tariff: float, config: dict[str, Any]
) -> tuple[float, float, str]:
    """An importing country's tariff raises the buyer's price by the incidence share."""
    incidence = _value(config, "tariff_incidence", "default", 0.5)
    elasticity = _value(config, "export_demand_elasticity", sector, -1.0)

    price_change = incidence * tariff
    revenue_change = elasticity * price_change

    detail = (
        f"{sector}: {tariff * 100:.1f}% tariff with exporter incidence {incidence:.2f} and "
        f"demand elasticity {elasticity:.2f}. Buyer price {price_change * 100:+.1f}%."
    )
    return baseline * revenue_change, revenue_change, detail


async def _simulate_agreement_loss(
    deps: AgentDeps,
    sector: str,
    item: str,
    baseline: float,
    config: dict[str, Any],
    sourced: SourcedRate | None = None,
) -> tuple[tuple[float, float, str] | None, str]:
    """Losing a preference re-imposes the MFN tariff.

    Returns `(outcome, cypher)`. `outcome` is None when the graph records no
    preference coverage — the SAD §4.1 refusal path, because without coverage
    there is no honest number to give. The Cypher comes back either way so the
    caller can cite the query that found nothing as the evidence for saying so.

    `sourced` is an MFN rate read out of a retrieved policy document (D10), or
    None. It replaces the configured constant when present, and the assumption
    text says which of the two was used on every run — the constant disappearing
    silently behind a sourced figure would be the same defect as inventing one.
    """
    hs_code = _hs_for_item(item)
    rows, cypher = await deps.kg.run(*q.agreement_coverage(hs_code))
    if not rows:
        return None, cypher

    preferences = [r for r in rows if r["agreement_type"] == "unilateral_preference"]
    if not preferences:
        return None, cypher

    # Deviation D9 cut the WITS tariff pull, so the fallback is a documented
    # constant. D10 supplies a sourced rate where a policy document states one;
    # where none does, the constant still stands and still says so.
    if sourced is not None:
        mfn_tariff = sourced.rate
        rate_basis = (
            f"MFN tariff of {mfn_tariff * 100:.1f}% read from {sourced.chunk.citation} "
            f"(unverified — no human has checked this rate against the official schedule)"
        )
    else:
        mfn_tariff = _value(config, "agreement_loss_mfn_tariff", sector, 0.095)
        rate_basis = (
            f"MFN tariff of {mfn_tariff * 100:.1f}% from config/elasticities.yaml "
            f"(a literature constant, not a queried tariff schedule — deviation D9)"
        )

    incidence = _value(config, "tariff_incidence", "default", 0.5)
    elasticity = _value(config, "export_demand_elasticity", sector, -1.0)

    price_change = incidence * mfn_tariff
    revenue_change = elasticity * price_change

    names = ", ".join(sorted({r["agreement"] for r in preferences}))
    verified = {r.get("agreement_verified", "unverified") for r in preferences}
    detail = (
        f"{sector}: preference coverage resolved from the knowledge graph ({names}, matched on "
        f"HS {preferences[0]['matched_on']}, status {'/'.join(sorted(verified))}). Loss modelled "
        f"as an {rate_basis}, with exporter incidence {incidence:.2f} "
        f"and demand elasticity {elasticity:.2f}."
    )
    return (baseline * revenue_change, revenue_change, detail), cypher


# --- describing a policy, rather than shocking one (D10) ------------------


async def _describe_policy(
    state: AgentState, deps: AgentDeps, intent: Intent
) -> dict[str, Any]:
    """Answer "what does this market's policy say" from the documents. No simulation.

    **Produces no impact figures, deliberately.** Nothing was shocked, so there
    is no impact to report, and inventing one is the failure this branch exists
    to prevent: before it, "What does India's Foreign Trade Policy say about
    imports from Sri Lanka?" fell through to the `fx` default and was answered
    with a 5% rupee depreciation and a figure of USD -8,240,802.

    Where the graph can answer part of the question it still does. A bare "which
    agreement gives cinnamon preferential access" is answered from the graph's
    `agreement_coverage` and a retrieved passage corroborates it — the graph is
    the source of record for coverage there. But `agreement_coverage` returns
    *Sri Lanka's* arrangements (APTA, GSP+, ISFTA), not the named market's
    policy, so when the question names a specific foreign market and the corpus
    holds no document for it, that lookup is withheld rather than allowed to
    stand in as that market's policy — the E09 misattribution (see step 2's
    comment). Retrieval runs first so this decision has the answered/unheld
    split to work from.
    """
    destinations = _destinations(state["query"], intent) or (None,)
    hs_prefixes = tuple(hs_hierarchy(_hs_for_item(intent.item))) if intent.item else ()
    measures = _descriptive_measures(state["query"])

    assumptions: list[str] = [
        "This answers what the cited documents say. Nothing is simulated and no "
        "impact figure is derived.",
        "Policy documents are recorded `unverified`: no human has checked these "
        "passages against the issuing authority's current text.",
    ]
    # A concrete foreign market is named (E09: "China"), as opposed to a bare
    # "which agreement covers tea" with no country.
    names_a_specific_market = destinations != (None,)

    # 1. What the documents say, per destination the question names. Run first
    # so the graph step below knows whether any named market was actually
    # answered from a document.
    #
    # Retrieved separately rather than in one pooled search: the allow-list is
    # built per country (`policy_documents_for`), and pooling would let the
    # country with documents supply passages the other country's half of the
    # question then appears to have been answered from.
    per_destination_limit = 4 if len(destinations) == 1 else 2
    doc_lines: list[str] = []
    doc_evidence: list[Evidence] = []
    answered: list[str] = []
    unheld: list[str] = []

    for partner in destinations:
        context = await _retrieve_policy(
            deps,
            state,
            partner=partner,
            hs_prefixes=hs_prefixes,
            measure_types=measures,
            limit=per_destination_limit,
        )
        where = f"for {partner}" if partner else "for the market named"

        # A named foreign market's policy can only be answered by a document
        # that market issued. `policy_documents_for` also allow-lists any
        # document that merely *covers the item's HS code* (its OR branch,
        # deliberate for the no-country case), so a question about China's tea
        # policy otherwise gets "answered" from Sri Lanka's own National Export
        # Strategy — the E09 misattribution. When a specific market is named,
        # require at least one allow-listed document actually issued by it.
        if partner:
            has_own_document = any(partner in (r.get("iso3") or []) for r in context.documents)
        else:
            has_own_document = True

        if context.chunks and has_own_document:
            answered.append(partner or "the market named")
            sources = sorted({c.title for c in context.chunks})
            scope = f" {where}" if len(destinations) > 1 else ""
            doc_lines.append(
                f"{len(context.chunks)} passage(s) from {', '.join(sources)} address this{scope}."
            )
            doc_evidence.extend(
                _policy_evidence(context.chunks, context.detail, limit=per_destination_limit)
            )
            continue

        # The corpus not holding something is a real answer and has to be said in
        # the prose, not left as an empty evidence list the reader must notice.
        unheld.append(partner or "the market named")
        doc_lines.append(
            f"No policy document {where} in the corpus addresses this, so that part of the "
            "question cannot be answered from the documents CeyNex holds."
            if len(destinations) > 1
            else f"No policy document {where} in the corpus addresses this, so the question "
            "cannot be answered from the documents CeyNex holds."
        )
        reason = (
            f"Policy retrieval returned nothing {where}: {context.detail}."
            if not context.chunks
            else (
                f"Passages came back {where}, but from documents not issued by that market "
                "(allow-listed only because they cover the item's HS code)."
            )
        )
        assumptions.append(
            f"{reason} Reporting the gap rather than answering from the wrong country's document."
        )
        if context.cypher:
            own = sorted(
                r["doc_id"] for r in context.documents if not partner or partner in (r.get("iso3") or [])
            )
            held = ", ".join(own) or "none"
            doc_evidence.append(
                evidence_from_query(
                    claim=(
                        f"The knowledge graph holds no policy document issued {where}: "
                        f"documents attributed to that market are: {held}."
                    ),
                    cypher=context.cypher,
                )
            )

    # 2. What the graph knows about the item's *outbound* preferential access.
    #
    # `agreement_coverage()` returns Sri Lanka's own arrangements (APTA, GSP+,
    # ISFTA) that cover the HS code — it is not a lookup of the named market's
    # policy, and the graph carries no agreement-membership edge to scope it to
    # one. So it is only surfaced when it can actually corroborate rather than
    # substitute: either no specific foreign market was named (the bare "which
    # agreement covers tea" case), or at least one named market was answered
    # from a real document. When a market is named and none is held — E09,
    # "what does China's trade policy say about Sri Lankan tea", with no China
    # document — listing APTA/GSP+/ISFTA here reads as China's policy and is
    # withheld; the answer is the clean gap the loop above already produced.
    graph_lines: list[str] = []
    graph_evidence: list[Evidence] = []
    if intent.item and (not names_a_specific_market or answered):
        rows, cypher = await deps.kg.run(*q.agreement_coverage(_hs_for_item(intent.item)))
        if rows:
            names = ", ".join(sorted({r["agreement"] for r in rows}))
            lead = "Separately, the" if names_a_specific_market else "The"
            caveat = (
                " These are Sri Lanka's own trade arrangements, not "
                f"{', '.join(answered)}'s domestic policy."
                if names_a_specific_market
                else ""
            )
            graph_lines.append(
                f"{lead} knowledge graph records Sri Lanka's {intent.item} exports "
                f"(HS {rows[0]['matched_on']}) as receiving preferential access under {names}.{caveat}"
            )
            graph_evidence.append(
                evidence_from_query(
                    claim=(
                        f"Sri Lanka's {intent.item.replace('_', ' ')} exports under HS "
                        f"{rows[0]['matched_on']} receive preferential access under {names} "
                        "(Sri Lanka's own arrangements, per the knowledge graph)."
                    ),
                    cypher=cypher,
                )
            )

    # The partial `eval/policy_questions.yaml` asks for on P10: answer for the
    # market that is held, and say plainly that the other one is not. Without
    # this the two halves sit side by side and a reader has to work out which
    # country the passages came from.
    if answered and unheld:
        assumptions.append(
            f"This compares {', '.join(answered)} against nothing held for "
            f"{', '.join(unheld)}; the comparison is one-sided and stated as such."
        )

    # Graph-first for a bare coverage question (it *is* the answer there);
    # docs-first when a market is named (the graph line is a corroborating aside).
    ordered_lines = doc_lines + graph_lines if names_a_specific_market else graph_lines + doc_lines
    ordered_evidence = (
        doc_evidence + graph_evidence if names_a_specific_market else graph_evidence + doc_evidence
    )

    return await finish(
        agent=AGENT,
        state=state,
        deps=deps,
        summary=" ".join(ordered_lines),
        figures={},
        evidence=ordered_evidence,
        assumptions=assumptions,
    )


# --- policy retrieval (D10) ----------------------------------------------


@dataclass
class PolicyContext:
    """What retrieval found, plus the graph lookup that scoped it.

    The Cypher comes back so a descriptive answer can cite the query that decided
    which documents were eligible — including when it found none, which is the
    evidence for saying the corpus holds nothing for that country.
    """

    chunks: list[PolicyChunk]
    detail: str
    doc_ids: tuple[str, ...] = ()
    documents: list[dict[str, Any]] = field(default_factory=list)
    cypher: str = ""


async def _retrieve_policy(
    deps: AgentDeps,
    state: AgentState,
    *,
    partner: str | None,
    hs_prefixes: tuple[str, ...],
    measure_types: tuple[str, ...] = SIMULATION_MEASURES,
    limit: int = 5,
) -> PolicyContext:
    """Graph-anchored policy retrieval. Never raises.

    Anchoring happens in two steps, and both matter:

    1. `policy_documents_for()` asks Neo4j which documents could be relevant to
       this country and these goods. That is a Cypher query over the graph, and
       its result is a `doc_id` allow-list.
    2. Qdrant searches only inside that allow-list, further filtered to tariff
       and FTA passages.

    Without step 1 the search is similarity over the whole corpus, and
    trade-policy documents are similar to each other by construction — the top
    hit for a question about the United States is whichever document phrased the
    same idea most fluently, whoever wrote it.

    Every failure path returns `([], reason)`. Retrieval is an enhancement to an
    agent that already works without it; it is never the reason a query fails.
    """
    retriever = deps.extras.get("policy")
    if retriever is None:
        return PolicyContext([], "policy retrieval not configured")

    try:
        rows, doc_cypher = await deps.kg.run(
            *q.policy_documents_for(iso3=partner, hs_code=hs_prefixes[0] if hs_prefixes else None)
        )
    except KnowledgeGraphUnavailableError as exc:
        return PolicyContext([], f"policy document lookup failed: {exc}")

    doc_ids = tuple(row["doc_id"] for row in rows)
    if not doc_ids:
        return PolicyContext(
            [],
            "the knowledge graph holds no indexed policy document for this country",
            cypher=doc_cypher,
            documents=rows,
        )

    filters = RetrievalFilter(
        iso3=(partner,) if partner else (),
        hs_prefixes=hs_prefixes,
        measure_types=measure_types,
        doc_ids=doc_ids,
        extra_notes=[f"documents scoped by Cypher: {' '.join(doc_cypher.split())[:160]}"],
    )
    try:
        chunks, description = await retriever.search(state["query"], filters=filters, limit=limit)
    except Exception as exc:  # noqa: BLE001 - retrieval degrades, it never fails the query
        log.warning("%s: policy retrieval unavailable: %s", AGENT, exc)
        return PolicyContext([], f"policy retrieval unavailable: {exc}", doc_ids, rows, doc_cypher)
    return PolicyContext(chunks, description, doc_ids, rows, doc_cypher)


def _policy_evidence(
    chunks: list[PolicyChunk],
    detail: str,
    *,
    limit: int,
    exclude: SourcedRate | None = None,
) -> list[Evidence]:
    """Cite retrieved passages as context, without deriving anything from them.

    Capped at `limit`. Evidence is read by a person and merged into an answer by
    an LLM, and five near-identical policy passages crowd out the graph evidence
    that carries the actual figures — the KG entries are the ones a marker checks.

    `exclude` drops the chunk a rate was already read from, so the same passage
    is not cited twice under two different claims.
    """
    already_cited = (
        (exclude.chunk.doc_id, exclude.chunk.chunk_index) if exclude is not None else None
    )
    entries: list[Evidence] = []
    for chunk in chunks:
        if already_cited is not None and (chunk.doc_id, chunk.chunk_index) == already_cited:
            continue
        snippet = " ".join(chunk.text.split())[:220]
        entries.append(
            evidence_from_policy(
                claim=f"{chunk.publisher} — {chunk.title}: \"{snippet}\"",
                detail=f"{chunk.citation}; {detail}",
                url=chunk.url,
            )
        )
        if len(entries) >= limit:
            break
    return entries


# --- helpers -------------------------------------------------------------


def _posits_a_change(query: str) -> bool:
    """Does the question ask what *would happen*, rather than what the rules are?"""
    lowered = query.lower()
    return any(marker in lowered for marker in CHANGE_MARKERS)


def _classify_shock(query: str) -> str:
    """`policy` | `agreement` | `tariff` | `fx`.

    The descriptive check runs first and is the only one that can veto the
    others: "what non-tariff measures does the EU apply" contains "tariff" and is
    not a tariff shock, and "which trade agreement covers cinnamon" contains
    "agreement" and simulates nothing. A question is descriptive when it asks
    what the rules *are* and nothing in it posits a change.

    Text only. Whether a tariff question actually carries a rate to simulate with
    is `_simulate`'s call, because that is where the parsed intent lives.
    """
    lowered = query.lower()
    if any(marker in lowered for marker in DESCRIPTIVE_MARKERS) and not _posits_a_change(query):
        return "policy"
    for shock, keywords in SHOCK_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return shock
    return "fx"


def _destination(query: str, intent: Intent) -> str | None:
    """The destination market the question is about. Never Sri Lanka.

    `parse_intent().partner` resolves the *longest* country name in the query,
    and "Sri Lanka" is longer than most. In "What does India's Foreign Trade
    Policy say about imports from Sri Lanka?" that returns LKA, which anchors
    policy retrieval on Sri Lanka's own export strategy and answers a question
    about Indian policy with four confident passages from the wrong country's
    document — measured, before this existed.

    Sri Lanka is always the reporter (`queries.REPORTER_ISO3`); it is never one
    of its own export destinations, so it can be excluded outright rather than
    disambiguated.
    """
    destinations = _destinations(query, intent)
    return destinations[0] if destinations else None


def _destinations(query: str, intent: Intent) -> tuple[str, ...]:
    """Every destination market the question names, in the order they appear.

    A simulation shocks one market and `_destination` above is right for it. A
    *descriptive* question can name two -- "how do Japan's tariffs on Sri Lankan
    tea compare with Germany's?" -- and answering it from one country's documents
    is how the corpus ends up answering about the wrong government. Live
    2026-09-03 that question ran a Japan tariff simulation and said nothing about
    Germany; `eval/policy_questions.yaml` wants the opposite, and wants the Japan
    gap named rather than filled.
    """
    ordered: list[str] = []
    if intent.partner and intent.partner != q.REPORTER_ISO3:
        ordered.append(intent.partner)
    for iso3 in countries_in(query):
        if iso3 != q.REPORTER_ISO3 and iso3 not in ordered:
            ordered.append(iso3)
    return tuple(ordered)


def _descriptive_measures(query: str) -> tuple[str, ...]:
    """Which measure types a descriptive question is about. Empty means all."""
    lowered = query.lower()
    for measure, keywords in DESCRIPTIVE_MEASURES:
        if any(keyword in lowered for keyword in keywords):
            return (measure,)
    return ()


SECTOR_WORDS: dict[str, tuple[str, ...]] = {
    "agriculture": ("agricultur", "farm", "crop", "commodit", "tea", "cinnamon", "rubber", "coconut"),
    "apparel": ("apparel", "garment", "clothing", "textile", "knit", "woven"),
}


def _sectors_for(state: AgentState, item: str | None) -> list[str]:
    """Which sectors to simulate, widest evidence first.

    The router's `sectors` wins when set. Failing that, read the query directly:
    "apparel exports compared to agriculture" names both, and resolving it to the
    single item the keyword parser happened to match first would answer half the
    question without saying so.
    """
    named = [s for s in state.get("sectors", []) if s in ("agriculture", "apparel")]
    if named:
        return named

    lowered = state["query"].lower()
    mentioned = [
        sector
        for sector, words in SECTOR_WORDS.items()
        if any(word in lowered for word in words)
    ]
    if mentioned:
        return mentioned

    if item and item in SECTOR_OF_ITEM:
        return [SECTOR_OF_ITEM[item]]
    return ["agriculture", "apparel"]


def _representative_item(sector: str, requested: str | None) -> str:
    if requested and SECTOR_OF_ITEM.get(requested) == sector:
        return requested
    return "tea" if sector == "agriculture" else "apparel_knit"


def _hs_for_item(item: str) -> str:
    """The HS code an item's trade is recorded under.

    The table moved to `ceynex/retrieval/tagging.py` when the policy pipeline
    started needing it too: the code a chunk is tagged with and the code this
    agent filters on have to be the same value, and two copies of a mapping are
    two copies that can drift.
    """
    return HS_FOR_ITEM.get(item, "61")


async def _baseline_value(deps: AgentDeps, item: str) -> tuple[float | None, int | None, str]:
    """Most recent annual export value for an item, from the graph."""
    latest_rows, _ = await deps.kg.run(*q.latest_observation_year(item))
    year = int(latest_rows[0]["latest_year"]) if latest_rows and latest_rows[0]["latest_year"] else 2023

    rows, cypher = await deps.kg.run(*q.market_share(item, year))
    if not rows:
        return None, year, cypher
    return float(rows[0]["total_export_value_usd"]), year, cypher


def _value(config: dict[str, Any], group: str, key: str, default: float) -> float:
    """Read an elasticity from the config table, tolerating a missing entry."""
    entry = config.get(group, {}).get(key)
    if isinstance(entry, dict) and "value" in entry:
        return float(entry["value"])
    if isinstance(entry, int | float):
        return float(entry)
    return default


def _base_assumptions(config: dict[str, Any], shock: str, magnitude: float) -> list[str]:
    """SRS 3.1.5 requires these to be stated. They must never be empty."""
    model = config.get("model", {})
    assumptions = [
        f"Shock modelled: {shock}, magnitude {magnitude * 100:.1f}%.",
        f"Functional form is {model.get('form', 'linear')} over a "
        f"{model.get('horizon_months', 12)}-month horizon; effects do not compound.",
        "Elasticities are literature ranges recorded in config/elasticities.yaml, "
        "not estimates fitted to Sri Lankan data.",
    ]
    if note := model.get("note"):
        assumptions.append(str(note))
    return assumptions


__all__ = ["AGENT", "trade_economics_node"]
