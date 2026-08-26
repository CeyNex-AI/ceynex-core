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
from typing import Any

from ceynex.agents.common import (
    AgentDeps,
    evidence_from_query,
    finish,
    parse_intent,
)
from ceynex.contracts import AgentState, Evidence, failed_output
from ceynex.kg import queries as q
from ceynex.kg.client import KnowledgeGraphUnavailableError
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
    ("agreement", ("gsp", "gsp+", "agreement", "fta", "preference", "duty-free", "duty free")),
    ("tariff", ("tariff", "duty", "customs", "import tax")),
    ("fx", ("deprecia", "apprecia", "exchange rate", "rupee", "lkr", "currency", "devalu")),
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

    # Which sectors the question touches. Both, unless it named one.
    sectors = _sectors_for(state, intent.item)

    figures: dict[str, float] = {}
    evidence: list[Evidence] = []
    assumptions = _base_assumptions(config, shock, magnitude)
    lines: list[str] = []

    for sector in sectors:
        item = _representative_item(sector, intent.item)
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
                deps, sector, item, baseline, config
            )
        elif shock == "tariff":
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
) -> tuple[tuple[float, float, str] | None, str]:
    """Losing a preference re-imposes the MFN tariff.

    Returns `(outcome, cypher)`. `outcome` is None when the graph records no
    preference coverage — the SAD §4.1 refusal path, because without coverage
    there is no honest number to give. The Cypher comes back either way so the
    caller can cite the query that found nothing as the evidence for saying so.
    """
    hs_code = _hs_for_item(item)
    rows, cypher = await deps.kg.run(*q.agreement_coverage(hs_code))
    if not rows:
        return None, cypher

    preferences = [r for r in rows if r["agreement_type"] == "unilateral_preference"]
    if not preferences:
        return None, cypher

    # Without a WITS tariff pull this is the documented fallback (the plan's
    # "static GSP+ table" cut). Stated as an assumption, not hidden as a constant.
    mfn_tariff = _value(config, "agreement_loss_mfn_tariff", sector, 0.095)
    incidence = _value(config, "tariff_incidence", "default", 0.5)
    elasticity = _value(config, "export_demand_elasticity", sector, -1.0)

    price_change = incidence * mfn_tariff
    revenue_change = elasticity * price_change

    names = ", ".join(sorted({r["agreement"] for r in preferences}))
    verified = {r.get("agreement_verified", "unverified") for r in preferences}
    detail = (
        f"{sector}: preference coverage resolved from the knowledge graph ({names}, matched on "
        f"HS {preferences[0]['matched_on']}, status {'/'.join(sorted(verified))}). Loss modelled "
        f"as an MFN tariff of {mfn_tariff * 100:.1f}% with exporter incidence {incidence:.2f} "
        f"and demand elasticity {elasticity:.2f}."
    )
    return (baseline * revenue_change, revenue_change, detail), cypher


# --- helpers -------------------------------------------------------------


def _classify_shock(query: str) -> str:
    lowered = query.lower()
    for shock, keywords in SHOCK_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return shock
    return "fx"


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
    return {
        "tea": "0902",
        "cinnamon": "0906",
        "rubber": "4001",
        "coconut": "1513",
        "apparel_knit": "61",
        "apparel_woven": "62",
    }.get(item, "61")


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
