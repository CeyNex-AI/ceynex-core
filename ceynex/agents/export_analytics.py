"""Implements SRS 3.1.6 — cross-sector trends and market concentration.

**This agent answers from the knowledge graph, not from a model.** That is the
architectural claim the SRS makes explicitly, and it is checkable rather than
asserted because every `Evidence.detail` carries the literal Cypher that produced
the figure beside it. Undermining it for convenience — reaching for a fitted
model because a graph query is fiddly — would make the claim false.

Agent node contract (root CLAUDE.md), all five parts:

1. writes exactly one key into `agent_outputs`, keyed by its own `AgentName`;
2. **never raises** — catches, calls `failed_output`, appends to `errors`;
3. attaches at least two `Evidence` entries naming the real source and period;
4. derives confidence through `ceynex/orchestrator/confidence.py`;
5. sets `degraded=True` and returns figures without prose when the LLM is down.
"""

from __future__ import annotations

import logging
from typing import Any

from ceynex.agents.common import (
    AgentDeps,
    evidence_from_query,
    figures_evidence,
    finish,
    parse_intent,
)
from ceynex.contracts import AgentState, Evidence, failed_output
from ceynex.kg import queries as q
from ceynex.kg.client import KnowledgeGraphUnavailableError

log = logging.getLogger(__name__)

AGENT = "export_analytics"


async def export_analytics_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """`AgentState -> partial state`. Returns only the keys it changed."""
    try:
        output = await _analyse(state, deps)
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
    return output


async def _analyse(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    intent = parse_intent(state["query"])
    item = intent.item or "tea"
    year = intent.year or await _latest_year(deps)

    figures: dict[str, float] = {}
    evidence: list[Evidence] = []
    assumptions: list[str] = []

    # --- market share and the leading destination ---
    rows, cypher = await deps.kg.run(*q.market_share(item, year))
    if rows:
        leader = rows[0]
        figures["top_partner_share"] = round(float(leader["share"]), 4)
        figures["top_partner_value_usd"] = round(float(leader["export_value_usd"]), 2)
        figures["total_export_value_usd"] = round(float(leader["total_export_value_usd"]), 2)
        figures["partner_count"] = float(len(rows))
        figures["hhi"] = _herfindahl(rows)

        # Two distinct claims, both traceable to this one query: who leads, and
        # how concentrated the destination mix is. The agent node contract wants
        # at least two Evidence entries, and manufacturing a second from a
        # different source would be worse than reporting both findings honestly.
        evidence.append(
            evidence_from_query(
                claim=(
                    f"{leader['partner']} took {leader['share'] * 100:.1f}% of Sri Lanka's "
                    f"{item.replace('_', ' ')} export value in {year}, "
                    f"USD {leader['export_value_usd']:,.0f} of USD "
                    f"{leader['total_export_value_usd']:,.0f}."
                ),
                cypher=cypher,
                period=str(year),
            )
        )
        evidence.append(
            evidence_from_query(
                claim=(
                    f"{item.replace('_', ' ').title()} reached {len(rows)} destination markets in "
                    f"{year}, with a Herfindahl concentration index of {figures['hhi']:.2f} "
                    f"({_concentration_word(figures['hhi'])})."
                ),
                cypher=cypher,
                period=str(year),
            )
        )
    else:
        assumptions.append(f"No {item} export records for {year} in the knowledge graph.")

    # --- growth ---
    from_year = year - 4
    growth_rows, growth_cypher = await deps.kg.run(*q.cagr(item, intent.partner, from_year, year))
    growth = _cagr(growth_rows, from_year, year)
    if growth is not None:
        figures["cagr"] = round(growth, 4)
        where = f" to {intent.partner}" if intent.partner else ""
        evidence.append(
            evidence_from_query(
                claim=(
                    f"{item.replace('_', ' ').title()} export value{where} moved at "
                    f"{growth * 100:+.1f}% a year compounded between {from_year} and {year}."
                ),
                cypher=growth_cypher,
                period=f"{from_year}-{year}",
            )
        )
    else:
        assumptions.append(
            f"CAGR could not be computed for {item} between {from_year} and {year} — "
            "the starting year has no positive export value."
        )

    # --- fastest-growing partner ---
    fastest = await _fastest_growing_partner(deps, item, from_year, year)
    if fastest:
        partner, rate, cypher_text = fastest
        figures["fastest_growing_partner_cagr"] = round(rate, 4)
        evidence.append(
            evidence_from_query(
                claim=(
                    f"{partner} was the fastest-growing destination for {item.replace('_', ' ')} "
                    f"between {from_year} and {year}, at {rate * 100:+.1f}% a year."
                ),
                cypher=cypher_text,
                period=f"{from_year}-{year}",
            )
        )

    # --- district concentration (agriculture only) ---
    district_rows, district_cypher = await deps.kg.run(*q.district_concentration(item))
    if district_rows:
        top = district_rows[0]
        figures["top_district_share"] = round(float(top["share"] or 0.0), 4)
        evidence.append(
            evidence_from_query(
                claim=(
                    f"{top['district']} accounts for the largest share of {item} production, "
                    f"{float(top['share'] or 0) * 100:.1f}%."
                ),
                cypher=district_cypher,
            )
        )
    elif intent.wants_districts:
        assumptions.append(
            f"No district-level production data for {item} in the knowledge graph — "
            "PRODUCED_IN edges are loaded by the agriculture sector loader."
        )

    if not evidence:
        # Better to say the graph has nothing than to answer from nowhere.
        evidence.append(figures_evidence(f"No knowledge-graph records matched {item} for {year}."))
        assumptions.append("Answer is limited by missing data, not by the question.")

    summary = _summarize(item, year, figures, bool(district_rows))
    return await finish(
        agent=AGENT,
        state=state,
        deps=deps,
        summary=summary,
        figures=figures,
        evidence=evidence,
        assumptions=assumptions,
    )


async def _latest_year(deps: AgentDeps) -> int:
    rows, _ = await deps.kg.run(*q.latest_observation_year())
    if rows and rows[0].get("latest_year"):
        return int(rows[0]["latest_year"])
    return 2023


async def _fastest_growing_partner(
    deps: AgentDeps, item: str, from_year: int, to_year: int
) -> tuple[str, float, str] | None:
    """SRS 3.1.6 — "which importing country has shown the fastest-growing demand"."""
    cypher = """
    MATCH (i)-[e:EXPORTS_TO]->(c:Country)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND e.year IN [$from_year, $to_year]
    WITH c,
         sum(CASE WHEN e.year = $from_year THEN e.value ELSE 0 END) AS start_value,
         sum(CASE WHEN e.year = $to_year   THEN e.value ELSE 0 END) AS end_value
    WHERE start_value > $floor AND end_value > 0
    RETURN c.name AS partner, start_value, end_value
    """
    # A partner starting from near-nothing produces a meaningless four-digit
    # growth rate; USD 1m is the floor for calling a market "fast growing".
    params = {"item": item, "from_year": from_year, "to_year": to_year, "floor": 1_000_000.0}
    rows, text = await deps.kg.run(cypher, params)

    best: tuple[str, float] | None = None
    years = to_year - from_year
    for row in rows:
        start, end = float(row["start_value"]), float(row["end_value"])
        rate = (end / start) ** (1 / years) - 1 if years > 0 and start > 0 else None
        if rate is not None and (best is None or rate > best[1]):
            best = (row["partner"], rate)
    return (best[0], best[1], text) if best else None


def _cagr(rows: list[dict[str, Any]], from_year: int, to_year: int) -> float | None:
    """Compound annual growth rate, or None when it is not defined.

    Undefined when the base year is missing or non-positive. Returning 0.0 there
    would be a claim that nothing changed, which is a different statement from
    "this cannot be computed" (SAD §4.1).
    """
    by_year = {int(row["year"]): float(row["export_value_usd"] or 0.0) for row in rows}
    start, end = by_year.get(from_year), by_year.get(to_year)
    years = to_year - from_year
    if start is None or end is None or start <= 0 or end <= 0 or years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _concentration_word(hhi: float) -> str:
    """Plain-English reading of the index, for a non-economist (SRS 3.2.1)."""
    if hhi >= 0.25:
        return "highly concentrated"
    if hhi >= 0.15:
        return "moderately concentrated"
    return "diversified"


def _herfindahl(rows: list[dict[str, Any]]) -> float:
    """Herfindahl-Hirschman index of destination concentration, 0 to 1.

    Above 0.25 is a concentrated market: a shock in one destination carries
    straight through to national earnings. This is the number behind
    "market concentration" in SRS 3.1.6.
    """
    return round(sum(float(row["share"]) ** 2 for row in rows), 4)


def _summarize(item: str, year: int, figures: dict[str, float], has_districts: bool) -> str:
    label = item.replace("_", " ")
    if not figures:
        return f"The knowledge graph holds no export records for {label} in {year}."

    parts: list[str] = []
    if "total_export_value_usd" in figures:
        parts.append(
            f"Sri Lanka exported USD {figures['total_export_value_usd']:,.0f} of {label} "
            f"in {year} across {int(figures.get('partner_count', 0))} destination markets."
        )
    if "top_partner_share" in figures:
        parts.append(
            f"The largest single market took {figures['top_partner_share'] * 100:.1f}% of that value"
            + (
                f", giving a destination concentration index of {figures['hhi']:.2f}."
                if "hhi" in figures
                else "."
            )
        )
    if "cagr" in figures:
        direction = "grew" if figures["cagr"] > 0 else "contracted"
        parts.append(
            f"Over the preceding four years the value {direction} at "
            f"{abs(figures['cagr']) * 100:.1f}% a year."
        )
    if has_districts and "top_district_share" in figures:
        parts.append(
            f"Production is concentrated in one district at "
            f"{figures['top_district_share'] * 100:.1f}% of output."
        )
    return " ".join(parts)


__all__ = ["AGENT", "export_analytics_node"]
