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
import re
from typing import Any

from ceynex.agents.common import (
    AgentDeps,
    choose_item,
    combine_outputs,
    compared_items,
    evidence_from_query,
    figures_evidence,
    find_region,
    finish,
    item_label,
    parse_intent,
)
from ceynex.contracts import AgentState, Evidence, failed_output
from ceynex.data.crosswalk import region_of
from ceynex.kg import queries as q
from ceynex.kg.client import KnowledgeGraphUnavailableError
from ceynex.orchestrator.router import named_years

log = logging.getLogger(__name__)

AGENT = "export_analytics"
# "after 2020" is the same span to the latest data as "since 2020": X07 ("which
# sector recovered faster after 2020") was answered with 2020's market report.
_SINCE = re.compile(r"\b(since|after)\b", re.IGNORECASE)


async def export_analytics_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """`AgentState -> partial state`. Returns only the keys it changed."""
    try:
        items = compared_items(state["query"])
        output = await (_analyse_each(state, deps, items) if items else _analyse(state, deps))
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
    year = intent.year or await _latest_year(deps, item)
    asked_years = sorted(set(named_years(state["query"])))
    if len(asked_years) == 1 and _SINCE.search(state["query"]):
        # "since 2021" is a span to the latest data, not a question about 2021.
        latest = await _latest_year(deps, item)
        if asked_years[0] < latest:
            year = latest
            asked_years = [asked_years[0], latest]
    asked_years = [y for y in asked_years if y <= year]
    wants_list = _wants_partner_list(state["query"])
    region = find_region(state["query"])

    figures: dict[str, float] = {}
    evidence: list[Evidence] = []
    assumptions: list[str] = []
    partner_names: list[str] = []

    # --- market share and the leading destination ---
    rows, cypher = await deps.kg.run(*q.market_share(item, year))
    # `market_share`'s own share/total are relative to the *global* total --
    # found live 2026-08-27: "top apparel export markets in Asia" ran both
    # this agent and apparel_manufacturing (region-aware since #43); this one
    # still reported the global leader (USA) regardless of "in Asia", and the
    # merge LLM correctly flagged the two as disagreeing, which they only did
    # because this half ignored the region the question actually asked about.
    # Recomputed relative to the region's own total, not the global one --
    # "took 38% of Asian imports" would be wrong if it meant 38% of global.
    if region:
        rows = [row for row in rows if region_of(row["partner_iso3"]) == region]
        region_total = sum(float(row["export_value_usd"]) for row in rows)
        rows = sorted(
            (
                {
                    **row,
                    "total_export_value_usd": region_total,
                    "share": (float(row["export_value_usd"]) / region_total) if region_total else 0.0,
                }
                for row in rows
            ),
            key=lambda r: r["share"],
            reverse=True,
        )
    scope = f" among {region} markets" if region else ""
    if rows:
        leader = rows[0]
        # Unrounded: the summary and the evidence claim each format this once, as
        # "{:.1f}%". Pre-rounding to 4 dp double-rounded it (0.201476 -> 0.2015
        # -> "20.2%") while the evidence, formatted from the raw share, said "20.1%".
        figures["top_partner_share"] = float(leader["share"])
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
                    f"{item.replace('_', ' ')} export value{scope} in {year}, "
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
                    f"{item.replace('_', ' ').title()} reached {len(rows)} destination markets"
                    f"{scope} in {year}, with a Herfindahl concentration index of {figures['hhi']:.2f} "
                    f"({_concentration_word(figures['hhi'])})."
                ),
                cypher=cypher,
                period=str(year),
            )
        )

        # `rows` already carries every partner's name (`market_share`'s own
        # Cypher, ORDER BY share DESC) -- the two claims above only ever read
        # rows[0]. Found live 2026-08-27: "what are the 126 apparel data
        # countries" got the usual leader/concentration report and an honest
        # -sounding but wrong "the data does not specify the names" line, even
        # though every one of the 126 names was sitting in `rows` unread. Only
        # attached when actually asked for -- 126 names is not something every
        # market-share question should carry.
        if wants_list:
            partner_names = [str(row["partner"]) for row in rows]
            evidence.append(
                evidence_from_query(
                    claim=(
                        f"All {len(partner_names)} destination countries for {item.replace('_', ' ')}"
                        f"{scope} in {year}, ranked by export value: {', '.join(partner_names)}."
                    ),
                    cypher=cypher,
                    period=str(year),
                )
            )

        if item in ("apparel_knit", "apparel_woven"):
            # See apparel_manufacturing.py's matching note -- same finding,
            # from the other side. Comtrade's HS-code split and EDB's own
            # combined "Apparel" sub-category are separate sources, loaded
            # under different item keys specifically so they never collide
            # (kg/loaders/apparel.py's docstring); a same-country figure from
            # each is expected to differ, not a discrepancy to reconcile.
            assumptions.append(
                f"{item.replace('_', ' ').title()} here is Comtrade's HS-code-based category, a "
                "separate source and boundary from EDB's own combined 'Apparel' sub-category "
                "reported elsewhere -- the two are not reconciled against each other, so they "
                "will not match."
            )
    elif region:
        assumptions.append(f"No {region} destination has recorded {item} exports for {year}.")
    else:
        assumptions.append(f"No {item} export records for {year} in the knowledge graph.")

    # --- the years the question named ---
    # Found live 2026-10-02: "Compare tea export value in 2023 and 2025" got the
    # 2025 market report and a four-year CAGR, and the answer called 2023 "not
    # available" although the graph holds it. Each named year is read, and the
    # first-to-last change is computed here so no one has to do the arithmetic
    # in prose.
    if len(asked_years) >= 2 and not region:
        label = item.replace("_", " ")
        where = f" to {intent.partner}" if intent.partner else ""
        value_rows, value_cypher = await deps.kg.run(
            *q.export_value_by_year(item, intent.partner, asked_years)
        )
        values = {
            int(row["year"]): float(row["export_value_usd"])
            for row in value_rows
            if row.get("export_value_usd")
        }
        for asked in asked_years:
            if asked in values:
                figures[f"export_value_usd_{asked}"] = round(values[asked], 2)
                evidence.append(
                    evidence_from_query(
                        claim=f"Sri Lanka's {label} export value{where} was USD {values[asked]:,.0f} in {asked}.",
                        cypher=value_cypher,
                        period=str(asked),
                    )
                )
            else:
                assumptions.append(f"No {label} export value{where} is recorded for {asked}.")
        first, last = asked_years[0], asked_years[-1]
        if values.get(first, 0.0) > 0 and last in values:
            change = values[last] - values[first]
            figures["value_change_usd"] = round(change, 2)
            figures["value_change_pct"] = change / values[first]  # unrounded, see top_partner_share
            evidence.append(
                evidence_from_query(
                    claim=(
                        f"Sri Lanka's {label} export value{where} changed by "
                        f"{change / values[first] * 100:+.1f}% (USD {change:+,.0f}) "
                        f"between {first} and {last}."
                    ),
                    cypher=value_cypher,
                    period=f"{first}-{last}",
                )
            )

    # --- growth ---
    # Over the span the question named, when it named one; otherwise the four
    # years before `year`.
    from_year = asked_years[0] if len(asked_years) >= 2 else year - 4
    growth_rows, growth_cypher = await deps.kg.run(*q.cagr(item, intent.partner, from_year, year))
    growth = _cagr(growth_rows, from_year, year)
    if growth is not None:
        figures["cagr"] = growth  # unrounded, see top_partner_share
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
        figures["fastest_growing_partner_cagr"] = rate  # unrounded, see top_partner_share
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
        figures["top_district_share"] = float(top["share"] or 0.0)  # unrounded, see top_partner_share
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

    summary = _summarize(
        item, year, figures, bool(district_rows), partner_names, region, from_year=from_year
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


#: Figures two items can be compared on, with how to say which is higher.
COMPARABLE = (
    ("hhi", "destination concentration index", "{:.2f}"),
    ("top_partner_share", "share taken by its largest market", "{:.1%}"),
    ("cagr", "compound annual growth", "{:+.1%}"),
    ("value_change_pct", "change in export value over the years asked about", "{:+.1%}"),
    ("total_export_value_usd", "export value", "USD {:,.0f}"),
)


async def _analyse_each(state: AgentState, deps: AgentDeps, items: list[str]) -> dict[str, Any]:
    """A comparison question: the full analysis once per item, then the comparison.

    The comparison sentences are computed here from the figures, so the merge is
    handed the verdict's inputs side by side and never has to infer one side.
    """
    outputs = {}
    for item in items:
        patch = await _analyse(choose_item(state, item), deps)
        outputs[item] = patch["agent_outputs"][AGENT]
    comparison = []
    for key, what, fmt in COMPARABLE:
        values = {i: o["figures"][key] for i, o in outputs.items() if key in o["figures"]}
        if len(values) >= 2:
            ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
            listed = ", ".join(f"{item_label(i)} {fmt.format(v)}" for i, v in ranked)
            comparison.append(f"Ranked by {what}: {listed}.")
    missing = [item_label(i) for i, o in outputs.items() if not o["figures"]]
    if missing:
        comparison.append(f"No comparable figures were found for {', '.join(missing)}.")
    return combine_outputs(AGENT, outputs, comparison)


async def _latest_year(deps: AgentDeps, item: str) -> int:
    rows, _ = await deps.kg.run(*q.latest_observation_year(item))
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


def _wants_partner_list(query: str) -> bool:
    """"Which countries"/"what countries" wants every name; "which country" (no
    "s") is the existing singular ranking phrasing ("which country is
    fastest-growing") and must keep working exactly as before -- this only
    fires on the plural.
    """
    lowered = query.lower()
    return "countries" in lowered and any(
        marker in lowered for marker in ("list", "which", "what", "name")
    )


def _summarize(
    item: str,
    year: int,
    figures: dict[str, float],
    has_districts: bool,
    partner_names: list[str],
    region: str | None = None,
    *,
    from_year: int | None = None,
) -> str:
    label = item.replace("_", " ")
    if not figures:
        no_records = f"The knowledge graph holds no export records for {label} in {year}"
        return f"{no_records} to {region} destinations." if region else f"{no_records}."

    parts: list[str] = []
    if "total_export_value_usd" in figures:
        scope = f" to {region} destinations" if region else ""
        parts.append(
            f"Sri Lanka exported USD {figures['total_export_value_usd']:,.0f} of {label}{scope} "
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
    if "value_change_pct" in figures:
        years = sorted(int(k.rsplit("_", 1)[1]) for k in figures if k.startswith("export_value_usd_"))
        first, last = years[0], years[-1]
        parts.append(
            f"Export value went from USD {figures[f'export_value_usd_{first}']:,.0f} in {first} to "
            f"USD {figures[f'export_value_usd_{last}']:,.0f} in {last}, a change of "
            f"{figures['value_change_pct'] * 100:+.1f}%."
        )
    if "cagr" in figures:
        direction = "grew" if figures["cagr"] > 0 else "contracted"
        span = (
            f"Between {from_year} and {year}"
            if from_year is not None and from_year != year - 4
            else "Over the preceding four years"
        )
        parts.append(f"{span} the value {direction} at {abs(figures['cagr']) * 100:.1f}% a year.")
    if has_districts and "top_district_share" in figures:
        parts.append(
            f"Production is concentrated in one district at "
            f"{figures['top_district_share'] * 100:.1f}% of output."
        )
    if partner_names:
        parts.append(
            f"All {len(partner_names)} destination countries, ranked by export value, are "
            "listed in the evidence below."
        )
    return " ".join(parts)


__all__ = ["AGENT", "export_analytics_node"]
