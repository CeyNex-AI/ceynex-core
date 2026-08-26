"""Implements SRS 3.1.6, 3.1.4, 3.4.3 — the Apparel & Manufacturing Agent.

SRS 3.1.6 requires answering from the knowledge graph, never a model's own
knowledge — this node's only figures come from `KnowledgeGraphClientProtocol.run`
(the literal Cypher lands in `Evidence.detail`, per that protocol's own
docstring); an LLM, when available, is used only to phrase already-retrieved
figures in prose, never to originate them.

Query scoping: every query is pinned to the `apparel_edb` graph item — EDB's
"Apparel" sub-category, normalized in `ceynex/kg/loaders/apparel.py` from the
two spellings ('APPAREL'/'APPREL') the source PDFs actually use — rather than
the "Apparel & Textiles ... Total" aggregate table, because that total table
already includes the sub-category tables as components
(`data/raw/edb/PROFILE.md`) — summing across every `ApparelCategory` node for
a partner would double-count against itself. When
the query names the US or UK (the two markets JAAF confidently labels — see
`ceynex/data/connectors/jaaf.py`), a second, independently-sourced JAAF
figure is added as corroborating evidence; JAAF's own scope is the broader
"Total apparel & textiles", not EDB's narrower "Apparel" sub-category, so the
two are never averaged or reconciled against each other here — surfaced
side by side, with that scope difference stated in `assumptions`.

`deps.llm.generate_explanation` (`ceynex/llm/client.py`) returns `""` rather
than raising when the provider is unreachable, out of budget, or unconfigured
— this node treats an empty return identically to that live provider failure
(SRS 3.4.3): figures and evidence are still returned, `degraded=True`, with
the templated (non-LLM) summary standing in for prose.

Confidence derivation (SRS 3.1.4, never hardcoded): a base term that scales
with how many real KG-backed observations support the answer, minus
`ceynex/orchestrator/confidence.py`'s own `staleness_penalty` for how old the
latest observation is. No cross-source disagreement term is computed between
EDB and JAAF — their different category scopes (above) make a "% difference"
between them meaningless, not a real quality signal, so nothing is invented
in its place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ceynex.agents.common import AgentDeps
from ceynex.contracts.evidence import Evidence
from ceynex.contracts.forecast import ForecastPoint
from ceynex.contracts.protocols import KnowledgeGraphClientProtocol
from ceynex.contracts.state import AgentOutput, AgentState, failed_output
from ceynex.data.crosswalk import known_aliases, market_to_iso3
from ceynex.models.apparel import MIN_OBSERVATIONS_FOR_FORECAST, NaiveApparelForecastModel
from ceynex.orchestrator.confidence import clamp, staleness_penalty

_FORECAST_HORIZON = 2

# Graph item names — see ceynex/kg/loaders/apparel.py's module docstring for
# why EDB's two source spellings ('APPAREL'/'APPREL') both normalize to one.
_EDB_ITEM = "apparel_edb"
_JAAF_ITEM = "apparel_textiles"
_JAAF_COVERED_ISO3 = {"USA": "us", "GBR": "uk"}  # iso3 -> JAAF's own market label

_PARTNER_QUERY = """
MATCH (i:ApparelCategory {name: $item})-[e:EXPORTS_TO]->(c:Country {iso3: $iso3})
RETURN e.year AS year, e.value AS value
ORDER BY year DESC
LIMIT 5
"""

_OVERVIEW_QUERY = """
MATCH (i:ApparelCategory {name: $item})-[e:EXPORTS_TO]->(:Country)
WITH max(e.year) AS latest_year
MATCH (i2:ApparelCategory {name: $item})-[e2:EXPORTS_TO]->(c2:Country)
WHERE e2.year = latest_year
RETURN c2.iso3 AS partner, e2.value AS value, latest_year AS year
ORDER BY value DESC
LIMIT 5
"""

_JAAF_ANNUAL_QUERY = """
MATCH (i:ApparelCategory {name: $item})-[e:EXPORTS_TO]->(c:Country {iso3: $iso3})
RETURN e.year AS year, e.value AS total
ORDER BY year DESC
LIMIT 3
"""


def _detect_partner(query: str) -> tuple[str, int] | None:
    """Best-effort word-boundary scan of the query text for a known country name.

    Not NL->Cypher translation — that's the router/orchestrator's job (SRS
    3.6.4), not a single sector agent's. Longest aliases are checked first so
    e.g. "united states" wins over a shorter alias that happens to be a
    substring of it.
    """
    import re

    q = query.lower()
    for alias in sorted(known_aliases(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            iso3, m49 = market_to_iso3(alias)
            if iso3:
                return iso3, m49
    return None


def _derive_confidence(observation_count: int, latest_year: int | None) -> float:
    if observation_count == 0:
        return 0.0
    base = min(0.90, 0.55 + 0.05 * min(observation_count, 7))
    months_stale = None
    if latest_year is not None:
        today = datetime.now(UTC).date()
        months_stale = (today.year - latest_year) * 12 + (today.month - 1)
    return clamp(base - staleness_penalty(months_stale))


def _maybe_forecast(
    iso3: str, edb_rows: list[dict]
) -> tuple[list[ForecastPoint] | None, dict[str, float] | None]:
    """Naive forecast + its own backtest, only when there's enough EDB history
    to honestly evaluate it (see ceynex/models/apparel.py's module docstring
    for why a naive baseline, not a fitted model, is the right choice here).
    Returns (None, None) below that floor rather than a forecast nobody checked.
    """
    if len(edb_rows) < MIN_OBSERVATIONS_FOR_FORECAST:
        return None, None

    series = pd.DataFrame(
        [{"period": int(row["year"]), "value": float(row["value"])} for row in edb_rows]
    )
    if len(series) < MIN_OBSERVATIONS_FOR_FORECAST:
        return None, None

    model = NaiveApparelForecastModel(iso3).fit(series)
    return model.predict(_FORECAST_HORIZON), model.backtest()


async def _query_partner(kg: KnowledgeGraphClientProtocol, iso3: str, m49: int) -> AgentOutput:
    edb_rows, edb_cypher = await kg.run(_PARTNER_QUERY, {"iso3": iso3, "item": _EDB_ITEM})
    evidence: list[Evidence] = []
    figures: dict[str, float] = {}
    years: list[int] = []

    if edb_rows:
        evidence.append(
            Evidence(
                source_id="EDB",
                claim=(
                    f"Sri Lanka's EDB-reported Apparel sub-category exports to "
                    f"{iso3}, most recent {len(edb_rows)} years."
                ),
                detail=edb_cypher,
                period=f"{edb_rows[0]['year']}-01-01",
            )
        )
        for row in edb_rows:
            year = int(row["year"])
            years.append(year)
            figures[f"EDB_{year}"] = float(row["value"])

    forecast_points, forecast_metrics = _maybe_forecast(iso3, edb_rows)

    jaaf_label = _JAAF_COVERED_ISO3.get(iso3)
    if jaaf_label is not None:
        jaaf_rows, jaaf_cypher = await kg.run(_JAAF_ANNUAL_QUERY, {"iso3": iso3, "item": _JAAF_ITEM})
        if jaaf_rows:
            evidence.append(
                Evidence(
                    source_id="JAAF",
                    claim=(
                        f"JAAF-reported Total apparel & textile exports to {iso3} "
                        f"(broader category than EDB's Apparel sub-category above)."
                    ),
                    detail=jaaf_cypher,
                    period=f"{jaaf_rows[0]['year']}-01-01",
                )
            )
            for row in jaaf_rows:
                year = int(row["year"])
                years.append(year)
                figures[f"JAAF_{year}"] = float(row["total"])

    if not evidence:
        return failed_output(
            "apparel_manufacturing", f"No EDB Apparel sub-category data found for partner {iso3}."
        )

    confidence = _derive_confidence(len(edb_rows), max(years) if years else None)
    assumptions = [
        "Figures are Sri Lanka's Apparel sub-category exports only (EDB table "
        "APPREL/APPAREL), not the broader Apparel & Textiles total, to avoid "
        "double-counting against EDB's own aggregate table.",
    ]
    if jaaf_label is not None:
        assumptions.append(
            "JAAF figures cover a broader category (Total apparel & textiles) "
            "than the EDB Apparel figures above — the two are shown side by "
            "side, not reconciled or averaged."
        )

    output = AgentOutput(
        agent="apparel_manufacturing",
        summary=(
            f"Sri Lanka's apparel exports to {iso3}: {len(edb_rows)} years of "
            f"EDB data{' plus JAAF corroboration' if jaaf_label else ''}. "
            "See figures for values by year."
        ),
        figures=figures,
        assumptions=assumptions,
        evidence=evidence,
        confidence=confidence,
        degraded=False,
    )

    if forecast_points is not None:
        output["forecast"] = forecast_points
        evidence.append(
            Evidence(
                source_id="MODEL",
                claim=(
                    f"Naive (last-value) baseline forecast for {iso3} apparel exports, "
                    f"backtested on {int(forecast_metrics['n_obs'])} years "
                    f"({int(forecast_metrics['folds'])} rolling-origin folds): "
                    f"MAPE {forecast_metrics['mape']:.1%}, "
                    f"80% interval coverage {forecast_metrics['coverage']:.0%}."
                ),
                detail=(
                    "NaiveApparelForecastModel (ceynex/models/apparel.py): last observed "
                    "value carried forward, 80% interval from a bootstrap of one-step "
                    "residuals. No published Sri Lankan apparel-export forecasting "
                    "benchmark exists to compare against (.claude/commands/backtest.md)."
                ),
                period=f"{edb_rows[0]['year']}-01-01",
            )
        )
        assumptions.append(
            f"The forecast is a naive (flat) baseline, not a fitted model — the "
            f"underlying series is only {len(edb_rows)} annual points, well under "
            "the ~40-observation floor for anything more (risk register R3). "
            "Its own backtest MAPE and interval coverage are reported alongside it "
            "rather than a bare number, so the forecast's reliability is checkable, "
            "not just asserted."
        )

    return output


async def _query_overview(kg: KnowledgeGraphClientProtocol) -> AgentOutput:
    rows, cypher = await kg.run(_OVERVIEW_QUERY, {"item": _EDB_ITEM})
    if not rows:
        return failed_output(
            "apparel_manufacturing", "No EDB Apparel sub-category data found in the graph."
        )

    year = int(rows[0]["year"])
    evidence = [
        Evidence(
            source_id="EDB",
            claim=(
                f"Sri Lanka's top {len(rows)} apparel-export destination markets "
                f"(Apparel sub-category), latest available year."
            ),
            detail=cypher,
            period=f"{year}-01-01",
        )
    ]
    figures = {row["partner"]: float(row["value"]) for row in rows}
    confidence = _derive_confidence(len(rows), year)

    return AgentOutput(
        agent="apparel_manufacturing",
        summary=(
            f"Sri Lanka's top apparel export markets ({year}): "
            + ", ".join(f"{row['partner']}" for row in rows)
            + "."
        ),
        figures=figures,
        assumptions=[
            "Figures are Sri Lanka's Apparel sub-category exports only, not the "
            "broader Apparel & Textiles total, to avoid double-counting against "
            "EDB's own aggregate table.",
        ],
        evidence=evidence,
        confidence=confidence,
        degraded=False,
    )


async def apparel_manufacturing_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Answer an apparel-export question from the knowledge graph (SRS 3.1.6)."""
    try:
        partner = _detect_partner(state["query"])
        output = (
            await _query_partner(deps.kg, partner[0], partner[1])
            if partner is not None
            else await _query_overview(deps.kg)
        )
    except Exception as exc:  # noqa: BLE001 — contract requires never raising
        return {
            "agent_outputs": {
                "apparel_manufacturing": failed_output("apparel_manufacturing", str(exc))
            },
            "errors": [str(exc)],
        }

    if output.get("error") or not output["evidence"]:
        return {"agent_outputs": {"apparel_manufacturing": output}, "errors": [output.get("error", "")]}

    # `generate_explanation` never raises — "" is itself the degraded signal
    # (see `LLMReasoningClientProtocol`/`ceynex/llm/client.py`).
    prose = await deps.llm.generate_explanation(
        {"figures": output["figures"], "evidence": output["evidence"]}
    )
    if prose:
        output["summary"] = prose
    else:
        output["degraded"] = True

    return {"agent_outputs": {"apparel_manufacturing": output}}
