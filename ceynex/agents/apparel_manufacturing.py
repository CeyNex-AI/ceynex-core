"""Implements SRS 3.1.6, 3.1.4, 3.4.3 — the Apparel & Manufacturing Agent.

SRS 3.1.6 requires answering from the knowledge graph, never a model's own
knowledge — this node's only figures come from `KnowledgeGraphClientProtocol.run`
(the literal Cypher lands in `Evidence.detail`, per that protocol's own
docstring); an LLM, when available, is used only to phrase already-retrieved
figures in prose, never to originate them.

Query scoping: every query is pinned to EDB's "APPAREL" sub-category
(`Product.key in ('apparel:apprel', 'apparel:apparel')` — the two editions'
own wording) rather than the "APPAREL & TEXTILES ... TOTAL" aggregate table,
because that total table already includes the sub-category tables as
components (`ceynex/kg/schema.py`/`data/raw/edb/PROFILE.md`) — summing across
every `Product` node for a partner would double-count against itself. When
the query names the US or UK (the two markets JAAF confidently labels — see
`ceynex/data/connectors/jaaf.py`), a second, independently-sourced JAAF
figure is added as corroborating evidence; JAAF's own scope is the broader
"Total apparel & textiles", not EDB's narrower "Apparel" sub-category, so the
two are never averaged or reconciled against each other here — surfaced
side by side, with that scope difference stated in `assumptions`.

`ceynex/llm/` has no real client implementation yet (empty placeholder), so
`_get_llm_client` returns `None` until one exists — this node treats that
identically to a live provider failure (SRS 3.4.3): figures and evidence are
still returned, `degraded=True`, with a templated (non-LLM) summary.

Confidence derivation (SRS 3.1.4, never hardcoded): a base term that scales
with how many real KG-backed observations support the answer, minus
`ceynex/orchestrator/confidence.py`'s own `staleness_penalty` for how old the
latest observation is. No cross-source disagreement term is computed between
EDB and JAAF — their different category scopes (above) make a "% difference"
between them meaningless, not a real quality signal, so nothing is invented
in its place.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from ceynex.contracts.evidence import Evidence
from ceynex.contracts.forecast import ForecastPoint
from ceynex.contracts.protocols import KnowledgeGraphClientProtocol, LLMReasoningClientProtocol
from ceynex.contracts.state import AgentOutput, AgentState, failed_output
from ceynex.data.crosswalk import known_aliases, market_to_iso3
from ceynex.kg.client import Neo4jClient
from ceynex.models.apparel import MIN_OBSERVATIONS_FOR_FORECAST, NaiveApparelForecastModel
from ceynex.orchestrator.confidence import clamp, staleness_penalty

_FORECAST_HORIZON = 2

_EDB_APPAREL_KEYS = ["apparel:apprel", "apparel:apparel"]
_JAAF_COVERED_ISO3 = {"USA": "us", "GBR": "uk"}  # iso3 -> JAAF's own market label

_PARTNER_QUERY = """
MATCH (:Country {iso3: 'LKA'})-[:REPORTED]->(r:ExportRecord {source_id: 'EDB'})
      -[:TO]->(:Country {iso3: $iso3}),
      (r)-[:OF]->(p:Product)
WHERE p.key IN $product_keys AND r.frequency = 'A'
RETURN r.period_start AS period, r.export_value_usd AS value, p.name AS product_name
ORDER BY period DESC
LIMIT 5
"""

_OVERVIEW_QUERY = """
MATCH (:Country {iso3: 'LKA'})-[:REPORTED]->(r:ExportRecord {source_id: 'EDB'})
      -[:TO]->(partner:Country),
      (r)-[:OF]->(p:Product)
WHERE p.key IN $product_keys AND r.frequency = 'A' AND partner.iso3 <> 'WLD'
WITH max(r.period_start) AS latest_period
MATCH (:Country {iso3: 'LKA'})-[:REPORTED]->(r2:ExportRecord {source_id: 'EDB'})
      -[:TO]->(partner2:Country),
      (r2)-[:OF]->(p2:Product)
WHERE p2.key IN $product_keys AND r2.frequency = 'A' AND partner2.iso3 <> 'WLD'
      AND r2.period_start = latest_period
RETURN partner2.iso3 AS partner, r2.export_value_usd AS value, latest_period AS period
ORDER BY value DESC
LIMIT 5
"""

_JAAF_ANNUAL_QUERY = """
MATCH (:Country {iso3: 'LKA'})-[:REPORTED]->(r:ExportRecord {source_id: 'JAAF'})
      -[:TO]->(:Country {iso3: $iso3})
WHERE r.frequency = 'M'
WITH date.truncate('year', r.period_start) AS year, sum(r.export_value_usd) AS total,
     max(r.period_start) AS latest_month
RETURN year, total, latest_month
ORDER BY year DESC
LIMIT 3
"""


def _get_kg_client() -> KnowledgeGraphClientProtocol:
    return Neo4jClient()


def _get_llm_client() -> LLMReasoningClientProtocol | None:
    """No shared LLM client implementation exists yet — see module docstring."""
    return None


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


def _to_date(value) -> date | None:
    if value is None:
        return None
    s = str(value)[:10]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def _derive_confidence(observation_count: int, latest_period: date | None) -> float:
    if observation_count == 0:
        return 0.0
    base = min(0.90, 0.55 + 0.05 * min(observation_count, 7))
    months_stale = None
    if latest_period is not None:
        today = datetime.now(UTC).date()
        months_stale = (today.year - latest_period.year) * 12 + (today.month - latest_period.month)
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
        [
            {"period": period.year, "value": float(row["value"])}
            for row in edb_rows
            if (period := _to_date(row["period"])) is not None
        ]
    )
    if len(series) < MIN_OBSERVATIONS_FOR_FORECAST:
        return None, None

    model = NaiveApparelForecastModel(iso3).fit(series)
    return model.predict(_FORECAST_HORIZON), model.backtest()


async def _query_partner(kg: KnowledgeGraphClientProtocol, iso3: str, m49: int) -> AgentOutput:
    edb_rows, edb_cypher = await kg.run(
        _PARTNER_QUERY, {"iso3": iso3, "product_keys": _EDB_APPAREL_KEYS}
    )
    evidence: list[Evidence] = []
    figures: dict[str, float] = {}
    periods: list[date] = []

    if edb_rows:
        evidence.append(
            Evidence(
                source_id="EDB",
                claim=(
                    f"Sri Lanka's EDB-reported Apparel sub-category exports to "
                    f"{iso3}, most recent {len(edb_rows)} years."
                ),
                detail=edb_cypher,
                period=str(edb_rows[0]["period"])[:10],
            )
        )
        for row in edb_rows:
            period = _to_date(row["period"])
            if period is None:
                continue
            periods.append(period)
            figures[f"EDB_{period.year}"] = float(row["value"])

    forecast_points, forecast_metrics = _maybe_forecast(iso3, edb_rows)

    jaaf_label = _JAAF_COVERED_ISO3.get(iso3)
    if jaaf_label is not None:
        jaaf_rows, jaaf_cypher = await kg.run(_JAAF_ANNUAL_QUERY, {"iso3": iso3})
        if jaaf_rows:
            evidence.append(
                Evidence(
                    source_id="JAAF",
                    claim=(
                        f"JAAF-reported Total apparel & textile exports to {iso3} "
                        f"(broader category than EDB's Apparel sub-category above)."
                    ),
                    detail=jaaf_cypher,
                    period=str(jaaf_rows[0]["latest_month"])[:10],
                )
            )
            for row in jaaf_rows:
                period = _to_date(row["latest_month"])
                if period is not None:
                    periods.append(period)
                figures[f"JAAF_{period.year if period else 'latest'}"] = float(row["total"])

    if not evidence:
        return failed_output(
            "apparel_manufacturing", f"No EDB Apparel sub-category data found for partner {iso3}."
        )

    confidence = _derive_confidence(len(edb_rows), max(periods) if periods else None)
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
                period=str(edb_rows[0]["period"])[:10],
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
    rows, cypher = await kg.run(_OVERVIEW_QUERY, {"product_keys": _EDB_APPAREL_KEYS})
    if not rows:
        return failed_output(
            "apparel_manufacturing", "No EDB Apparel sub-category data found in the graph."
        )

    period = _to_date(rows[0]["period"])
    evidence = [
        Evidence(
            source_id="EDB",
            claim=(
                f"Sri Lanka's top {len(rows)} apparel-export destination markets "
                f"(Apparel sub-category), latest available year."
            ),
            detail=cypher,
            period=str(rows[0]["period"])[:10] if rows else None,
        )
    ]
    figures = {row["partner"]: float(row["value"]) for row in rows}
    confidence = _derive_confidence(len(rows), period)

    return AgentOutput(
        agent="apparel_manufacturing",
        summary=(
            f"Sri Lanka's top apparel export markets ({period.year if period else 'latest year'}): "
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


async def apparel_manufacturing_node(state: AgentState) -> AgentState:
    """Answer an apparel-export question from the knowledge graph (SRS 3.1.6)."""
    try:
        kg = _get_kg_client()
        partner = _detect_partner(state["query"])
        output = (
            await _query_partner(kg, partner[0], partner[1])
            if partner is not None
            else await _query_overview(kg)
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

    llm = _get_llm_client()
    if llm is None:
        output["degraded"] = True
        return {"agent_outputs": {"apparel_manufacturing": output}}

    try:
        prose = await llm.generate_explanation(
            {"figures": output["figures"], "evidence": output["evidence"]}
        )
        output["summary"] = prose
    except Exception:  # noqa: BLE001 — provider failure degrades, never raises (SRS 3.4.3)
        output["degraded"] = True

    return {"agent_outputs": {"apparel_manufacturing": output}}
