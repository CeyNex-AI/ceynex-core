"""Implements SRS 3.1.4, 3.1.6, 3.1.9, and 3.4.3 — agriculture answers.

This node distinguishes three kinds of evidence rather than treating every
agriculture question as an export-value query: annual price and export-volume
series come from the unified dataset, producing districts come from Neo4j, and
forward-looking tea-volume/cinnamon-price answers come from a target-matched
registered model.  It intentionally refuses production and substitution claims
when their required measurement or relationship is not represented faithfully.

Confidence is derived from the number and recency of supporting observations;
model forecasts also carry their rolling-origin MAPE.  The formula is local but
uses the project's shared staleness penalty, so it never disguises a fixed score
as a measurement.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ceynex.agents.common import (
    AgentDeps,
    evidence_from_dataset,
    evidence_from_model,
    evidence_from_query,
    finish,
    parse_intent,
)
from ceynex.contracts import AgentOutput, AgentState, Evidence, ForecastPoint, failed_output
from ceynex.data.reader import DatasetUnavailableError, annual_series, relevant_dq_flags
from ceynex.kg import queries as q
from ceynex.kg.client import KnowledgeGraphUnavailableError
from ceynex.orchestrator.confidence import clamp, staleness_penalty

log = logging.getLogger(__name__)

AGENT = "agriculture_commodity"
EXPORT_VALUE_TARGET = "export_value_usd"

SERIES = {
    "price": {
        "item": "cinnamon",
        "target": "price",
        "unit": "USD/kg",
        "source_id": "FAOSTAT",
        "source": "FAOSTAT annual Sri Lanka cinnamon producer-price series",
        "label": "cinnamon producer price",
    },
    "volume": {
        "item": "tea",
        "target": "export_volume",
        "unit": "kg",
        "source_id": "TEA_BOARD",
        "source": "Sri Lanka Tea Board annual total export-volume series",
        "label": "tea export volume",
    },
}


async def agriculture_commodity_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    """Return one contract-conformant, evidence-backed agriculture output."""
    try:
        return await _answer(state, deps)
    except KnowledgeGraphUnavailableError as exc:
        log.warning("%s: knowledge graph unavailable: %s", AGENT, exc)
        return _failed(f"knowledge graph unavailable: {exc}")
    except DatasetUnavailableError as exc:
        log.warning("%s: unified dataset unavailable: %s", AGENT, exc)
        return _failed(f"unified dataset unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001 - preserve orchestrator partial results
        log.exception("%s failed", AGENT)
        return _failed(str(exc))


def _failed(reason: str) -> dict[str, Any]:
    return {
        "agent_outputs": {AGENT: failed_output(AGENT, reason)},
        "errors": [f"{AGENT}: {reason}"],
        "degraded": True,
    }


async def _answer(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    intent = parse_intent(state["query"])
    kind = _question_kind(state["query"], intent)

    if kind == "substitution":
        return await _unsupported_substitution(state, deps)
    if kind == "district":
        return await _district_answer(state, deps, intent.item or "cinnamon")
    if kind == "forecast":
        return await _model_forecast(state, deps, intent.item, intent.forecast_target, intent)
    if kind == "deferred_forecast":
        # wants_forecast=True but forecast_target is None -- by parse_intent's
        # own docstring, that's the "bare 'forecast cinnamon exports'" case,
        # deliberately left for the separate forecast agent's USD export-value
        # path. Answering here too would mean duplicating (or worse,
        # mismatching) that agent's output -- e.g. falling through to the
        # "volume" trend below, which is hardcoded to tea regardless of the
        # item actually asked about, previously surfaced as tea evidence
        # under an unrelated cinnamon forecast question.
        return await _unsupported_target(
            state, deps, "This forecast is served by the export-value forecast agent; no M1 model target was requested."
        )
    if kind == "production":
        return await _unsupported_production(state, deps, intent.item or "tea")
    return await _trend_answer(state, deps, "price" if kind == "price" else "volume", intent.item)


def _question_kind(query: str, intent: Any) -> str:
    lowered = query.lower()
    if "demand" in lowered and ("price" in lowered or "substitut" in lowered):
        return "substitution"
    if intent.wants_districts:
        return "district"
    if intent.wants_forecast:
        if intent.forecast_target in {"export_volume", "producer_price"}:
            return "forecast"
        return "deferred_forecast"
    if "production" in lowered:
        return "production"
    if "price" in lowered:
        return "price"
    return "volume"


async def _trend_answer(
    state: AgentState, deps: AgentDeps, kind: str, requested_item: str | None
) -> dict[str, Any]:
    info = SERIES[kind]
    # Guard rail: SERIES has exactly one hardcoded item per kind (tea for
    # volume, cinnamon for price) -- that's the only combination this file
    # actually has a sourced series for. A query naming a *different* item
    # (rubber, coconut, or the other kind's item) must not silently answer
    # with that hardcoded item's data instead -- this is exactly the bug
    # found live 2026-08-26 (a cinnamon forecast question answered with tea
    # export-volume evidence). requested_item is None for a genuine no-item
    # question ("how are export volumes trending"), where defaulting to the
    # one sourced item for this kind is the existing, intended behaviour.
    if requested_item is not None and requested_item != info["item"]:
        reason = (
            f"No sourced {info['target'].replace('_', ' ')} series is available for {requested_item} -- "
            f"only {info['item']} is covered for this question shape."
        )
        return await _unsupported_target(state, deps, reason)
    frame = annual_series(
        str(info["item"]),
        sector="agriculture",
        target=str(info["target"]),
        dsn=deps.dsn,
    )
    frame = _clean_series(frame)
    if len(frame) < 2:
        return await _respond(
            state,
            deps,
            summary=f"There are only {len(frame)} usable annual observations for {info['label']}; a trend cannot be stated responsibly.",
            figures={},
            evidence=[
                evidence_from_dataset(
                    claim=f"{info['source']} has only {len(frame)} usable annual observations in the unified dataset.",
                    detail=_series_detail(info),
                    source_id=str(info["source_id"]),
                ),
                evidence_from_dataset(
                    claim="At least two observations are required to determine a direction of change.",
                    detail=_series_detail(info),
                    source_id=str(info["source_id"]),
                ),
            ],
            assumptions=["No trend was extrapolated from fewer than two observations."],
            confidence=0.20,
        )

    first, latest = frame.iloc[0], frame.iloc[-1]
    change = float(latest.value) - float(first.value)
    pct_change = change / float(first.value) if float(first.value) else None
    direction = _direction(change)
    figures = {
        f"latest_{info['target']}": round(float(latest.value), 2),
        f"change_{info['target']}": round(change, 2),
    }
    if pct_change is not None:
        figures[f"pct_change_{info['target']}"] = round(pct_change, 4)
    period = f"{int(first.period)}-{int(latest.period)}"
    pct_text = f" ({pct_change * 100:+.1f}%)" if pct_change is not None else ""
    evidence = [
        evidence_from_dataset(
            claim=(
                f"{info['source']} records {len(frame)} annual observations of {info['label']} "
                f"from {int(first.period)} to {int(latest.period)}."
            ),
            detail=_series_detail(info),
            source_id=str(info["source_id"]),
            period=period,
        ),
        evidence_from_dataset(
            claim=(
                f"{info['label'].title()} changed from {float(first.value):,.2f} to "
                f"{float(latest.value):,.2f} {info['unit']} between {int(first.period)} and "
                f"{int(latest.period)}: {direction}{pct_text}."
            ),
            detail=_series_detail(info),
            source_id=str(info["source_id"]),
            period=period,
        ),
    ]
    evidence, assumptions, confidence = _with_dq_flags(
        relevant_dq_flags(
            str(info["item"]),
            str(info["target"]),
            period_start=int(first.period),
            period_end=int(latest.period),
            dsn=deps.dsn,
        ),
        evidence,
        ["Trend compares the first and latest available annual source observations; no missing years are interpolated."],
        _series_confidence(len(frame), int(latest.period)),
    )
    return await _respond(
        state,
        deps,
        summary=(
            f"{info['label'].title()} is {direction}: {float(latest.value):,.2f} {info['unit']} "
            f"in {int(latest.period)}, versus {float(first.value):,.2f} {info['unit']} in "
            f"{int(first.period)}{pct_text}."
        ),
        figures=figures,
        evidence=evidence,
        assumptions=assumptions,
        confidence=confidence,
    )


async def _district_answer(state: AgentState, deps: AgentDeps, item: str) -> dict[str, Any]:
    rows, cypher = await deps.kg.run(*q.district_concentration(item))
    districts = [str(row["district"]) for row in rows if row.get("district")]
    shares = [row for row in rows if isinstance(row.get("share"), int | float)]
    if shares:
        top = max(shares, key=lambda row: float(row["share"]))
        district = str(top["district"])
        share = float(top["share"])
        return await _respond(
            state,
            deps,
            summary=f"{district} has the largest recorded {item} production share at {share * 100:.1f}%.",
            figures={"largest_district_share": round(share, 4)},
            evidence=[
                evidence_from_query(
                    claim=f"The knowledge graph records {district} with the largest {item} production share, {share * 100:.1f}%.",
                    cypher=cypher,
                    period="undated source relationship",
                ),
                evidence_from_query(
                    claim=f"The graph returned {len(districts)} producing districts for {item}.",
                    cypher=cypher,
                    period="undated source relationship",
                ),
            ],
            assumptions=["District share denotes the documented production-share proxy, not a partner-level export split."],
            confidence=0.65,
        )

    known = ", ".join(districts) if districts else "none"
    reason = (
        f"The graph records {item} producing districts ({known}), but has no sourced numerical district share. "
        "It would be misleading to name a largest contributor."
    )
    return await _respond(
        state,
        deps,
        summary=reason,
        figures={"districts_recorded": float(len(districts))},
        evidence=[
            evidence_from_query(
                claim=f"The knowledge graph records these {item} producing districts: {known}.",
                cypher=cypher,
                period="undated district-membership source",
            ),
            evidence_from_query(
                claim=f"No numerical production or export share is stored for any of the {len(districts)} recorded districts.",
                cypher=cypher,
                period="undated district-membership source",
            ),
        ],
        assumptions=["District membership is sourced; district shares are intentionally absent rather than estimated."],
        confidence=0.35,
    )


async def _model_forecast(
    state: AgentState, deps: AgentDeps, item: str | None, target: str | None, intent: Any
) -> dict[str, Any]:
    if item is None or target is None:
        return await _unsupported_target(state, deps, "No target-compatible agriculture forecast was requested.")
    selected = _load_model(item, target)
    if selected is None:
        return await _unsupported_target(
            state, deps, f"No registered {target.replace('_', ' ')} model is available for {item}."
        )
    model, metadata = selected
    points = model.predict(max(1, min(intent.horizon, 5)))
    if not points or any(point["lower"] > point["point"] or point["point"] > point["upper"] for point in points):
        raise ValueError("registered agriculture model returned an invalid forecast interval")
    unit = str(points[0]["unit"])
    if any(point["unit"] != unit for point in points):
        raise ValueError("registered agriculture model returned inconsistent forecast units")

    head = points[0]
    model_id = metadata.model_id
    label = "tea export volume" if target == "export_volume" else "cinnamon producer price"
    last_values = getattr(model, "_values", [])
    # `ForecastPoint` is intentionally rounded to two decimals. Compare at the
    # same displayed precision: an annual-naive 10.0533 -> 10.05 forecast must
    # not be described as "falling" merely because of formatting residue.
    direction = (
        _direction(round(float(head["point"]), 2) - round(float(last_values[-1]), 2))
        if last_values
        else "projected"
    )
    mape = float((metadata.metrics or {}).get("mape", float("nan")))
    mape_text = f" Rolling-origin MAPE: {mape:.1%}." if mape == mape else ""
    assumptions = [
        f"Served from registered model {model_id}.",
        "The model uses its own annual training series; KG export-value history was not substituted.",
        "Intervals are 80% prediction intervals.",
    ]
    if intent.requested_frequency in {"quarter", "month"}:
        assumptions.append(
            f"You asked about {intent.requested_frequency}s, but this model is annual; returned periods are annual."
        )
    evidence = [
        evidence_from_model(
            claim=(
                f"Registered {label} model forecasts {float(head['point']):,.2f} {unit} for "
                f"{head['period']} ({float(head['lower']):,.2f}–{float(head['upper']):,.2f} at 80%)."
            ),
            model_id=model_id,
            period=str(head["period"]),
        ),
        evidence_from_model(
            claim=(
                f"Model source: {metadata.source or 'not recorded'}; training window "
                f"{(metadata.training_window or {}).get('period_start', '?')}–"
                f"{(metadata.training_window or {}).get('period_end', '?')} with "
                f"{metadata.training_rows or 0} observations.{mape_text}"
            ),
            model_id=model_id,
            period=(
                f"{(metadata.training_window or {}).get('period_start', '?')}-"
                f"{(metadata.training_window or {}).get('period_end', '?')}"
            ),
        ),
    ]
    window = metadata.training_window or {}
    metric = "export_volume" if target == "export_volume" else "price"
    evidence, assumptions, confidence = _with_dq_flags(
        relevant_dq_flags(
            item,
            metric,
            period_start=window.get("period_start"),
            period_end=window.get("period_end"),
            dsn=deps.dsn,
        ),
        evidence,
        assumptions,
        _model_confidence(metadata),
    )
    prefix = f"forecast_next_{target}"
    return await _respond(
        state,
        deps,
        summary=(
            f"{label.title()} is {direction} at {float(head['point']):,.2f} {unit} for "
            f"{head['period']}, with an 80% interval of {float(head['lower']):,.2f} to "
            f"{float(head['upper']):,.2f} {unit}."
        ),
        figures={
            prefix: round(float(head["point"]), 2),
            f"{prefix}_lower": round(float(head["lower"]), 2),
            f"{prefix}_upper": round(float(head["upper"]), 2),
        },
        evidence=evidence,
        assumptions=assumptions,
        confidence=confidence,
        forecast=points,
    )


async def _unsupported_production(state: AgentState, deps: AgentDeps, item: str) -> dict[str, Any]:
    reason = (
        f"A sourced {item} production trend cannot be answered from fact_trade: production remains "
        "staged-only because the frozen schema has no production_volume field."
    )
    return await _unsupported_target(state, deps, reason)


async def _unsupported_substitution(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    cypher = """
    MATCH (:Commodity {name: $from_item})-[r:SUBSTITUTES_WITH]->(:Commodity {name: $to_item})
    RETURN r
    """
    rows, text = await deps.kg.run(cypher, {"from_item": "tea", "to_item": "rubber"})
    reason = (
        "No sourced tea-to-rubber substitution relationship or elasticity is available, so the effect "
        "of a tea-price change on rubber demand cannot be estimated responsibly."
    )
    return await _respond(
        state,
        deps,
        summary=reason,
        figures={},
        evidence=[
            evidence_from_query(claim=reason, cypher=text),
            evidence_from_query(
                claim=f"The substitution lookup returned {len(rows)} sourced tea-to-rubber relationships.",
                cypher=text,
            ),
        ],
        assumptions=["No substitution effect was inferred from two commodities merely sharing the agriculture sector."],
        confidence=0.20,
    )


async def _unsupported_target(state: AgentState, deps: AgentDeps, reason: str) -> dict[str, Any]:
    return await _respond(
        state,
        deps,
        summary=reason,
        figures={},
        evidence=[
            evidence_from_model(claim=reason, model_id="agriculture-agent/data-gap"),
            evidence_from_model(
                claim="No incompatible price, volume, or export-value series was substituted for the requested target.",
                model_id="agriculture-agent/data-gap",
            ),
        ],
        assumptions=[reason],
        confidence=0.20,
    )


async def _respond(
    state: AgentState,
    deps: AgentDeps,
    *,
    summary: str,
    figures: dict[str, float],
    evidence: list[Evidence],
    assumptions: list[str],
    confidence: float,
    forecast: list[ForecastPoint] | None = None,
) -> dict[str, Any]:
    """Use shared response plumbing, retaining figures when the LLM raises."""
    confidence = clamp(confidence)
    try:
        patch = await finish(
            agent=AGENT,
            state=state,
            deps=deps,
            summary=summary,
            figures=figures,
            evidence=evidence,
            assumptions=assumptions,
            forecast=forecast,
        )
    except Exception as exc:  # noqa: BLE001 - LLM outage must retain grounded output
        log.warning("%s explanation unavailable: %s", AGENT, exc)
        output = AgentOutput(
            agent=AGENT,
            summary=summary,
            figures=figures,
            evidence=evidence,
            assumptions=assumptions,
            confidence=confidence,
            degraded=True,
        )
        if forecast:
            output["forecast"] = forecast
        return {"agent_outputs": {AGENT: output}, "degraded": True}

    patch["agent_outputs"][AGENT]["confidence"] = confidence
    return patch


def _clean_series(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"period", "value"}.issubset(frame.columns):
        raise ValueError("annual_series must return period and value columns")
    result = frame.loc[:, ["period", "value"]].copy()
    result["period"] = pd.to_numeric(result["period"], errors="coerce")
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    return result.dropna().sort_values("period", ignore_index=True)


def _series_detail(info: dict[str, object]) -> str:
    return (
        "unified fact_trade annual series: "
        f"source={info['source_id']}; item={info['item']}; target={info['target']}; unit={info['unit']}"
    )


def _with_dq_flags(
    flags: list[dict[str, Any]],
    evidence: list[Evidence],
    assumptions: list[str],
    confidence: float,
) -> tuple[list[Evidence], list[str], float]:
    """Surface material/severe discrepancies without changing measured values.

    Does NOT reduce `confidence` here anymore -- that field is passed through
    unchanged. The DQ penalty is applied exactly once, centrally, by
    `ceynex.orchestrator.merger`'s scan of the merged evidence for these same
    `DQ_FLAG` entries (which is the one place `confidence.py`'s own docstring
    says the formula is allowed to live: "nothing else in the codebase is
    allowed to invent its own"). Applying it here too would have silently
    double-penalized any agriculture answer with a DQ flag once the
    orchestrator-level term was wired up to real data (found 2026-08-26: that
    term had a working formula and its own passing unit tests, but nothing in
    the actual graph ever called `merge()` with a non-default
    `dq_severities`, so it always contributed 0 in production).
    """
    relevant = [flag for flag in flags if flag.get("severity") in {"material", "severe"}]
    if not relevant:
        return evidence, assumptions, confidence

    severities = [str(flag["severity"]) for flag in relevant]
    counts = ", ".join(f"{severity}={severities.count(severity)}" for severity in sorted(set(severities)))
    assumptions = [
        *assumptions,
        f"This answer includes cross-source discrepancy flags ({counts}); measured values were retained, not reconciled.",
    ]
    for flag in relevant:
        period = flag.get("period_start")
        period_text = period.isoformat() if hasattr(period, "isoformat") else str(period)
        pct_diff = float(flag["pct_diff"])
        evidence.append(
            evidence_from_dataset(
                claim=(
                    f"{str(flag['severity']).title()} data-quality flag for {flag['metric']} in {period_text}: "
                    f"{flag['source_a']} reports {float(flag['value_a']):,.2f}, while {flag['source_b']} "
                    f"reports {float(flag['value_b']):,.2f}; difference {pct_diff:.1f}%."
                ),
                detail=(
                    f"dq_flag: source_a={flag['source_a']}; source_b={flag['source_b']}; "
                    f"metric={flag['metric']}; pct_diff={pct_diff:.6g}; severity={flag['severity']}"
                ),
                source_id="DQ_FLAG",
                period=period_text,
            )
        )
    return evidence, assumptions, confidence


def _direction(change: float) -> str:
    if abs(change) < 1e-9:
        return "flat"
    return "rising" if change > 0 else "falling"


def _series_confidence(observations: int, latest_year: int) -> float:
    months_old = max(0, (datetime.now(UTC).year - latest_year) * 12)
    support = min(0.35, observations * 0.035)
    return clamp(0.45 + support - staleness_penalty(months_old))


def _model_confidence(metadata: Any) -> float:
    metrics = metadata.metrics or {}
    mape = metrics.get("mape")
    observations = int(metadata.training_rows or 0)
    latest_year = int((metadata.training_window or {}).get("period_end", datetime.now(UTC).year))
    accuracy_penalty = min(0.30, float(mape)) if isinstance(mape, int | float) else 0.20
    support = min(0.25, observations * 0.015)
    months_old = max(0, (datetime.now(UTC).year - latest_year) * 12)
    return clamp(0.65 + support - accuracy_penalty - staleness_penalty(months_old))


def _load_model(item: str, target: str) -> tuple[Any, Any] | None:
    """Return the best scored model and its auditable registry metadata."""
    from ceynex.models.registry import list_models, load

    candidates = [
        metadata
        for metadata in list_models(sector="agriculture", item=item)
        if metadata.target == target
    ]
    if not candidates:
        return None
    scored = [
        metadata
        for metadata in candidates
        if metadata.metrics is not None and isinstance(metadata.metrics.get("mape"), int | float)
    ]
    selected = min(scored, key=lambda metadata: float(metadata.metrics["mape"])) if scored else max(
        candidates, key=lambda metadata: metadata.saved_at
    )
    return load(selected.sector, selected.item, selected.target, selected.version), selected


__all__ = ["AGENT", "agriculture_commodity_node"]
