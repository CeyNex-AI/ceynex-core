"""Implements SRS 3.1.10 and 3.1.3 — short-horizon forecasts with confidence intervals.

**Never an unqualified number.** SRS 3.1.3 forbids presenting a forecast as a
single figure, and `ForecastPoint` makes `lower` and `upper` required fields so
the contract enforces it rather than relying on discipline. A model that cannot
produce an interval gets one by residual bootstrap; it does not get to omit them.

This node serves models from the registry — M1's agriculture models and M3's
apparel models both land there. Until they do, it falls back to a **drift
forecast with a residual-bootstrap interval** computed from the graph's own
history, and says so in its evidence. That is deliberate: a Day-6 node that
returns nothing until Day 9 cannot be integration-tested, and the fallback is
honest about being a naive baseline rather than dressing itself up as a model.

The baseline also earns its keep permanently — SRS 3.3.4 wants forecast accuracy
compared against a benchmark, and apparel has no published Sri Lankan benchmark
to compare to, so "did it beat drift" is the comparison that remains.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from ceynex.agents.common import (
    AgentDeps,
    evidence_from_model,
    evidence_from_query,
    finish,
    parse_intent,
)
from ceynex.contracts import AgentState, Evidence, ForecastPoint, failed_output
from ceynex.kg.client import KnowledgeGraphUnavailableError

log = logging.getLogger(__name__)

AGENT = "forecast"

# 80% prediction interval is the contract's default (see contracts/forecast.py).
# A different level must be declared in the model's metadata.json and in evidence.
Z_80 = 1.2816
DEFAULT_HORIZON = 2
MIN_OBSERVATIONS = 4


async def forecast_node(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    try:
        return await _forecast(state, deps)
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


async def _forecast(state: AgentState, deps: AgentDeps) -> dict[str, Any]:
    intent = parse_intent(state["query"])
    item = intent.item or "tea"
    horizon = max(1, min(intent.horizon, 5))

    history, cypher = await _history(deps, item, intent.partner)
    figures: dict[str, float] = {}
    evidence: list[Evidence] = []
    assumptions: list[str] = []

    if len(history) < MIN_OBSERVATIONS:
        # SAD §4.1: say what cannot be answered rather than answering it badly.
        assumptions.append(
            f"Only {len(history)} annual observations for {item}; at least "
            f"{MIN_OBSERVATIONS} are needed before a forecast is meaningful."
        )
        evidence.append(
            evidence_from_query(
                claim=(
                    f"The knowledge graph holds {len(history)} annual observations for "
                    f"{item.replace('_', ' ')}, too few to forecast from."
                ),
                cypher=cypher,
            )
        )
        return await finish(
            agent=AGENT,
            state=state,
            deps=deps,
            summary=(
                f"There is not enough history for {item.replace('_', ' ')} to produce a "
                "forecast with a defensible interval."
            ),
            figures=figures,
            evidence=evidence,
            assumptions=assumptions,
        )

    registered = _load_registered_model(item)
    if registered is not None:
        points = registered.predict(horizon)
        # The version is part of the identity. "A registered model produced this"
        # is not a traceable claim if two versions of that model disagree.
        version = getattr(registered, "version", None)
        model_id = f"{registered.sector}/{registered.item}/{registered.target}"
        if version:
            model_id = f"{model_id}@{version}"
        assumptions.append(f"Served from the model registry: {model_id}.")
        evidence.append(
            evidence_from_model(
                claim=(
                    f"Forecast produced by the registered model {model_id} over a "
                    f"{horizon}-period horizon."
                ),
                model_id=model_id,
            )
        )
    else:
        points, diagnostics = _drift_forecast(history, horizon)
        model_id = "baseline/drift+residual-bootstrap"
        figures.update(diagnostics)
        assumptions += [
            "No registered model for this item yet, so the figures come from a drift "
            "baseline: the average year-on-year change carried forward.",
            "The interval is an 80% band from the standard deviation of historical "
            "year-on-year changes, widening with the square root of the horizon.",
            "A naive baseline. It is a floor for a real model to beat, not a substitute "
            "for one.",
        ]
        evidence.append(
            evidence_from_model(
                claim=(
                    f"Drift baseline fitted to {len(history)} annual observations "
                    f"({history[0][0]}-{history[-1][0]}), mean annual change "
                    f"USD {diagnostics['mean_annual_change_usd']:,.0f}."
                ),
                model_id=model_id,
                period=f"{history[0][0]}-{history[-1][0]}",
            )
        )

    evidence.append(
        evidence_from_query(
            claim=(
                f"History used: {len(history)} annual observations for "
                f"{item.replace('_', ' ')} from {history[0][0]} to {history[-1][0]}, "
                f"latest USD {history[-1][1]:,.0f}."
            ),
            cypher=cypher,
            period=f"{history[0][0]}-{history[-1][0]}",
        )
    )

    # Comtrade is annual. Answering a quarterly question with annual periods is
    # acceptable; doing it silently is not (SRS 3.1.3 wants the uncertainty
    # visible, and the period is part of that).
    if intent.requested_frequency in ("quarter", "month"):
        assumptions.append(
            f"You asked about {intent.requested_frequency}s. The underlying trade data is "
            f"annual, so the figures below are annual periods, not {intent.requested_frequency}s."
        )

    figures["latest_actual_usd"] = round(history[-1][1], 2)
    figures["forecast_next_usd"] = round(points[0]["point"], 2)
    figures["forecast_next_lower_usd"] = round(points[0]["lower"], 2)
    figures["forecast_next_upper_usd"] = round(points[0]["upper"], 2)

    summary = (
        f"{item.replace('_', ' ').title()} export value is projected at "
        f"USD {points[0]['point']:,.0f} for {points[0]['period']}, within an 80% interval of "
        f"USD {points[0]['lower']:,.0f} to USD {points[0]['upper']:,.0f}, "
        f"against USD {history[-1][1]:,.0f} in {history[-1][0]}."
    )

    return await finish(
        agent=AGENT,
        state=state,
        deps=deps,
        summary=summary,
        figures=figures,
        evidence=evidence,
        assumptions=assumptions,
        forecast=points,
    )


async def _history(
    deps: AgentDeps, item: str, partner: str | None
) -> tuple[list[tuple[int, float]], str]:
    """Annual export value series for an item, oldest first."""
    cypher = """
    MATCH (i)-[e:EXPORTS_TO]->(c:Country)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND ($partner_iso3 IS NULL OR c.iso3 = $partner_iso3)
    RETURN e.year AS year, sum(e.value) AS value
    ORDER BY year
    """
    rows, text = await deps.kg.run(cypher, {"item": item, "partner_iso3": partner})
    series = [
        (int(row["year"]), float(row["value"]))
        for row in rows
        if row.get("year") is not None and row.get("value")
    ]
    return series, text


def _drift_forecast(
    history: list[tuple[int, float]], horizon: int
) -> tuple[list[ForecastPoint], dict[str, float]]:
    """Drift (mean year-on-year change) with an 80% residual-bootstrap interval.

    The interval widens with sqrt(h) because independent annual shocks accumulate
    in variance, not in standard deviation — a flat band over a 3-year horizon
    would understate the uncertainty it is there to express.
    """
    years = [year for year, _ in history]
    values = [value for _, value in history]

    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    drift = sum(changes) / len(changes)

    mean_change = drift
    variance = sum((c - mean_change) ** 2 for c in changes) / max(1, len(changes) - 1)
    sigma = math.sqrt(variance)

    last_year, last_value = years[-1], values[-1]
    points: list[ForecastPoint] = []
    for step in range(1, horizon + 1):
        point = last_value + drift * step
        band = Z_80 * sigma * math.sqrt(step)
        points.append(
            ForecastPoint(
                period=str(last_year + step),
                point=round(point, 2),
                # Export value cannot go negative; clamping is more honest than
                # a lower bound the quantity cannot reach.
                lower=round(max(0.0, point - band), 2),
                upper=round(point + band, 2),
                unit="USD",
            )
        )

    diagnostics = {
        "observations": float(len(history)),
        "mean_annual_change_usd": round(drift, 2),
        "annual_change_sd_usd": round(sigma, 2),
    }
    return points, diagnostics


def _load_registered_model(item: str) -> Any | None:
    """Look for a model M1 or M3 has registered for this item.

    Imported lazily and failure-tolerantly: the registry lands later in the
    sprint, and this node has to work before it does.
    """
    try:
        from ceynex.models.registry import load_best, load_latest
    except ImportError:
        return None
    try:
        # Best-scoring first. Newest is only the right answer when nothing has
        # been backtested yet, and serving a model with twice the error because
        # it was registered second is a loss nobody would see.
        return load_best(item=item) or load_latest(item=item)
    except Exception as exc:  # noqa: BLE001 - no registered model is the normal case
        log.debug("no registered model for %s: %s", item, exc)
        return None


__all__ = ["AGENT", "forecast_node"]
