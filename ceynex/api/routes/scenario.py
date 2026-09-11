"""POST /api/scenario/run — the scenario workbench (deviation D17).

SRS 3.1.5 asks for exchange-rate, tariff and agreement-loss simulation that
states its assumptions. `agents/trade_economics.py` does that from a question;
this does it from sliders, re-running in place, over the same formulas
(`models/shocks.py`) and the same baseline read (`trade_economics.baseline_value`)
so the two can never disagree.

**Deterministic, and no model is called.** Two bounded graph reads and some
arithmetic. That is why it has its own, generous rate-limit allowance rather
than sharing the query one, and why it can answer in tens of milliseconds while
a slider moves.

**It refuses the way the agent refuses** (SAD §4.1): no baseline in the graph,
or no recorded preference coverage for an agreement shock, means no number —
`refused: true` and a reason, with the assumptions still stated.

**Every parameter comes back with its `basis` and `source`.** Several sources in
`config/elasticities.yaml` are still `TBD` placeholders. The page shows them as
such; a slider over a number nobody has sourced must not look like a fitted one.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ceynex import settings
from ceynex.agents import trade_economics
from ceynex.agents.common import evidence_from_query
from ceynex.api import rate_limit
from ceynex.api.deps import Runtime, get_runtime
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import (
    EvidenceItem,
    ScenarioOutcome,
    ScenarioParameter,
    ScenarioRequest,
    ScenarioResponse,
)
from ceynex.kg import queries as q
from ceynex.models import shocks
from ceynex.retrieval.tagging import HS_FOR_ITEM

log = logging.getLogger(__name__)

router = APIRouter(tags=["scenario"])


# --- rate limiting --------------------------------------------------------

_window_singleton: rate_limit.Window | None = None


def _window() -> rate_limit.Window:
    """Built on first use, not at import — `build_window` reads REDIS_URL."""
    global _window_singleton  # noqa: PLW0603 - one process-lifetime object
    if _window_singleton is None:
        _window_singleton = rate_limit.build_window()
    return _window_singleton


def set_window(window: rate_limit.Window | None) -> None:
    """Test seam. Production never calls this."""
    global _window_singleton  # noqa: PLW0603
    _window_singleton = window


async def enforce_scenario_rate_limit(
    http_request: Request,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> None:
    """SRS 3.4.6's shape, on this endpoint's own `scenario:` allowance — see
    `config/api.yaml` for why it is separate from, and larger than, the query
    limit. Disclosed on the usage page like the others."""
    config = settings.load_config("api").get("scenario_rate_limit", {})
    if not config.get("enabled", True):
        return

    limit = int(config.get("runs_per_minute", 60))
    window_s = int(config.get("window_seconds", 60))
    identity = "scenario:" + rate_limit.identity_of(
        user.email, http_request.client.host if http_request.client else None
    )

    decision = await _window().check(identity, limit, window_s)
    if decision.allowed:
        return

    raise HTTPException(
        status_code=429,
        detail=f"rate limit exceeded: at most {limit} scenario runs per {window_s} seconds.",
        headers={"Retry-After": str(decision.retry_after_s)},
    )


# --- run ------------------------------------------------------------------


@router.post(
    "/api/scenario/run",
    response_model=ScenarioResponse,
    dependencies=[Depends(enforce_scenario_rate_limit)],
)
async def run_scenario(
    request: ScenarioRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
    runtime: Runtime = Depends(get_runtime),  # noqa: B008
) -> ScenarioResponse:
    if not settings.scenario_enabled():
        raise HTTPException(status_code=404, detail="the scenario workbench is not enabled")

    sector = request.sector
    item = request.item or trade_economics._representative_item(sector, None)
    if trade_economics.SECTOR_OF_ITEM.get(item) != sector:
        raise HTTPException(
            status_code=422,
            detail=f"{item!r} is not a {sector} item; one of "
            f"{sorted(k for k, v in trade_economics.SECTOR_OF_ITEM.items() if v == sector)}",
        )
    overrides = request.overrides.model_dump() if request.overrides else {}
    config = settings.elasticity_config()

    # For an agreement shock the "magnitude" is the MFN rate itself, which the
    # parameters carry; the assumption line still names what was modelled.
    assumptions = shocks.base_assumptions(config, request.shock, request.magnitude)
    assumptions.append(
        "Run from the scenario workbench: the same formulas as the trade-economics "
        "analysis, over the same baseline, with no policy document consulted."
    )

    try:
        baseline, year, cypher = await trade_economics.baseline_value(runtime.kg, item)
    except Exception as exc:  # noqa: BLE001 - a graph outage is a 503, not a 500
        log.warning("scenario baseline read failed: %s", exc)
        raise HTTPException(status_code=503, detail="knowledge graph unavailable") from exc

    common = {
        "shock": request.shock,
        "sector": sector,
        "item": item,
        "magnitude": request.magnitude,
        "baseline_year": year,
        "baseline_cypher": " ".join(cypher.split()),
    }
    if baseline is None:
        return ScenarioResponse(
            **common,
            refused=True,
            reason=f"No baseline export value for {item} in the knowledge graph, so the "
            "impact cannot be simulated.",
            assumptions=assumptions,
            evidence=[
                _item(evidence_from_query(
                    claim=f"The knowledge graph records no export value for {item} in {year}.",
                    cypher=cypher, period=str(year),
                ))
            ],
        )

    evidence = [
        _item(evidence_from_query(
            claim=f"{sector.title()} exports of {item} were USD {baseline:,.0f} in {year}; "
            "this is the baseline the shock moves.",
            cypher=cypher, period=str(year),
        ))
    ]

    if request.shock == "fx":
        outcome = shocks.fx_shock(sector, baseline, request.magnitude, config, overrides=overrides)
    elif request.shock == "tariff":
        outcome = shocks.tariff_shock(
            sector, baseline, request.magnitude, config, overrides=overrides
        )
    else:
        hs_code = HS_FOR_ITEM.get(item, "61")
        try:
            rows, coverage_cypher = await runtime.kg.run(*q.agreement_coverage(hs_code))
        except Exception as exc:  # noqa: BLE001
            log.warning("scenario coverage read failed: %s", exc)
            raise HTTPException(status_code=503, detail="knowledge graph unavailable") from exc
        preferences = [r for r in rows if r.get("agreement_type") == "unilateral_preference"]
        if not preferences:
            evidence.append(_item(evidence_from_query(
                claim=f"No trade-agreement coverage is recorded for {item} (HS {hs_code}) in "
                "the knowledge graph, so an agreement-loss simulation would rest on an "
                "unchecked assumption.",
                cypher=coverage_cypher, period=str(year),
            )))
            return ScenarioResponse(
                **common, refused=True, baseline_usd=baseline,
                reason=f"The knowledge graph does not record preference coverage for {item}, "
                "so the effect of losing it cannot be simulated.",
                assumptions=assumptions, evidence=evidence,
            )
        outcome = shocks.agreement_loss_shock(
            sector, baseline, config,
            coverage=shocks.describe_coverage(preferences), overrides=overrides,
        )
        rate = outcome.parameters[0]
        evidence.append(_item(evidence_from_query(
            claim=f"Preference coverage for {item} (HS {hs_code}) is recorded in the knowledge "
            f"graph, and losing it is modelled as an MFN tariff of {rate.value * 100:.1f}%.",
            cypher=coverage_cypher, period=str(year),
        )))

    assumptions.append(outcome.detail)
    return ScenarioResponse(
        **common,
        refused=False,
        baseline_usd=baseline,
        outcome=ScenarioOutcome(
            **{k: v for k, v in outcome.as_dict().items() if k != "parameters"},
            parameters=[ScenarioParameter(**p.as_dict()) for p in outcome.parameters],
        ),
        assumptions=assumptions,
        evidence=evidence,
    )


def _item(evidence: dict) -> EvidenceItem:
    return EvidenceItem(**evidence)


__all__ = ["enforce_scenario_rate_limit", "router", "run_scenario", "set_window"]
