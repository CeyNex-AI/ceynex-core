"""What the language models cost, and what the limits are — SRS 3.4.6, D15.

**SRS 3.4.6 is the reason this exists**, and it is worth quoting because the
requirement is easy to read as already met: *"any usage restrictions … such as
rate limits … must be disclosed to the user within the application rather than
enforced silently."* The rate limiter has been enforcing since it shipped; until
there was a page saying so, it was enforcing silently. `GET /api/usage/limits`
is the disclosure.

Everything else here reads `observability/ledger.py`, which has been recording
one row per LLM call all along.
"""

from __future__ import annotations

import logging

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query

from ceynex import settings
from ceynex.api.routes.auth import TokenPayload, require_admin, require_user
from ceynex.api.schemas import UsageLimitsResponse, UsageRollupItem, UsageSummaryResponse
from ceynex.observability import ledger, spend

log = logging.getLogger(__name__)

router = APIRouter(tags=["usage"])

MAX_DAYS = 365


def _rollups(rows) -> list[UsageRollupItem]:
    return [
        UsageRollupItem(
            key=row.key,
            calls=row.calls,
            tokens_in=row.tokens_in,
            tokens_out=row.tokens_out,
            cost_usd=round(row.cost_usd, 6),
        )
        for row in rows
    ]


async def _rollups_or_503(user_email: str | None, days: int):
    """The ledger's rollups, or a 503 that says the ledger is the problem.

    Unlike history recording, a *read* of spend must not quietly return zero on
    an outage: "you have spent nothing" is a claim, and it would be false.
    """
    try:
        return (
            await ledger.by_day(user_email, days=days),
            await ledger.by_role_and_model(user_email, days=days),
        )
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="usage ledger unavailable") from exc


@router.get("/api/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(
    days: int = Query(30, ge=1, le=MAX_DAYS),
    user: TokenPayload = Depends(require_user),  # noqa: B008 - FastAPI's dependency idiom
) -> UsageSummaryResponse:
    """This caller's own spend. Scoped to them and never to anyone else."""
    by_day, by_role = await _rollups_or_503(user.email, days)
    return UsageSummaryResponse(
        days=days,
        scope="user",
        by_day=_rollups(by_day),
        by_role=_rollups(by_role),
        total_cost_usd=round(sum(r.cost_usd for r in by_day), 6),
        total_calls=sum(r.calls for r in by_day),
        total_tokens_in=sum(r.tokens_in for r in by_day),
        total_tokens_out=sum(r.tokens_out for r in by_day),
    )


@router.get("/api/usage/all", response_model=UsageSummaryResponse)
async def usage_all(
    days: int = Query(30, ge=1, le=MAX_DAYS),
    user: TokenPayload = Depends(require_admin),  # noqa: B008
) -> UsageSummaryResponse:
    """Every user's spend. `require_admin`, because it is everyone's data."""
    by_day, by_role = await _rollups_or_503(None, days)
    return UsageSummaryResponse(
        days=days,
        scope="all",
        by_day=_rollups(by_day),
        by_role=_rollups(by_role),
        total_cost_usd=round(sum(r.cost_usd for r in by_day), 6),
        total_calls=sum(r.calls for r in by_day),
        total_tokens_in=sum(r.tokens_in for r in by_day),
        total_tokens_out=sum(r.tokens_out for r in by_day),
    )


@router.get("/api/usage/limits", response_model=UsageLimitsResponse)
async def usage_limits(
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> UsageLimitsResponse:
    """The limits in force, in the reader's own words — SRS 3.4.6's disclosure.

    Read from `config/api.yaml` and `config/llm.yaml` rather than restated here,
    so the page cannot drift from what is actually enforced.

    Two sources for spend, and they are different numbers for a reason. The
    ledger is the accounting — cross-worker accurate, one row per call — so it
    is what the page reports. The counter in `observability/spend.py` is the
    enforcement, answering before every paid call; it is shared through Redis
    when `REDIS_URL` is set, and `cap_is_per_worker` says plainly when it is not.
    If the ledger cannot be read, the counter's figure is reported instead
    rather than a false zero.
    """
    api = settings.load_config("api")
    llm = settings.load_config("llm")
    limits = llm.get("limits", {})
    cap = float(limits.get("daily_spend_cap_usd", 0) or 0)
    per_user = float(limits.get("per_user_daily_cap_usd", 0) or 0)

    described = []
    for block, label, field in (
        ("rate_limit", "Questions", "query_per_minute"),
        ("chat_rate_limit", "Conversation turns", "turns_per_minute"),
        ("news_rate_limit", "News searches", "search_per_minute"),
        ("graph_rate_limit", "Graph expansions", "expand_per_minute"),
        ("scenario_rate_limit", "Scenario runs", "runs_per_minute"),
    ):
        config = api.get(block, {})
        if not config.get("enabled", True):
            continue
        value = config.get(field)
        if value is None:
            continue
        described.append(
            UsageRollupItem(
                key=f"{label}: {value} per {int(config.get('window_seconds', 60))}s",
                calls=int(value),
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
            )
        )

    counter = spend.shared_counter()
    counted = await counter.spent(user.email)
    try:
        total = await ledger.spent_today()
        yours = await ledger.spent_today(user.email)
    except psycopg.Error as exc:
        log.warning("ledger unreadable; reporting the enforcement counter instead: %s", exc)
        total, yours = counted.total_usd, counted.user_usd

    return UsageLimitsResponse(
        limits=described,
        daily_spend_cap_usd=cap,
        spent_today_usd=round(total, 6),
        # Named rather than hidden. Without a shared counter each worker counts
        # only its own spend, so the true ceiling is `worker_count` times the cap.
        # A page that showed the cap as exact would be the silent enforcement the
        # requirement forbids.
        cap_is_per_worker=not counter.shared,
        worker_count=2,
        per_user_daily_cap_usd=per_user,
        spent_today_by_you_usd=round(yours, 6),
        resets_at=spend.resets_at().isoformat(),
        your_budget_spent=bool(per_user and counted.user_usd >= per_user),
        deployment_cap_spent=bool(cap and counted.total_usd >= cap),
    )


__all__ = ["router"]
