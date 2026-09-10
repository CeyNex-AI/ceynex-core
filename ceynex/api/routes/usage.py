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

from fastapi import APIRouter, Depends, Query

from ceynex import settings
from ceynex.api.routes.auth import TokenPayload, require_admin, require_user
from ceynex.api.schemas import UsageLimitsResponse, UsageRollupItem, UsageSummaryResponse
from ceynex.observability import ledger

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


@router.get("/api/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(
    days: int = Query(30, ge=1, le=MAX_DAYS),
    user: TokenPayload = Depends(require_user),  # noqa: B008 - FastAPI's dependency idiom
) -> UsageSummaryResponse:
    """This caller's own spend. Scoped to them and never to anyone else."""
    by_day = await ledger.by_day(user.email, days=days)
    by_role = await ledger.by_role_and_model(user.email, days=days)
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
    by_day = await ledger.by_day(None, days=days)
    by_role = await ledger.by_role_and_model(None, days=days)
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
    so the page cannot drift from what is actually enforced. `spent_today` comes
    from the ledger, which is cross-worker accurate; the *cap* it is compared
    against is not, and the response says so plainly rather than presenting a
    number the deployment cannot actually hold to.
    """
    api = settings.load_config("api")
    llm = settings.load_config("llm")
    limits = llm.get("limits", {})
    cap = float(limits.get("daily_spend_cap_usd", 0) or 0)

    described = []
    for block, label, field in (
        ("rate_limit", "Questions", "query_per_minute"),
        ("chat_rate_limit", "Conversation turns", "turns_per_minute"),
        ("news_rate_limit", "News searches", "search_per_minute"),
        ("graph_rate_limit", "Graph expansions", "expand_per_minute"),
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

    return UsageLimitsResponse(
        limits=described,
        daily_spend_cap_usd=cap,
        spent_today_usd=round(await ledger.spent_today(), 6),
        # Named rather than hidden. `_cap_reached()` reads a per-process counter
        # and the deployed image runs two workers, so the true ceiling is up to
        # twice the configured cap. A page that showed the cap as though it were
        # exact would be the silent enforcement the requirement forbids.
        cap_is_per_worker=True,
        worker_count=2,
    )


__all__ = ["router"]
