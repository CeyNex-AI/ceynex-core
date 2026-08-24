"""GET /api/history and the save/unsave actions — SRS 3.5.2.

Recording happens opportunistically in `query.py`; these routes only read and
toggle. All three require a valid token — listing or saving someone's history
without knowing who they are makes no sense.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import history
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import QueryHistoryItem, QueryHistoryResponse, SaveQueryResponse

router = APIRouter(tags=["history"])


@router.get("/api/history", response_model=QueryHistoryResponse)
async def get_history(
    saved: bool | None = None,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> QueryHistoryResponse:
    try:
        entries = history.list_for_user(user.email, saved=saved)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="history unavailable") from exc

    return QueryHistoryResponse(
        items=[
            QueryHistoryItem(
                id=e.id,
                query=e.query,
                answer=e.answer,
                confidence=e.confidence,
                degraded=e.degraded,
                asked_at=e.asked_at,
                saved=e.saved,
            )
            for e in entries
        ]
    )


def _set_saved(entry_id: int, user: TokenPayload, *, saved: bool) -> SaveQueryResponse:
    try:
        found = history.set_saved(entry_id, user.email, saved=saved)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not update history") from exc
    if not found:
        raise HTTPException(status_code=404, detail=f"no history entry with id {entry_id}")
    return SaveQueryResponse(id=entry_id, saved=saved)


@router.post("/api/history/{entry_id}/save", response_model=SaveQueryResponse)
async def save_query(
    entry_id: int,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> SaveQueryResponse:
    return _set_saved(entry_id, user, saved=True)


@router.post("/api/history/{entry_id}/unsave", response_model=SaveQueryResponse)
async def unsave_query(
    entry_id: int,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> SaveQueryResponse:
    return _set_saved(entry_id, user, saved=False)
