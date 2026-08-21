"""GET /api/history — SRS 3.5.2.

Recording happens opportunistically in `query.py`; this route only reads.
Unlike `POST /api/query`'s optional dependency, this one requires a valid
token — listing someone's history without knowing who they are makes no sense.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import history
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import QueryHistoryItem, QueryHistoryResponse

router = APIRouter(tags=["history"])


@router.get("/api/history", response_model=QueryHistoryResponse)
async def get_history(user: TokenPayload = Depends(require_user)) -> QueryHistoryResponse:  # noqa: B008
    try:
        entries = history.list_for_user(user.email)
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
            )
            for e in entries
        ]
    )
