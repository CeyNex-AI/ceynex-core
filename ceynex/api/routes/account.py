"""GET/PUT notification preferences and API-key management — Account.tsx's
two remaining "planned, not built yet" stub items, now built.

All routes require a signed-in user (`require_user`); preferences and keys
are always scoped to the caller's own email, the same ownership pattern as
`routes/history.py`.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import api_keys, preferences
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import (
    ApiKeyItem,
    ApiKeyListResponse,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    NotificationPreferences,
    RevokeApiKeyResponse,
    UserInstructionRequest,
    UserInstructionResponse,
)
from ceynex.chat import instructions

router = APIRouter(tags=["account"])


@router.get("/api/account/preferences", response_model=NotificationPreferences)
async def get_preferences(user: TokenPayload = Depends(require_user)) -> NotificationPreferences:  # noqa: B008
    try:
        prefs = preferences.get_for_user(user.email)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="preferences unavailable") from exc
    return NotificationPreferences(
        dq_flag_alerts=prefs.dq_flag_alerts,
        forecast_updates=prefs.forecast_updates,
        weekly_digest=prefs.weekly_digest,
    )


@router.put("/api/account/preferences", response_model=NotificationPreferences)
async def put_preferences(
    body: NotificationPreferences,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> NotificationPreferences:
    try:
        prefs = preferences.upsert(
            user.email,
            dq_flag_alerts=body.dq_flag_alerts,
            forecast_updates=body.forecast_updates,
            weekly_digest=body.weekly_digest,
        )
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not update preferences") from exc
    return NotificationPreferences(
        dq_flag_alerts=prefs.dq_flag_alerts,
        forecast_updates=prefs.forecast_updates,
        weekly_digest=prefs.weekly_digest,
    )


@router.get("/api/account/api-keys", response_model=ApiKeyListResponse)
async def list_api_keys(user: TokenPayload = Depends(require_user)) -> ApiKeyListResponse:  # noqa: B008
    try:
        entries = api_keys.list_for_user(user.email)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="api keys unavailable") from exc
    return ApiKeyListResponse(
        keys=[
            ApiKeyItem(
                id=e.id,
                label=e.label,
                key_prefix=e.key_prefix,
                created_at=e.created_at,
                last_used_at=e.last_used_at,
                revoked=e.revoked,
            )
            for e in entries
        ]
    )


@router.post("/api/account/api-keys", response_model=CreateApiKeyResponse)
async def create_api_key(
    body: CreateApiKeyRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> CreateApiKeyResponse:
    try:
        new_key = api_keys.create(user.email, body.label)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not create api key") from exc
    return CreateApiKeyResponse(
        id=new_key.id,
        label=new_key.label,
        key=new_key.key,
        key_prefix=new_key.key_prefix,
        created_at=new_key.created_at,
    )


@router.post("/api/account/api-keys/{key_id}/revoke", response_model=RevokeApiKeyResponse)
async def revoke_api_key(
    key_id: int,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> RevokeApiKeyResponse:
    try:
        found = api_keys.revoke(key_id, user.email)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not revoke api key") from exc
    if not found:
        raise HTTPException(status_code=404, detail=f"no api key with id {key_id}")
    return RevokeApiKeyResponse(id=key_id, revoked=True)


@router.get("/api/account/instructions", response_model=UserInstructionResponse)
async def get_instructions(
    user: TokenPayload = Depends(require_user),  # noqa: B008 - FastAPI's dependency idiom
) -> UserInstructionResponse:
    """This reader's standing preference about how answers read (D15)."""
    content, enabled = await instructions.get(user.email)
    return UserInstructionResponse(
        content=content, enabled=enabled, max_chars=instructions.MAX_INSTRUCTION_CHARS
    )


@router.put("/api/account/instructions", response_model=UserInstructionResponse)
async def put_instructions(
    request: UserInstructionRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> UserInstructionResponse:
    """Save it. Tone only — see `ceynex/chat/instructions.py` for what this
    cannot do, which is the more important half of the feature."""
    content = request.content.strip()[: instructions.MAX_INSTRUCTION_CHARS]
    try:
        await instructions.save(user.email, content, request.enabled)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not save instructions") from exc
    return UserInstructionResponse(
        content=content, enabled=request.enabled, max_chars=instructions.MAX_INSTRUCTION_CHARS
    )
