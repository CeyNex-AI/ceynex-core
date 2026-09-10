"""Account self-service: notification preferences, API-key management, and
password change.

All routes require a signed-in user (`require_user`) and act only on the
caller's own account — same ownership pattern as `routes/history.py`. An admin
changing *someone else's* password is a separate route on the admin router
(`routes/admin.py`), behind `require_admin` and audited.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import api_keys, preferences, users
from ceynex.api.auth import authenticate
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import (
    ApiKeyItem,
    ApiKeyListResponse,
    ChangePasswordRequest,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    NotificationPreferences,
    PasswordChangedResponse,
    RevokeApiKeyResponse,
)

router = APIRouter(tags=["account"])


@router.post("/api/account/password", response_model=PasswordChangedResponse)
async def change_password(
    body: ChangePasswordRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> PasswordChangedResponse:
    """Self-service password change. Requires the current password (proof the
    session isn't just a stolen token); the new one goes through the same
    8-char floor as signup. Existing tokens — including this one — stay valid
    until they expire (see `users.set_password`)."""
    if body.new_password == body.current_password:
        raise HTTPException(status_code=422, detail="new password must be different")
    try:
        current = authenticate(user.email, body.current_password)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="password change unavailable") from exc
    if current is None:
        # Wrong current password, or the account was disabled/deleted since the
        # token was issued — one outcome, same reasoning as `users.authenticate`.
        raise HTTPException(status_code=403, detail="current password is incorrect")
    try:
        updated = users.set_password(current.id, body.new_password)
    except users.WeakPasswordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not change password") from exc
    if updated is None:
        raise HTTPException(status_code=403, detail="current password is incorrect")
    return PasswordChangedResponse(email=updated.email)


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
