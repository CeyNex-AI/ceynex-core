"""Account self-service: notification preferences, API-key management, password
change, email change, and account deletion.

All routes require a signed-in user (`require_user`) and act only on the
caller's own account — same ownership pattern as `routes/history.py`. The three
identity-affecting routes (password, email, delete) each re-check the current
password first: a valid token is not enough to change or destroy an account,
in case the token was lifted. An admin changing *someone else's* password is a
separate route on the admin router (`routes/admin.py`), behind `require_admin`
and audited.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from ceynex.api import api_keys, preferences, users
from ceynex.api.auth import authenticate, issue_token
from ceynex.api.routes.auth import TokenPayload, require_user
from ceynex.api.schemas import (
    AccountDeletedResponse,
    ApiKeyItem,
    ApiKeyListResponse,
    ChangeEmailRequest,
    ChangePasswordRequest,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    DeleteAccountRequest,
    LoginResponse,
    NotificationPreferences,
    PasswordChangedResponse,
    RevokeApiKeyResponse,
    UserInstructionRequest,
    UserInstructionResponse,
)
from ceynex.chat import instructions

router = APIRouter(tags=["account"])


def _require_current_password(email: str, password: str) -> users.User:
    """Re-authenticate the caller by password. 403 on a wrong password or an
    account that has since been disabled/deleted — one outcome, so a caller
    can't tell those apart. 503 if Postgres is unreachable."""
    try:
        user = authenticate(email, password)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="account service unavailable") from exc
    if user is None:
        raise HTTPException(status_code=403, detail="current password is incorrect")
    return user


@router.post("/api/account/password", response_model=PasswordChangedResponse)
async def change_password(
    body: ChangePasswordRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> PasswordChangedResponse:
    """Self-service password change. Requires the current password (proof the
    session isn't just a stolen token); the new one goes through the same
    8-char floor as signup.

    `users.set_password` bumps the account's `token_epoch`, which invalidates
    every session for it — including this request's own token — at the next
    request. The response carries a fresh token so the caller's device stays
    signed in while every *other* session is cut."""
    if body.new_password == body.current_password:
        raise HTTPException(status_code=422, detail="new password must be different")
    current = _require_current_password(user.email, body.current_password)
    try:
        updated = users.set_password(current.id, body.new_password)
    except users.WeakPasswordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not change password") from exc
    if updated is None:
        raise HTTPException(status_code=403, detail="current password is incorrect")
    return PasswordChangedResponse(
        email=updated.email,
        token=issue_token(updated.email, updated.role, updated.token_epoch),
    )


@router.post("/api/account/email", response_model=LoginResponse)
async def change_email(
    body: ChangeEmailRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> LoginResponse:
    """Change the caller's own email. Current password required. The new
    address must be free (409 otherwise). `users.set_email` bumps
    `token_epoch` and moves the caller's history / API keys / preferences to
    the new address; the response carries a fresh token (the old one's `sub`
    is the old email) so the caller stays signed in."""
    current = _require_current_password(user.email, body.current_password)
    try:
        updated = users.set_email(current.id, body.new_email)
    except users.EmailTakenError as exc:
        raise HTTPException(status_code=409, detail="that email is already in use") from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not change email") from exc
    if updated is None:
        raise HTTPException(status_code=403, detail="current password is incorrect")
    return LoginResponse(
        token=issue_token(updated.email, updated.role, updated.token_epoch),
        email=updated.email,
        role=updated.role,
    )


@router.delete("/api/account", response_model=AccountDeletedResponse)
async def delete_account(
    body: DeleteAccountRequest,
    user: TokenPayload = Depends(require_user),  # noqa: B008
) -> AccountDeletedResponse:
    """Delete the caller's own account and everything keyed to its email
    (history, API keys, preferences). Current password required. Refused (409)
    if the caller is the last enabled admin — deleting your way to a
    zero-admin deployment is the same lockout the admin routes guard against."""
    current = _require_current_password(user.email, body.current_password)
    try:
        removed = users.delete_user(current.id)
    except users.LastAdminError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not delete the account") from exc
    if not removed:
        raise HTTPException(status_code=403, detail="current password is incorrect")
    return AccountDeletedResponse(deleted=True)


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
