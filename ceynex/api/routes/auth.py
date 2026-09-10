"""POST /api/auth/signup, POST /api/auth/login and GET /api/auth/me — SRS 3.1.11.

Signup creates a real `users` row at the default role and logs the new account
straight in; login checks a stored bcrypt hash (see `ceynex/api/users.py`) and
issues a signed JWT. `/me` exists so any protected route can share
`require_user` as one verification path instead of each re-deriving it.
Admin-provisioned accounts and role changes live on the admin router
(`routes/admin.py`), behind `require_admin`.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ceynex.api import api_keys, users
from ceynex.api.auth import TokenPayload, authenticate, issue_token, verify_token
from ceynex.api.schemas import LoginRequest, LoginResponse, SignupRequest, UserResponse

router = APIRouter(tags=["auth"])

_bearer = HTTPBearer(auto_error=False)


@router.post("/api/auth/signup", response_model=LoginResponse, status_code=201)
async def signup(request: SignupRequest) -> LoginResponse:
    try:
        user = users.create_user(request.email, request.password, users.DEFAULT_ROLE)
    except users.EmailTakenError as exc:
        raise HTTPException(status_code=409, detail="an account with that email already exists") from exc
    except users.WeakPasswordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not create account") from exc
    return LoginResponse(token=issue_token(user.email, user.role), email=user.email, role=user.role)


@router.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    try:
        user = authenticate(request.email, request.password)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="login temporarily unavailable") from exc
    if user is None:
        raise HTTPException(status_code=401, detail="invalid email or password")
    return LoginResponse(token=issue_token(user.email, user.role), email=user.email, role=user.role)


def _verify_bearer(token: str) -> TokenPayload | None:
    """A `ck_`-prefixed token is an API key (`ceynex/api/api_keys.py`);
    anything else is a login JWT. Shared by `require_user` and
    `get_optional_user` so a key works everywhere a token does — query
    history included, which is the point of "programmatic access"."""
    if token.startswith(api_keys.KEY_PREFIX):
        return api_keys.authenticate(token)
    return verify_token(token)


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),  # noqa: B008
) -> TokenPayload:
    """FastAPI dependency for any route that needs a signed-in user."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    payload = _verify_bearer(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return payload


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),  # noqa: B008
) -> TokenPayload | None:
    """Like `require_user`, but never 401s — None instead of raising when
    there's no (valid) token. For routes where being signed in unlocks
    something extra (query history) without gating the route itself."""
    if credentials is None:
        return None
    return _verify_bearer(credentials.credentials)


def require_admin(user: TokenPayload = Depends(require_user)) -> TokenPayload:  # noqa: B008
    """For the admin routes (SRS 3.5.4) — a valid token is not enough, the role
    on it has to be "admin". The frontend already hides the Admin nav link and
    its own page content for other roles, but that is client-side convenience,
    not enforcement: without this, any signed-in researcher/exporter/
    policymaker could hit these routes directly and trigger a real retrain or
    ingest run."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return user


@router.get("/api/auth/me", response_model=UserResponse)
async def me(user: TokenPayload = Depends(require_user)) -> UserResponse:  # noqa: B008
    return UserResponse(email=user.email, role=user.role)
