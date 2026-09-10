"""POST /api/auth/signup, POST /api/auth/login and GET /api/auth/me — SRS 3.1.11.

Signup creates a real `users` row — at whichever of `users.SIGNUP_ROLES` the
request names, or `DEFAULT_ROLE` if it names none; never `admin`, which is
admin-provisioned only — and logs the new account straight in. Login checks a
stored bcrypt hash (see `ceynex/api/users.py`) and issues a signed JWT. `/me` exists so any protected route can share
`require_user` as one verification path instead of each re-deriving it.
Admin-provisioned accounts and role changes live on the admin router
(`routes/admin.py`), behind `require_admin`.

Login and signup are rate limited (SRS 3.4.6's mechanism, its own config
block and identity namespace) — `/api/query`'s limiter protects *serving
capacity*, this one is about credential stuffing and signup spam. Every
attempt is counted twice: once against the client address, once against the
email in the body, so neither "one IP, many emails" nor "one email, many
IPs" gets a free pass. bcrypt already makes each attempt cost real CPU; this
bounds a script that does not care.
"""

from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ceynex import settings
from ceynex.api import api_keys, rate_limit, users
from ceynex.api.auth import TokenPayload, authenticate, issue_token, verify_token
from ceynex.api.schemas import LoginRequest, LoginResponse, SignupRequest, UserResponse

router = APIRouter(tags=["auth"])

_bearer = HTTPBearer(auto_error=False)


# --- rate limiting -------------------------------------------------------

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


async def _enforce_auth_rate_limit(http_request: Request, *, email: str | None) -> None:
    config = settings.load_config("api").get("auth_rate_limit", {})
    if not config.get("enabled", True):
        return

    limit = int(config.get("attempts_per_minute", 10))
    window_s = int(config.get("window_seconds", 60))
    host = rate_limit.client_ip(http_request)

    # The `auth:` prefix keeps these off `POST /api/query`'s Redis keys (see
    # routes/news.py for the same reasoning). Both an IP counter and an email
    # counter must pass.
    identities = [f"auth:ip:{host or 'unknown'}"]
    if email:
        identities.append(f"auth:email:{email.strip().lower()}")

    for identity in identities:
        decision = await _window().check(identity, limit, window_s)
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"too many attempts: at most {limit} per {window_s} seconds. "
                    f"Try again in {decision.retry_after_s}s."
                ),
                headers={"Retry-After": str(decision.retry_after_s)},
            )


@router.post("/api/auth/signup", response_model=LoginResponse, status_code=201)
async def signup(request: SignupRequest, http_request: Request) -> LoginResponse:
    await _enforce_auth_rate_limit(http_request, email=request.email)

    role = request.role or users.DEFAULT_ROLE
    if role == "admin":
        raise HTTPException(status_code=403, detail="the admin role cannot be self-assigned")
    if role not in users.SIGNUP_ROLES:
        raise HTTPException(
            status_code=422, detail=f"role must be one of {list(users.SIGNUP_ROLES)}"
        )

    try:
        user = users.create_user(request.email, request.password, role)
    except users.EmailTakenError as exc:
        raise HTTPException(status_code=409, detail="an account with that email already exists") from exc
    except users.WeakPasswordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="could not create account") from exc
    return LoginResponse(token=issue_token(user.email, user.role, user.token_epoch), email=user.email, role=user.role)


@router.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, http_request: Request) -> LoginResponse:
    await _enforce_auth_rate_limit(http_request, email=request.email)
    try:
        user = authenticate(request.email, request.password)
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="login temporarily unavailable") from exc
    if user is None:
        raise HTTPException(status_code=401, detail="invalid email or password")
    return LoginResponse(token=issue_token(user.email, user.role, user.token_epoch), email=user.email, role=user.role)


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
