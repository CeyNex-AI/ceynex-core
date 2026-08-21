"""POST /api/auth/login and GET /api/auth/me — SRS 3.1.11.

Login checks a fixed demo account (see `ceynex/api/auth.py`) and issues a
signed JWT. `/me` exists so any future protected route (the admin routes in
docs/DEFERRED.md, when they're built) can share `require_user` as one
verification path instead of each re-deriving it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ceynex.api.auth import TokenPayload, authenticate, issue_token, verify_token
from ceynex.api.schemas import LoginRequest, LoginResponse, UserResponse

router = APIRouter(tags=["auth"])

_bearer = HTTPBearer(auto_error=False)


@router.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest) -> LoginResponse:
    user = authenticate(request.email, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid email or password")
    return LoginResponse(token=issue_token(user), email=user.email, role=user.role)


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),  # noqa: B008
) -> TokenPayload:
    """FastAPI dependency for any route that needs a signed-in user."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return payload


@router.get("/api/auth/me", response_model=UserResponse)
async def me(user: TokenPayload = Depends(require_user)) -> UserResponse:  # noqa: B008
    return UserResponse(email=user.email, role=user.role)
