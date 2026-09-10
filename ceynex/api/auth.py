"""Credential check and token issue/verify — SRS 3.1.11.

Backed by the `users` table (`ceynex/api/users.py`) since RBAC landed. This
module used to hold four fixed demo accounts with a shared password; that was
always flagged as "swap for a real table the day this needs to be more than a
demo" and this is that day. What stays here is the token half — issuing a
signed JWT at login and verifying one on the way back in — which has no
database in it and is worth keeping as a pure function for the route tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jwt

from ceynex.api import users
from ceynex.settings import jwt_secret

ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 60 * 60 * 8  # 8h — long enough for one working session


def role_for_email(email: str) -> str | None:
    """The role an account currently authenticates at, or None if there is no
    such account or it is disabled. `api_keys.py` calls this on every request
    so a key always tracks its account's *current* role and stops working the
    moment the account is disabled or deleted."""
    user = users.get_by_email(email)
    if user is None or user.disabled:
        return None
    return user.role


def authenticate(email: str, password: str) -> users.User | None:
    """None on any failure — unknown email, wrong password, disabled account —
    with no way for a caller to tell those apart. Thin passthrough to
    `users.authenticate`; kept here so `routes/auth.py` imports its whole auth
    surface from one module."""
    return users.authenticate(email, password)


def issue_token(email: str, role: str) -> str:
    now = int(time.time())
    payload = {"sub": email, "role": role, "iat": now, "exp": now + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, jwt_secret(), algorithm=ALGORITHM)


@dataclass(frozen=True)
class TokenPayload:
    email: str
    role: str


def verify_token(token: str) -> TokenPayload | None:
    """None on any failure (expired, malformed, wrong signature) — one outcome
    for the route layer to turn into a 401, no exception type to keep in sync.

    The role comes straight off the token's own claim, set at login time — a
    role change made after this token was issued is not reflected until the
    token expires (see `ceynex/api/users.py`'s module docstring)."""
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return TokenPayload(email=payload["sub"], role=payload["role"])
