"""Credential check and token issue/verify — SRS 3.1.11.

Backed by the `users` table (`ceynex/api/users.py`) since RBAC landed. This
module used to hold four fixed demo accounts with a shared password; that was
always flagged as "swap for a real table the day this needs to be more than a
demo" and this is that day.

`verify_token` now makes one indexed DB lookup per authed request
(`users.current_token_epoch`) so that a password change, role change or
disable cuts existing sessions at their next request instead of waiting out
the 8 h token TTL. If that lookup fails (Postgres unreachable) the token is
accepted on its signature alone — a datastore blip must not log everyone out,
same degrade-don't-fail posture as the rest of the codebase.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import jwt
import psycopg

from ceynex.api import users
from ceynex.settings import jwt_secret

log = logging.getLogger(__name__)

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


def issue_token(email: str, role: str, token_epoch: int = 0) -> str:
    now = int(time.time())
    payload = {
        "sub": email,
        "role": role,
        "ep": token_epoch,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, jwt_secret(), algorithm=ALGORITHM)


@dataclass(frozen=True)
class TokenPayload:
    email: str
    role: str


def verify_token(token: str) -> TokenPayload | None:
    """None on any failure (expired, malformed, wrong signature, superseded) —
    one outcome for the route layer to turn into a 401.

    "Superseded" means the account's `token_epoch` has moved on since this
    token was issued (a password/role change or a disable), so the token is
    rejected even though its signature is still good. The role on the returned
    payload is the token's own claim — safe now, because a role change bumps
    the epoch and this same check then rejects the stale-role token."""
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    email, role = payload["sub"], payload["role"]
    try:
        current_epoch = users.current_token_epoch(email)
    except psycopg.Error as exc:
        log.warning("token epoch check skipped (postgres unreachable?): %s", exc)
        return TokenPayload(email=email, role=role)
    if current_epoch is None or current_epoch != payload.get("ep", 0):
        return None
    return TokenPayload(email=email, role=role)
