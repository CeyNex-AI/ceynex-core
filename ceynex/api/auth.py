"""Login and token verification — SRS 3.1.11, deferred until now (see docs/DEFERRED.md).

Fixed demo accounts, matching `ceynex-web/src/lib/roles.ts` exactly, rather than
a real user database. There is no signup flow (SRS 3.1.11 covers login/logout,
not account creation) and never has been — the frontend already committed to
four fixed emails, one per role, so a per-account password would gate nothing
that the fixed email list doesn't already gate. What was actually missing was a
server-side credential check and a token the API can verify; that's what this
module adds.

`DEMO_PASSWORD` is a class-project demo credential, printed on the login page
itself, not a secret. Swap `_USERS` for a real table the day this needs to be
more than a demo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import bcrypt
import jwt

from ceynex.settings import jwt_secret

ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = 60 * 60 * 8  # 8h — long enough for one working session

DEMO_PASSWORD = "ceynex-demo"

_ROLE_BY_EMAIL = {
    "policymaker@ceynex.dev": "policymaker",
    "admin@ceynex.dev": "admin",
    "researcher@ceynex.dev": "researcher",
    "exporter@ceynex.dev": "exporter",
}


@dataclass(frozen=True)
class DemoUser:
    email: str
    role: str
    password_hash: bytes


# Hashed once at import time — four accounts, not a per-request cost.
_USERS: dict[str, DemoUser] = {
    email: DemoUser(
        email=email,
        role=role,
        password_hash=bcrypt.hashpw(DEMO_PASSWORD.encode(), bcrypt.gensalt()),
    )
    for email, role in _ROLE_BY_EMAIL.items()
}


def role_for_email(email: str) -> str | None:
    """The role a fixed demo account maps to, or None if `email` isn't one of
    the four. Exposed for `api_keys.py` to build a `TokenPayload` for a key
    without duplicating `_ROLE_BY_EMAIL`, and so a key always authenticates at
    its account's *current* role rather than one frozen at key-creation time."""
    return _ROLE_BY_EMAIL.get(email.strip().lower())


def authenticate(email: str, password: str) -> DemoUser | None:
    """None on any failure — never distinguish "no such user" from "wrong
    password" to a caller, which would let someone enumerate valid emails."""
    user = _USERS.get(email.strip().lower())
    if user is None:
        return None
    if not bcrypt.checkpw(password.encode(), user.password_hash):
        return None
    return user


def issue_token(user: DemoUser) -> str:
    now = int(time.time())
    payload = {"sub": user.email, "role": user.role, "iat": now, "exp": now + TOKEN_TTL_SECONDS}
    return jwt.encode(payload, jwt_secret(), algorithm=ALGORITHM)


@dataclass(frozen=True)
class TokenPayload:
    email: str
    role: str


def verify_token(token: str) -> TokenPayload | None:
    """None on any failure (expired, malformed, wrong signature) — one outcome
    for the route layer to turn into a 401, no exception type to keep in sync."""
    try:
        payload = jwt.decode(token, jwt_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return TokenPayload(email=payload["sub"], role=payload["role"])
