"""Credential check and token issue/verify — SRS 3.1.11.

Backed by the `users` table (`ceynex/api/users.py`) since RBAC landed. This
module used to hold four fixed demo accounts with a shared password; that was
always flagged as "swap for a real table the day this needs to be more than a
demo" and this is that day.

`verify_token` now makes one indexed DB lookup per authed request
(`users.current_token_epoch`) so that a password change, role change or
disable cuts existing sessions at their next request instead of waiting out
the 8 h token TTL.

**If that lookup fails** (Postgres unreachable), a datastore blip must not log
everyone out, so a token is still accepted on its signature, with one
exception: each worker remembers the last epoch it read for every account it
has checked, and a token *that* rules out is refused. A session cut before the
outage and checked since stays cut. What remains exposed is an account this
worker has not checked since its epoch moved: same "fail open, but never weaker
than before" line as the spend cap (ARCHITECTURE_DELTA.md D16).
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


#: The last epoch this worker read for each account; None for an account that
#: is gone or disabled. Consulted only when the database cannot be reached.
#: Bounded by the number of accounts, which is small; cleared outright if that
#: ever stops being true rather than growing without limit.
_known_epochs: dict[str, int | None] = {}
_KNOWN_EPOCHS_MAX = 10_000


def _remember_epoch(email: str, epoch: int | None) -> None:
    if len(_known_epochs) >= _KNOWN_EPOCHS_MAX and email not in _known_epochs:
        _known_epochs.clear()
    _known_epochs[email] = epoch


def forget_known_epochs() -> None:
    """Test seam."""
    _known_epochs.clear()


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
    token_epoch = payload.get("ep", 0)
    try:
        current_epoch = users.current_token_epoch(email)
    except psycopg.Error as exc:
        if email in _known_epochs and _known_epochs[email] != token_epoch:
            log.warning("epoch check unreachable; refusing a token this worker saw superseded")
            return None
        log.warning("token epoch check skipped (postgres unreachable?): %s", exc)
        return TokenPayload(email=email, role=role)
    _remember_epoch(email, current_epoch)
    if current_epoch is None or current_epoch != token_epoch:
        return None
    return TokenPayload(email=email, role=role)
