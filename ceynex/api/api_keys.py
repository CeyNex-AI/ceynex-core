"""API keys for programmatic access — Account.tsx's "API keys for programmatic
access" stub, now built. A second, longer-lived way to authenticate
`/api/query` and everything else `require_user` gates, alongside the
short-lived JWT issued at login (see `routes/auth.py`'s `require_user`, which
tries a `ck_`-prefixed bearer token here before falling back to `verify_token`).

Same persistence shape as `ceynex/api/history.py`: idempotent DDL in
`ensure_table()`, opportunistic on the `last_used_at` touch (a key that still
authenticates even if that update fails matters more than the timestamp being
perfectly current), strict everywhere else.

A key authenticates as whichever of the four fixed demo accounts created it,
at that account's *current* role (`auth.role_for_email`) rather than one
frozen at creation time, so a key never outlives a role change to the account
that made it.

The plaintext key is only ever returned once, from `create()`, right after
generation. Only its SHA-256 hash is stored; `authenticate()` hashes the
presented key and looks it up, the same one-way pattern as a password hash —
appropriate here because the key itself is high-entropy random text rather
than something a person chose, so slow hashing (bcrypt) buys nothing.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass

import psycopg

from ceynex.api.auth import TokenPayload, role_for_email
from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

KEY_PREFIX = "ck_"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id BIGSERIAL PRIMARY KEY,
    user_email TEXT NOT NULL,
    label TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS api_keys_user_idx ON api_keys (user_email);
"""


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("api_keys table not ensured (postgres unreachable?): %s", exc)


def _hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


@dataclass(frozen=True)
class ApiKeyEntry:
    id: int
    label: str
    key_prefix: str
    created_at: str
    last_used_at: str | None
    revoked: bool


@dataclass(frozen=True)
class NewApiKey:
    id: int
    label: str
    key: str
    key_prefix: str
    created_at: str


def create(user_email: str, label: str) -> NewApiKey:
    raw_key = KEY_PREFIX + secrets.token_urlsafe(32)
    prefix = raw_key[:11]
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO api_keys (user_email, label, key_hash, key_prefix)
            VALUES (%s, %s, %s, %s)
            RETURNING id, created_at
            """,
            (user_email, label, _hash(raw_key), prefix),
        )
        row = cur.fetchone()
        conn.commit()
    return NewApiKey(id=row[0], label=label, key=raw_key, key_prefix=prefix, created_at=row[1].isoformat())


def list_for_user(user_email: str) -> list[ApiKeyEntry]:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, label, key_prefix, created_at, last_used_at, revoked_at
            FROM api_keys
            WHERE user_email = %s
            ORDER BY created_at DESC
            """,
            (user_email,),
        )
        rows = cur.fetchall()
    return [
        ApiKeyEntry(
            id=r[0],
            label=r[1],
            key_prefix=r[2],
            created_at=r[3].isoformat(),
            last_used_at=r[4].isoformat() if r[4] else None,
            revoked=r[5] is not None,
        )
        for r in rows
    ]


def revoke(key_id: int, user_email: str) -> bool:
    """True if a key with this id, owned by this user and not already
    revoked, was revoked — same ownership-in-the-`WHERE` pattern as
    `history.set_saved`, so "not yours" and "doesn't exist" look identical."""
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE api_keys SET revoked_at = now() "
            "WHERE id = %s AND user_email = %s AND revoked_at IS NULL",
            (key_id, user_email),
        )
        updated = cur.rowcount > 0
        conn.commit()
    return updated


def authenticate(raw_key: str) -> TokenPayload | None:
    """None for any failure — unknown key, revoked key, or an account that no
    longer maps to one of the four fixed roles — one outcome for
    `require_user` to turn into the same 401 an invalid JWT gets."""
    if not raw_key.startswith(KEY_PREFIX):
        return None
    key_hash = _hash(raw_key)
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT user_email FROM api_keys WHERE key_hash = %s AND revoked_at IS NULL",
                (key_hash,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            user_email = row[0]
            role = role_for_email(user_email)
            if role is None:
                return None
            cur.execute("UPDATE api_keys SET last_used_at = now() WHERE key_hash = %s", (key_hash,))
            conn.commit()
    except psycopg.Error as exc:
        log.warning("api key lookup failed (postgres unreachable?): %s", exc)
        return None
    return TokenPayload(email=user_email, role=role)
