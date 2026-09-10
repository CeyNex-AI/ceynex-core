"""Real user accounts and role-based access — replaces the four fixed demo
accounts that used to live in `ceynex/api/auth.py`.

The old design hard-coded one account per role with a shared password, on the
argument that a per-account password would gate nothing the fixed email list
didn't already gate. That stops being true the moment anyone other than the
four of us needs a login, so this module adds the real thing: a `users` table,
per-account bcrypt password hashes, self-service signup at the default role,
and admin routes to provision accounts at any role and to change or revoke a
role later.

Persistence shape is the same as `ceynex/api/history.py` /
`ceynex/api/api_keys.py`: idempotent DDL in `ensure_table()` run once at API
startup (`main.py`'s lifespan), `psycopg.connect(postgres_dsn(),
connect_timeout=3)` per call, frozen dataclasses out. Additive to the frozen
`ceynex-contracts` schema — no other member's code reads `users` — so it does
not go through that repo's 3-way-approval gate.

**Bootstrapping.** Dropping the demo accounts means a fresh deployment has zero
users and nobody can sign in to become an admin. `ensure_table()` seeds one
admin from `CEYNEX_BOOTSTRAP_ADMIN` (`email:password`) if that account does not
already exist; `python -m ceynex.api.users create-admin <email> <password>`
does the same thing by hand. Neither ever overwrites an existing row, so
leaving the env var set across restarts is harmless.

**Session invalidation.** `token_epoch` is an integer on each row, bumped by
`set_password`, `set_role`, and `set_disabled(disabled=True)`. The login JWT
carries the value it was issued against; `auth.verify_token` reads
`current_token_epoch` once per authed request and rejects a token whose epoch
no longer matches (or whose account is gone/disabled). So a password change,
role change or disable cuts every existing session for that account at its
next request — not 8 h later. The self-service password route
(`routes/account.py`) hands the caller a fresh token in the same response so
their own device stays signed in while every other session drops. If the
per-request lookup can't reach Postgres the token is accepted on its
signature alone — a datastore blip must not sign everyone out.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import bcrypt
import psycopg

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

# The same four the frontend and `require_admin` already know about. A signup
# picks none of these — it always lands on `DEFAULT_ROLE`; only an admin moves
# an account to another one.
VALID_ROLES: tuple[str, ...] = ("policymaker", "admin", "researcher", "exporter")
DEFAULT_ROLE = "researcher"

MIN_PASSWORD_LENGTH = 8

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    disabled_at TIMESTAMPTZ,
    token_epoch INTEGER NOT NULL DEFAULT 0
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS token_epoch INTEGER NOT NULL DEFAULT 0;
"""

# Every SELECT/RETURNING that feeds `_row_to_user` uses exactly this list, in
# this order — one place to change if a column is added.
_COLS = "id, email, password_hash, role, created_at, disabled_at, token_epoch"


class EmailTakenError(Exception):
    """`create_user` for an email that already has a row."""


class InvalidRoleError(Exception):
    """A role string that is not one of `VALID_ROLES`."""


class WeakPasswordError(Exception):
    """A password shorter than `MIN_PASSWORD_LENGTH`. The route layer also
    enforces this via pydantic; kept here so the domain function is safe to
    call directly (tests, the `create-admin` CLI)."""


class LastAdminError(Exception):
    """A role change or disable that would leave no enabled admin. Refused so a
    deployment can't lock itself out through its own admin UI."""


@dataclass(frozen=True)
class User:
    id: int
    email: str
    role: str
    created_at: str
    disabled: bool
    #: Bumped on every password change, role change, and disable. The login JWT
    #: carries the value it was issued against; `auth.verify_token` rejects a
    #: token whose epoch no longer matches, so those actions cut existing
    #: sessions instead of waiting out the 8 h token TTL.
    token_epoch: int


def _norm_email(email: str) -> str:
    return email.strip().lower()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        # A malformed hash in the row — treat as "does not match" rather than
        # 500 the request.
        return False


def _row_to_user(row: tuple) -> User:
    return User(
        id=row[0],
        email=row[1],
        role=row[3],
        created_at=row[4].isoformat(),
        disabled=row[5] is not None,
        token_epoch=row[6],
    )


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("users table not ensured (postgres unreachable?): %s", exc)
        return
    _seed_bootstrap_admin()


def _seed_bootstrap_admin() -> None:
    """Insert the `CEYNEX_BOOTSTRAP_ADMIN` account if it is set and not already
    present. Best-effort: a failure here is logged, not raised — it must not
    stop the API coming up."""
    raw = os.environ.get("CEYNEX_BOOTSTRAP_ADMIN", "").strip()
    if not raw or ":" not in raw:
        return
    email, password = raw.split(":", 1)
    try:
        created = create_user(email, password, "admin")
    except EmailTakenError:
        return
    except (InvalidRoleError, WeakPasswordError) as exc:
        log.warning("CEYNEX_BOOTSTRAP_ADMIN ignored: %s", exc)
        return
    except psycopg.Error as exc:
        log.warning("could not seed bootstrap admin (postgres unreachable?): %s", exc)
        return
    log.info("seeded bootstrap admin account %s", created.email)


def create_user(email: str, password: str, role: str) -> User:
    """Raises `InvalidRoleError` or `WeakPasswordError` before touching the
    database, and translates a unique-violation into `EmailTakenError`.
    `psycopg.Error` propagates for the route layer to turn into a 503."""
    if role not in VALID_ROLES:
        raise InvalidRoleError(f"role must be one of {VALID_ROLES}, got {role!r}")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPasswordError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    email = _norm_email(email)
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        try:
            cur.execute(
                f"""
                INSERT INTO users (email, password_hash, role)
                VALUES (%s, %s, %s)
                RETURNING {_COLS}
                """,  # noqa: S608 - _COLS is a fixed module constant, no input
                (email, _hash_password(password), role),
            )
        except psycopg.errors.UniqueViolation as exc:
            raise EmailTakenError(f"an account already exists for {email}") from exc
        row = cur.fetchone()
        conn.commit()
    return _row_to_user(row)


def get_by_email(email: str) -> User | None:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM users WHERE email = %s",  # noqa: S608 - _COLS is a fixed constant
            (_norm_email(email),),
        )
        row = cur.fetchone()
    return _row_to_user(row) if row else None


def current_token_epoch(email: str) -> int | None:
    """The `token_epoch` an enabled account's tokens must currently match, or
    None if there is no such account or it is disabled. `auth.verify_token`
    calls this once per authed request — a single indexed lookup, the same
    per-request DB cost `api_keys.authenticate` already pays for key auth."""
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT token_epoch FROM users WHERE email = %s AND disabled_at IS NULL",
            (_norm_email(email),),
        )
        row = cur.fetchone()
    return row[0] if row else None


def authenticate(email: str, password: str) -> User | None:
    """None on any failure — unknown email, wrong password, or a disabled
    account — never distinguishing them, the same single-outcome rule the old
    `auth.authenticate` had, so a caller can't enumerate which emails are real
    or which accounts are disabled."""
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM users WHERE email = %s",  # noqa: S608 - _COLS is a fixed constant
            (_norm_email(email),),
        )
        row = cur.fetchone()
    if row is None:
        return None
    if row[5] is not None:  # disabled_at
        return None
    if not _verify_password(password, row[2]):
        return None
    return _row_to_user(row)


@dataclass(frozen=True)
class UserSummary:
    id: int
    email: str
    role: str
    created_at: str
    disabled: bool


def list_users() -> list[UserSummary]:
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, email, role, created_at, disabled_at FROM users ORDER BY created_at, id"
        )
        rows = cur.fetchall()
    return [
        UserSummary(
            id=r[0],
            email=r[1],
            role=r[2],
            created_at=r[3].isoformat(),
            disabled=r[4] is not None,
        )
        for r in rows
    ]


def _enabled_admin_count(cur: psycopg.Cursor, *, excluding_id: int | None = None) -> int:
    cur.execute(
        "SELECT count(*) FROM users "
        "WHERE role = 'admin' AND disabled_at IS NULL AND id <> %s",
        (excluding_id if excluding_id is not None else -1,),
    )
    return cur.fetchone()[0]


def set_role(user_id: int, role: str) -> User | None:
    """New `User` on success, None if `user_id` is unknown. Raises
    `InvalidRoleError` for a bad role and `LastAdminError` if demoting this
    account would leave no enabled admin."""
    if role not in VALID_ROLES:
        raise InvalidRoleError(f"role must be one of {VALID_ROLES}, got {role!r}")
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM users WHERE id = %s",  # noqa: S608 - _COLS is a fixed constant
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        current = _row_to_user(row)
        if current.role == "admin" and role != "admin" and _enabled_admin_count(
            cur, excluding_id=user_id
        ) == 0:
            raise LastAdminError("cannot remove the last enabled admin")
        # Bump token_epoch so any session already signed in as this user is cut
        # at its next request — a role change has to take effect now, not in 8 h.
        cur.execute(
            f"UPDATE users SET role = %s, token_epoch = token_epoch + 1 "
            f"WHERE id = %s RETURNING {_COLS}",  # noqa: S608 - _COLS is a fixed constant
            (role, user_id),
        )
        updated = cur.fetchone()
        conn.commit()
    return _row_to_user(updated)


def set_disabled(user_id: int, *, disabled: bool) -> User | None:
    """New `User` on success, None if `user_id` is unknown. Raises
    `LastAdminError` if disabling this account would leave no enabled admin.
    Re-enabling is always allowed."""
    # `disabled_at` is set to now() or cleared to NULL; the two cases are
    # separate literal statements rather than a bound parameter because now()
    # has to be evaluated by the database, not passed as a value.
    new_disabled_at_sql = "now()" if disabled else "NULL"
    # Disabling also bumps token_epoch so existing sessions die immediately (not
    # just at the next request, which `current_token_epoch` already covers) and,
    # more importantly, so a later re-enable does not resurrect them. Re-enable
    # does not bump — a fresh login is required either way.
    epoch_bump_sql = ", token_epoch = token_epoch + 1" if disabled else ""
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_COLS} FROM users WHERE id = %s",  # noqa: S608 - _COLS is a fixed constant
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        current = _row_to_user(row)
        if (
            disabled
            and current.role == "admin"
            and _enabled_admin_count(cur, excluding_id=user_id) == 0
        ):
            raise LastAdminError("cannot disable the last enabled admin")
        cur.execute(
            f"UPDATE users SET disabled_at = {new_disabled_at_sql}{epoch_bump_sql} "  # noqa: S608 - fixed literals, no input
            f"WHERE id = %s RETURNING {_COLS}",
            (user_id,),
        )
        updated = cur.fetchone()
        conn.commit()
    return _row_to_user(updated)


def set_password(user_id: int, new_password: str) -> User | None:
    """Replace the stored bcrypt hash. New `User` on success, None if `user_id`
    is unknown. Raises `WeakPasswordError` for a password under
    `MIN_PASSWORD_LENGTH`.

    Does not check the *old* password — that is the caller's job (the
    self-service route checks it via `authenticate`; the admin reset route
    deliberately does not). Bumps `token_epoch`, so every existing session for
    this account — including the one that made the change — is cut at its next
    request; the self-service route hands the caller a fresh token in the same
    response so their own device stays signed in."""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        raise WeakPasswordError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE users SET password_hash = %s, token_epoch = token_epoch + 1 "
            f"WHERE id = %s RETURNING {_COLS}",  # noqa: S608 - _COLS is a fixed constant
            (_hash_password(new_password), user_id),
        )
        updated = cur.fetchone()
        conn.commit()
    return _row_to_user(updated) if updated else None


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m ceynex.api.users")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ca = sub.add_parser("create-admin", help="create (or report) an admin account")
    ca.add_argument("email")
    ca.add_argument("password")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_table()
    if args.cmd == "create-admin":
        try:
            user = create_user(args.email, args.password, "admin")
        except EmailTakenError:
            print(f"an account already exists for {args.email!r}; no change made")
            return 1
        except (InvalidRoleError, WeakPasswordError) as exc:
            print(f"error: {exc}")
            return 2
        print(f"created admin account {user.email} (id {user.id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
