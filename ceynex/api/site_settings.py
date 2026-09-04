"""Site-wide settings that a single admin action changes for every visitor.

Today just the UI theme -- not an SRS-named requirement, a real product need
surfaced by the frontend's 2026-09-04 redesign, which shipped as an optional
theme rather than forced on every user. Additive to the frozen
`ceynex-contracts` schema, same reasoning as `history.py`/`preferences.py`: a
`ceynex-core`-only concern with no business behind that repo's 3-way-approval
gate.

Unlike `preferences.py` (one row per user, private to them), this is a single
shared row: every visitor's frontend reads the same value on page load --
anonymous or signed in, it's public, deliberately not behind `require_user` --
and only the admin role can change it (enforced at the route layer, see
`routes/site.py`). `ensure_table()` runs once at API startup, same
idempotent-DDL spirit as every other `ensure_table()` in this package.

The read degrades to `DEFAULT_THEME` on any error (a DB hiccup must not break
page load for every single visitor); the write does not -- an admin action
that silently no-ops would be worse than one that fails loudly.
"""

from __future__ import annotations

import logging

import psycopg

from ceynex.settings import postgres_dsn

log = logging.getLogger(__name__)

VALID_THEMES = ("classic", "signal-deck")
DEFAULT_THEME = "classic"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS site_settings (
    id BOOLEAN PRIMARY KEY DEFAULT true,
    theme TEXT NOT NULL DEFAULT 'classic',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by TEXT,
    CONSTRAINT site_settings_singleton CHECK (id)
);
"""


def ensure_table() -> None:
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            conn.commit()
    except psycopg.Error as exc:
        log.warning("site_settings table not ensured (postgres unreachable?): %s", exc)


def get_theme() -> str:
    """The current site theme. Degrades to `DEFAULT_THEME` if unset or the
    database is unreachable, rather than raising -- this is called on every
    page load, by every visitor, and a read failure here must not take the
    whole site down with it."""
    try:
        with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT theme FROM site_settings WHERE id = true")
            row = cur.fetchone()
            return row[0] if row else DEFAULT_THEME
    except psycopg.Error as exc:
        log.warning("could not read site theme, defaulting to %r: %s", DEFAULT_THEME, exc)
        return DEFAULT_THEME


def set_theme(theme: str, *, updated_by: str) -> str:
    """Upserts the singleton row. Raises `ValueError` for an unrecognized
    theme name and lets `psycopg.Error` propagate on a real DB failure --
    unlike `get_theme`, a write should fail loudly: an admin who just changed
    the site theme and got a silent no-op back would have no way to know it
    didn't take."""
    if theme not in VALID_THEMES:
        raise ValueError(f"unknown theme {theme!r}; must be one of {VALID_THEMES}")
    with psycopg.connect(postgres_dsn(), connect_timeout=3) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO site_settings (id, theme, updated_by)
            VALUES (true, %s, %s)
            ON CONFLICT (id) DO UPDATE
                SET theme = EXCLUDED.theme, updated_at = now(), updated_by = EXCLUDED.updated_by
            """,
            (theme, updated_by),
        )
        conn.commit()
    return theme
