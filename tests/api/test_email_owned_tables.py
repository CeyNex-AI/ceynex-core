"""Every table that keys rows to a user by email is known to the account
lifecycle, so deleting or re-addressing an account takes its rows along.

Found 2026-09-12: the conversational layer's tables were never listed in
`users._EMAIL_OWNED_TABLES`. A deleted account's conversations and instructions
stayed behind, and whoever next signed up with the address inherited them. The
tables and the list lived in different modules, written in parallel, and
nothing connected them. This does: it reads every `CREATE TABLE` in the package
and fails on a table with a `user_email` column that the list does not name.
"""

from __future__ import annotations

import re
from pathlib import Path

from ceynex.api import users

PACKAGE = Path(users.__file__).resolve().parents[1]
TABLE = re.compile(r"CREATE TABLE IF NOT EXISTS (\w+) \((.*?)\n\);", re.S)
EMAIL_COLUMN = re.compile(r"^\s*user_email\s", re.M)


def _email_keyed_tables() -> set[str]:
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        for name, body in TABLE.findall(path.read_text()):
            if EMAIL_COLUMN.search(body):
                found.add(name)
    return found


def _registered() -> set[str]:
    return set(users._EMAIL_OWNED_TABLES) | set(users._EMAIL_ATTRIBUTED_TABLES)


def test_the_scan_finds_the_tables_it_exists_to_guard():
    """Otherwise the next test would pass by finding nothing."""
    assert {"query_history", "chat_conversation", "user_instruction", "llm_usage"} <= (
        _email_keyed_tables()
    )


def test_every_email_keyed_table_is_known_to_the_account_lifecycle():
    missing = _email_keyed_tables() - _registered()
    assert not missing, (
        f"deleting an account would leave these behind, for the next holder of "
        f"the address to inherit: {sorted(missing)}"
    )


def test_every_listed_table_is_one_that_keys_rows_by_email():
    assert _registered() <= _email_keyed_tables()


def test_a_table_is_either_owned_or_attributed_never_both():
    assert not set(users._EMAIL_OWNED_TABLES) & set(users._EMAIL_ATTRIBUTED_TABLES)
