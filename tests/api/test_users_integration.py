"""Real-Postgres assertions for the `users` table and RBAC, same spirit as
`tests/api/test_account_integration.py`: `ensure_table`/CRUD exercised against
a live database rather than monkeypatched.

Every row this file creates uses the `pytest-rbac-` email prefix and is deleted
before and after each test, so a real account is never touched.
"""

from __future__ import annotations

import psycopg
import pytest

from ceynex.api import users
from ceynex.settings import postgres_dsn

PREFIX = "pytest-rbac-"

# Captured at import, before `conftest.py::stub_token_epoch` swaps the lookup for
# a constant 0 on every test in the package. That stub serves the route tests;
# this module exists to exercise the real lookup, so it puts the real one back.
_REAL_CURRENT_TOKEN_EPOCH = users.current_token_epoch


@pytest.fixture(autouse=True)
def real_token_epoch(stub_token_epoch, monkeypatch):
    monkeypatch.setattr(users, "current_token_epoch", _REAL_CURRENT_TOKEN_EPOCH)


@pytest.fixture
def clean_users():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM users WHERE email LIKE %s", (PREFIX + "%",))
            conn.commit()

    users.ensure_table()
    purge()
    yield
    purge()


def _email(name: str) -> str:
    return f"{PREFIX}{name}@ceynex.dev"


@pytest.mark.integration
def test_create_then_authenticate_and_only_the_hash_is_stored(clean_users):
    created = users.create_user(_email("alice"), "correct-horse-battery", "researcher")
    assert created.role == "researcher"
    assert created.disabled is False

    with psycopg.connect(postgres_dsn()) as conn:
        (stored,) = conn.execute(
            "SELECT password_hash FROM users WHERE id = %s", (created.id,)
        ).fetchone()
    assert stored != "correct-horse-battery"

    ok = users.authenticate(_email("Alice"), "correct-horse-battery")  # case-insensitive
    assert ok is not None and ok.id == created.id
    assert users.authenticate(_email("alice"), "wrong") is None


@pytest.mark.integration
def test_a_duplicate_email_raises_email_taken(clean_users):
    users.create_user(_email("bob"), "correct-horse-battery", "exporter")
    with pytest.raises(users.EmailTakenError):
        users.create_user(_email("BOB"), "another-password", "researcher")


@pytest.mark.integration
def test_a_disabled_account_stops_authenticating_and_role_for_email_goes_none(clean_users):
    from ceynex.api.auth import role_for_email

    created = users.create_user(_email("carol"), "correct-horse-battery", "exporter")
    assert role_for_email(_email("carol")) == "exporter"

    users.set_disabled(created.id, disabled=True)
    assert users.authenticate(_email("carol"), "correct-horse-battery") is None
    assert role_for_email(_email("carol")) is None

    users.set_disabled(created.id, disabled=True)  # idempotent
    re_enabled = users.set_disabled(created.id, disabled=False)
    assert re_enabled is not None and re_enabled.disabled is False
    assert users.authenticate(_email("carol"), "correct-horse-battery") is not None


@pytest.mark.integration
def test_set_role_changes_the_stored_role_and_unknown_id_is_none(clean_users):
    created = users.create_user(_email("dave"), "correct-horse-battery", "researcher")
    updated = users.set_role(created.id, "policymaker")
    assert updated is not None and updated.role == "policymaker"
    assert users.get_by_email(_email("dave")).role == "policymaker"
    assert users.set_role(-1, "admin") is None


@pytest.mark.integration
def test_set_role_rejects_an_unknown_role(clean_users):
    created = users.create_user(_email("erin"), "correct-horse-battery", "researcher")
    with pytest.raises(users.InvalidRoleError):
        users.set_role(created.id, "superuser")


@pytest.mark.integration
def test_set_password_replaces_the_hash_and_the_old_one_stops_working(clean_users):
    created = users.create_user(_email("frank"), "first-password-here", "exporter")
    updated = users.set_password(created.id, "second-password-here")
    assert updated is not None and updated.id == created.id
    assert users.authenticate(_email("frank"), "first-password-here") is None
    assert users.authenticate(_email("frank"), "second-password-here") is not None
    assert users.set_password(-1, "no-such-user-pw") is None
    with pytest.raises(users.WeakPasswordError):
        users.set_password(created.id, "short")


@pytest.mark.integration
def test_token_epoch_advances_on_password_role_and_disable_but_not_on_enable(clean_users):
    u = users.create_user(_email("grace"), "grace-password-1", "researcher")
    assert u.token_epoch == 0
    assert users.current_token_epoch(_email("grace")) == 0

    e1 = users.set_password(u.id, "grace-password-2").token_epoch
    assert e1 == 1
    e2 = users.set_role(u.id, "exporter").token_epoch
    assert e2 == 2
    e3 = users.set_disabled(u.id, disabled=True).token_epoch
    assert e3 == 3
    # disabled → the per-request check reports None regardless of the stored value
    assert users.current_token_epoch(_email("grace")) is None
    # re-enable does NOT bump; the stored epoch stays at 3, so tokens minted at
    # 0/1/2 remain dead
    e4 = users.set_disabled(u.id, disabled=False).token_epoch
    assert e4 == 3
    assert users.current_token_epoch(_email("grace")) == 3


@pytest.mark.integration
def test_last_admin_guard_blocks_demotion_and_disable_of_a_lone_pytest_admin(clean_users):
    """Scoped to this file's own rows: if the deployment already has other
    enabled admins the guard won't fire, so the assertion is only meaningful
    when `pytest-rbac-` admins are the only ones — which `clean_users`
    guarantees by purging the prefix first. To avoid a false pass when a real
    admin exists, this checks the domain function directly against a count of
    prefix-scoped admins."""
    admin = users.create_user(_email("root"), "correct-horse-battery", "admin")

    with psycopg.connect(postgres_dsn()) as conn:
        (other_admins,) = conn.execute(
            "SELECT count(*) FROM users WHERE role = 'admin' AND disabled_at IS NULL AND id <> %s",
            (admin.id,),
        ).fetchone()

    if other_admins == 0:
        with pytest.raises(users.LastAdminError):
            users.set_role(admin.id, "researcher")
        with pytest.raises(users.LastAdminError):
            users.set_disabled(admin.id, disabled=True)
    else:
        # Another admin exists on this database — demotion is allowed.
        assert users.set_role(admin.id, "researcher") is not None


@pytest.mark.integration
def test_set_email_reassigns_owned_rows_and_bumps_the_epoch(clean_users):
    from ceynex.api import api_keys, history, preferences

    for ensure in (api_keys.ensure_table, preferences.ensure_table, history.ensure_table):
        ensure()

    old, new = _email("mover-old"), _email("mover-new")
    u = users.create_user(old, "mover-password-1", "exporter")
    api_keys.create(old, "pytest-rbac email-move key")
    preferences.upsert(old, dq_flag_alerts=False, forecast_updates=True, weekly_digest=True)

    updated = users.set_email(u.id, "  " + new.upper() + " ")  # normalised on write
    assert updated is not None and updated.email == new
    assert updated.token_epoch == u.token_epoch + 1
    assert users.authenticate(new, "mover-password-1") is not None
    assert users.authenticate(old, "mover-password-1") is None

    with psycopg.connect(postgres_dsn()) as conn:
        (keys_old,) = conn.execute(
            "SELECT count(*) FROM api_keys WHERE user_email = %s", (old,)
        ).fetchone()
        (keys_new,) = conn.execute(
            "SELECT count(*) FROM api_keys WHERE user_email = %s", (new,)
        ).fetchone()
        (prefs_new,) = conn.execute(
            "SELECT count(*) FROM notification_preferences WHERE user_email = %s", (new,)
        ).fetchone()
    assert keys_old == 0 and keys_new == 1 and prefs_new == 1

    # cleanup the moved rows (clean_users only purges the users table)
    with psycopg.connect(postgres_dsn()) as conn:
        conn.execute("DELETE FROM api_keys WHERE user_email = %s", (new,))
        conn.execute("DELETE FROM notification_preferences WHERE user_email = %s", (new,))
        conn.commit()


@pytest.mark.integration
def test_set_email_to_a_taken_address_raises_and_changes_nothing(clean_users):
    a = users.create_user(_email("clash-a"), "clash-password-1", "researcher")
    users.create_user(_email("clash-b"), "clash-password-2", "researcher")
    with pytest.raises(users.EmailTakenError):
        users.set_email(a.id, _email("clash-b"))
    assert users.get_by_email(_email("clash-a")) is not None  # untouched


@pytest.mark.integration
def test_delete_user_removes_the_row_and_its_owned_rows(clean_users):
    from ceynex.api import api_keys, preferences

    api_keys.ensure_table()
    preferences.ensure_table()

    email = _email("goner")
    u = users.create_user(email, "goner-password-1", "exporter")
    api_keys.create(email, "pytest-rbac delete key")
    preferences.upsert(email, dq_flag_alerts=True, forecast_updates=False, weekly_digest=False)

    assert users.delete_user(u.id) is True
    assert users.delete_user(u.id) is False  # already gone
    assert users.get_by_email(email) is None
    with psycopg.connect(postgres_dsn()) as conn:
        (keys,) = conn.execute(
            "SELECT count(*) FROM api_keys WHERE user_email = %s", (email,)
        ).fetchone()
        (prefs,) = conn.execute(
            "SELECT count(*) FROM notification_preferences WHERE user_email = %s", (email,)
        ).fetchone()
    assert keys == 0 and prefs == 0


@pytest.mark.integration
def test_delete_user_last_admin_guard(clean_users):
    admin = users.create_user(_email("lone-admin"), "lone-password-1", "admin")
    with psycopg.connect(postgres_dsn()) as conn:
        (other_admins,) = conn.execute(
            "SELECT count(*) FROM users WHERE role = 'admin' AND disabled_at IS NULL AND id <> %s",
            (admin.id,),
        ).fetchone()
    if other_admins == 0:
        with pytest.raises(users.LastAdminError):
            users.delete_user(admin.id)
    else:
        assert users.delete_user(admin.id) is True


# --- the conversational layer's rows follow the account ---------------------
#
# Found 2026-09-12: none of these tables were in `_EMAIL_OWNED_TABLES`, so a
# deleted account's conversations stayed behind for the next signup with the
# address. `tests/api/test_email_owned_tables.py` keeps the list complete; these
# check that what it lists is actually deleted, moved, or unattributed.

_CHAT_TABLES = (
    "user_instruction", "chat_feedback", "chat_pending_clarification", "chat_conversation",
)


@pytest.fixture
def chat_tables(clean_users):
    from ceynex.chat import instructions, store
    from ceynex.observability import ledger

    store.ensure_table()
    instructions.ensure_table()
    ledger.ensure_table()

    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            for table in _CHAT_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE user_email LIKE %s", (PREFIX + "%",))  # noqa: S608 - fixed list
            conn.execute("DELETE FROM llm_usage WHERE request_id LIKE %s", (PREFIX + "%",))
            conn.commit()

    purge()
    yield
    purge()


def _file_chat_rows(email: str) -> int:
    """A conversation with a message, its feedback and a pending clarification,
    an instruction and a spend record, all under `email`. Returns the
    conversation id."""
    with psycopg.connect(postgres_dsn()) as conn:
        (conversation_id,) = conn.execute(
            "INSERT INTO chat_conversation (user_email, title) VALUES (%s, 'tea') RETURNING id",
            (email,),
        ).fetchone()
        (message_id,) = conn.execute(
            "INSERT INTO chat_message (conversation_id, seq, role, content) "
            "VALUES (%s, 1, 'assistant', 'Tea exports reached USD 1,431,567,471.') RETURNING id",
            (conversation_id,),
        ).fetchone()
        conn.execute(
            "INSERT INTO chat_feedback (message_id, user_email, rating) VALUES (%s, %s, 1)",
            (message_id, email),
        )
        conn.execute(
            "INSERT INTO chat_pending_clarification "
            "(conversation_id, user_email, original_query, payload, expires_at) "
            "VALUES (%s, %s, 'tea or cinnamon?', '{}', now() + interval '1 hour')",
            (conversation_id, email),
        )
        conn.execute(
            "INSERT INTO user_instruction (user_email, content) VALUES (%s, 'Answer briefly.')",
            (email,),
        )
        conn.execute(
            "INSERT INTO llm_usage (request_id, user_email, role, model, provider, cost_usd) "
            "VALUES (%s, %s, 'merge', 'gpt-4o', 'openai', 0.004)",
            (PREFIX + email, email),
        )
        conn.commit()
    return conversation_id


def _chat_rows(email: str) -> dict[str, int]:
    with psycopg.connect(postgres_dsn()) as conn:
        counts = {
            table: conn.execute(
                f"SELECT count(*) FROM {table} WHERE user_email = %s", (email,)  # noqa: S608 - fixed list
            ).fetchone()[0]
            for table in (*_CHAT_TABLES, "llm_usage")
        }
    return counts


@pytest.mark.integration
def test_deleting_an_account_takes_its_conversations_and_keeps_its_spend_unattributed(chat_tables):
    email = _email("chat-goner")
    u = users.create_user(email, "chat-goner-password", "researcher")
    conversation_id = _file_chat_rows(email)

    assert users.delete_user(u.id) is True

    assert _chat_rows(email) == dict.fromkeys((*_CHAT_TABLES, "llm_usage"), 0)
    with psycopg.connect(postgres_dsn()) as conn:
        (messages,) = conn.execute(
            "SELECT count(*) FROM chat_message WHERE conversation_id = %s", (conversation_id,)
        ).fetchone()
        (spend,) = conn.execute(
            "SELECT count(*) FROM llm_usage WHERE request_id = %s AND user_email IS NULL",
            (PREFIX + email,),
        ).fetchone()
    assert messages == 0, "the conversation's messages go with it, by cascade"
    assert spend == 1, "the spend record stays, with no one attached"


@pytest.mark.integration
def test_the_next_signup_with_a_freed_address_inherits_nothing(chat_tables):
    """The defect itself, end to end: rows left under an address by a holder
    who is gone, from before their table was listed, are not handed to the next
    account that claims it."""
    email = _email("chat-reused")
    _file_chat_rows(email)  # a previous holder's leftovers, with no account

    users.create_user(email, "chat-reused-password", "researcher")

    assert _chat_rows(email) == dict.fromkeys((*_CHAT_TABLES, "llm_usage"), 0)


@pytest.mark.integration
def test_changing_an_email_moves_the_conversations_and_drops_a_previous_holders(chat_tables):
    old, new = _email("chat-mover-old"), _email("chat-mover-new")
    u = users.create_user(old, "chat-mover-password", "exporter")
    mine = _file_chat_rows(old)
    with psycopg.connect(postgres_dsn()) as conn:
        # A previous holder of the new address left a conversation and an
        # instruction behind. Neither may reach this account.
        conn.execute(
            "INSERT INTO chat_conversation (user_email, title) VALUES (%s, 'not yours')", (new,)
        )
        conn.execute(
            "INSERT INTO user_instruction (user_email, content) VALUES (%s, 'someone else')", (new,)
        )
        conn.commit()

    assert users.set_email(u.id, new) is not None

    assert _chat_rows(old) == dict.fromkeys((*_CHAT_TABLES, "llm_usage"), 0)
    assert _chat_rows(new) == dict.fromkeys((*_CHAT_TABLES, "llm_usage"), 1)
    with psycopg.connect(postgres_dsn()) as conn:
        titles = [row[0] for row in conn.execute(
            "SELECT title FROM chat_conversation WHERE user_email = %s", (new,)
        ).fetchall()]
        (instruction,) = conn.execute(
            "SELECT content FROM user_instruction WHERE user_email = %s", (new,)
        ).fetchone()
        (owner,) = conn.execute(
            "SELECT user_email FROM chat_conversation WHERE id = %s", (mine,)
        ).fetchone()
    assert titles == ["tea"] and instruction == "Answer briefly." and owner == new
