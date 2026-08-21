"""Real-Postgres assertions for query history (SRS 3.5.2), same spirit as
tests/data/test_writer.py's integration section: `ensure_table`/`record`/
`list_for_user` exercised against a live database instead of monkeypatched.
"""

from __future__ import annotations

import psycopg
import pytest

from ceynex.api import history
from ceynex.settings import postgres_dsn

TEST_USER = "pytest-history@ceynex.dev"


@pytest.fixture
def clean_user():
    """Remove this test's rows before and after, leaving real data alone."""

    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM query_history WHERE user_email = %s", (TEST_USER,))
            conn.commit()

    history.ensure_table()
    purge()
    yield
    purge()


@pytest.mark.integration
def test_a_recorded_query_is_listed_back(clean_user):
    history.record(
        user_email=TEST_USER,
        query="how are apparel exports doing",
        answer="Exports grew.",
        confidence=0.8,
        degraded=False,
    )

    entries = history.list_for_user(TEST_USER)

    assert len(entries) == 1
    assert entries[0].query == "how are apparel exports doing"
    assert entries[0].answer == "Exports grew."
    assert entries[0].confidence == pytest.approx(0.8)
    assert entries[0].degraded is False
    assert entries[0].asked_at


@pytest.mark.integration
def test_history_is_scoped_to_the_user_and_most_recent_first(clean_user):
    other_user = "someone-else@ceynex.dev"
    try:
        history.record(user_email=TEST_USER, query="first", answer="a", confidence=0.5, degraded=False)
        history.record(user_email=TEST_USER, query="second", answer="b", confidence=0.6, degraded=False)
        history.record(user_email=other_user, query="not mine", answer="c", confidence=0.7, degraded=False)

        entries = history.list_for_user(TEST_USER)

        assert [e.query for e in entries] == ["second", "first"]
    finally:
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM query_history WHERE user_email = %s", (other_user,))
            conn.commit()


@pytest.mark.integration
def test_list_for_user_respects_the_limit(clean_user):
    for i in range(5):
        history.record(user_email=TEST_USER, query=f"q{i}", answer="a", confidence=0.5, degraded=False)

    assert len(history.list_for_user(TEST_USER, limit=3)) == 3


@pytest.mark.integration
def test_recording_against_an_unreachable_database_does_not_raise(monkeypatch):
    """A history-write failure must never surface as a failed query — see
    ceynex/api/history.py's docstring. Points `record()` at a closed port
    (not a fixture swap) so this exercises the real psycopg.Error path."""
    monkeypatch.setattr(history, "postgres_dsn", lambda: "postgresql://ceynex:x@127.0.0.1:1/ceynex")

    history.record(user_email=TEST_USER, query="q", answer="a", confidence=0.5, degraded=False)
