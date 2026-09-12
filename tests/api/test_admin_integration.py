"""Real-Postgres assertions for the admin routes' DB-facing helpers
(`ceynex/api/admin.py`): `pipeline_status`, `list_dq_flags`, `resolve_dq_flag`.
Same spirit as `tests/data/test_writer.py` and `test_history_integration.py` —
insert directly via SQL, read back through the real function.

Retrain and ingest aren't covered here: they reuse `ceynex.models.registry`
and `ceynex.data.pipeline`, which already have their own integration coverage
(`tests/models/test_registry.py`, `tests/data/test_writer.py`) — re-proving
the same DB behaviour through the admin routes would just be slower, not more
confident.
"""

from __future__ import annotations

import psycopg
import pytest

from ceynex.api import admin, audit
from ceynex.settings import postgres_dsn

TEST_SOURCE = "PYTEST_ADMIN"
TEST_ACTOR = "pytest-admin@ceynex.dev"


@pytest.fixture
def clean_ingest_runs():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM ingest_run WHERE source_id = %s", (TEST_SOURCE,))
            conn.commit()

    purge()
    yield
    purge()


@pytest.fixture
def clean_dq_flags():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM dq_flag WHERE source_a = %s", (TEST_SOURCE,))
            conn.commit()

    purge()
    yield
    purge()


@pytest.mark.integration
def test_pipeline_status_reads_real_ingest_runs(clean_ingest_runs):
    with psycopg.connect(postgres_dsn()) as conn:
        conn.execute(
            """
            INSERT INTO ingest_run (source_id, status, rows_written, finished_at)
            VALUES (%s, 'success', 42, now())
            """,
            (TEST_SOURCE,),
        )
        conn.commit()

    runs = admin.pipeline_status()

    matching = [r for r in runs if r.source_id == TEST_SOURCE]
    assert len(matching) == 1
    assert matching[0].status == "success"
    assert matching[0].rows_written == 42
    assert matching[0].finished_at is not None


@pytest.mark.integration
def test_dq_flags_lists_unresolved_before_resolved_and_survives_a_resolve(clean_dq_flags):
    with psycopg.connect(postgres_dsn()) as conn:
        cur = conn.execute(
            """
            INSERT INTO dq_flag (item, source_a, source_b, value_a, value_b, pct_diff, severity)
            VALUES (%s, %s, 'JAAF', 100.0, 120.0, 18.2, 'material')
            RETURNING flag_id
            """,
            (TEST_SOURCE + "_item", TEST_SOURCE),
        )
        flag_id = cur.fetchone()[0]
        conn.commit()

    flags = admin.list_dq_flags(severity="material")
    matching = [f for f in flags if f.source_a == TEST_SOURCE]
    assert len(matching) == 1
    assert matching[0].resolved is False
    assert matching[0].value_a == pytest.approx(100.0)
    assert matching[0].pct_diff == pytest.approx(18.2)

    assert admin.resolve_dq_flag(flag_id) is True

    resolved_flags = admin.list_dq_flags(resolved=True, severity="material")
    assert any(f.flag_id == flag_id for f in resolved_flags)


@pytest.mark.integration
def test_resolving_a_nonexistent_flag_returns_false():
    assert admin.resolve_dq_flag(-1) is False


@pytest.fixture
def clean_audit_log():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM audit_log WHERE actor_email = %s", (TEST_ACTOR,))
            conn.commit()

    audit.ensure_table()
    purge()
    yield
    purge()


@pytest.mark.integration
def test_a_recorded_admin_action_is_listed_back(clean_audit_log):
    audit.record(actor_email=TEST_ACTOR, action="retrain", target="agriculture/cinnamon")

    entries = audit.list_entries()

    matching = [e for e in entries if e.actor_email == TEST_ACTOR]
    assert len(matching) == 1
    assert matching[0].action == "retrain"
    assert matching[0].target == "agriculture/cinnamon"
    assert matching[0].logged_at is not None
