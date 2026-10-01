"""GET /api/data/freshness: how current the data is.

Until 2026-10 the last-ingest query filtered `ingest_run.status = 'ok'`, a
status the writer never writes (it writes running | success | failed), so
`last_ingest_at` was null on every deployment and no test noticed: there was no
test.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import psycopg
import pytest
from fastapi.testclient import TestClient

from ceynex.api.main import app
from ceynex.api.routes import data as data_routes
from ceynex.settings import postgres_dsn
from tests.api.test_query import signed_in

TEST_SOURCE = "PYTEST_FRESHNESS"


class _Cursor:
    def __init__(self, results):
        self._results = list(results)
        self.sql: list[str] = []

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchone(self):
        return self._results.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_last_ingest_is_the_last_successful_run(monkeypatch):
    cursor = _Cursor([(date(2025, 12, 1), 13112), (datetime(2026, 9, 27, 4, 0, tzinfo=UTC),)])
    monkeypatch.setattr(data_routes.psycopg, "connect", lambda *a, **k: _Connection(cursor))

    response = TestClient(app, headers=signed_in()).get("/api/data/freshness")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["observations"] == 13112
    assert body["last_ingest_at"].startswith("2026-09-27T04:00")
    assert "status = 'success'" in cursor.sql[1], "the writer's own status, not 'ok'"


def test_an_unreachable_database_is_reported_not_raised(monkeypatch):
    def down(*_a, **_k):
        raise psycopg.OperationalError("postgres down")

    monkeypatch.setattr(data_routes.psycopg, "connect", down)

    response = TestClient(app, headers=signed_in()).get("/api/data/freshness")

    assert response.status_code == 200
    assert response.json()["available"] is False


def test_freshness_requires_sign_in():
    assert TestClient(app).get("/api/data/freshness").status_code == 401


@pytest.fixture
def a_successful_run():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM ingest_run WHERE source_id = %s", (TEST_SOURCE,))
            conn.commit()

    purge()
    with psycopg.connect(postgres_dsn()) as conn:
        conn.execute(
            "INSERT INTO ingest_run (source_id, status, rows_written, finished_at) "
            "VALUES (%s, 'success', 1, now())",
            (TEST_SOURCE,),
        )
        conn.commit()
    yield
    purge()


@pytest.mark.integration
def test_a_real_successful_run_is_reported(a_successful_run):
    assert data_routes._freshness_sync()["last_ingest_at"] is not None
