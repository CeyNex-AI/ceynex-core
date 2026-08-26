"""Regression coverage for ceynex/data/reader.py's connection handling.

Found live 2026-08-26: every psycopg.connect() call elsewhere in this codebase
(api/history.py, api/admin.py, api/routes/health.py) already passes
connect_timeout=3 -- reader.py's three calls were the one place that didn't,
so an unreachable Postgres hung indefinitely instead of failing fast into the
existing DatasetUnavailableError path.
"""

from __future__ import annotations

import pytest

from ceynex.data import reader


class _FakeCursor:
    def execute(self, *_args, **_kwargs):
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeConn:
    def cursor(self, *_args, **_kwargs):
        return _FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.mark.parametrize(
    "call",
    [
        lambda: reader.annual_series("tea", dsn="postgresql://x"),
        lambda: reader.items(dsn="postgresql://x"),
        lambda: reader.relevant_dq_flags("tea", "price", dsn="postgresql://x"),
    ],
)
def test_every_connect_call_passes_a_bounded_connect_timeout(monkeypatch, call):
    recorded: list[dict] = []
    monkeypatch.setattr(reader.psycopg, "connect", lambda *a, **kw: recorded.append(kw) or _FakeConn())

    call()

    assert recorded, "psycopg.connect was never called"
    assert recorded[0].get("connect_timeout") == 3
