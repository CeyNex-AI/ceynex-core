"""Per-source freshness against config/sources.yaml's cadence (FR-DAT-03, SAD C11)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from ceynex.data import freshness
from ceynex.settings import postgres_dsn

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
CONFIG = {
    "UN_COMTRADE": {"cadence_days": 35, "refresh": True},
    "FAOSTAT": {"cadence_days": 400, "refresh": False},
    "EDB": {"cadence_days": None, "refresh": False},
}


def by_name(rows):
    return {row.source_id: row for row in rows}


def test_a_source_inside_its_cadence_is_fresh():
    rows = by_name(freshness.assess(CONFIG, {"UN_COMTRADE": (NOW - timedelta(days=3), 6712)}, {}, NOW))
    assert rows["UN_COMTRADE"].stale is False
    assert rows["UN_COMTRADE"].age_days == 3.0
    assert rows["UN_COMTRADE"].last_success_rows == 6712


def test_a_source_past_its_cadence_is_stale():
    rows = by_name(freshness.assess(CONFIG, {"UN_COMTRADE": (NOW - timedelta(days=36), 1)}, {}, NOW))
    assert rows["UN_COMTRADE"].stale is True


def test_a_source_with_a_cadence_and_no_success_ever_is_stale():
    assert by_name(freshness.assess(CONFIG, {}, {}, NOW))["UN_COMTRADE"].stale is True


def test_a_hand_refreshed_source_is_never_stale():
    rows = by_name(freshness.assess(CONFIG, {"EDB": (NOW - timedelta(days=900), 4750)}, {}, NOW))
    assert rows["EDB"].stale is False
    assert rows["EDB"].cadence_days is None


def test_a_failure_since_the_last_success_is_shown_beside_it():
    rows = by_name(freshness.assess(
        CONFIG,
        {"UN_COMTRADE": (NOW - timedelta(days=40), 1)},
        {"UN_COMTRADE": (NOW - timedelta(days=1), "comtradeapi.un.org unreachable")},
        NOW,
    ))
    assert rows["UN_COMTRADE"].stale is True
    assert rows["UN_COMTRADE"].last_error == "comtradeapi.un.org unreachable"


def test_every_configured_source_is_listed_and_unknown_sources_follow():
    rows = freshness.assess(CONFIG, {"PYTEST_ODD": (NOW, 1)}, {}, NOW)
    assert [row.source_id for row in rows] == ["UN_COMTRADE", "FAOSTAT", "EDB", "PYTEST_ODD"]
    assert freshness.stale_count(rows) == 2  # UN_COMTRADE and FAOSTAT have never succeeded


def test_the_shipped_config_names_every_connector():
    from ceynex.data.pipeline import SOURCE_IDS

    assert set(freshness.configured()) == set(SOURCE_IDS.values())
    for name, entry in freshness.configured().items():
        assert entry["connector"] in SOURCE_IDS and SOURCE_IDS[entry["connector"]] == name


@pytest.fixture
def runs():
    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM ingest_run WHERE source_id = 'PYTEST_FRESH'")
            conn.commit()

    purge()
    with psycopg.connect(postgres_dsn()) as conn:
        conn.execute(
            "INSERT INTO ingest_run (source_id, status, rows_written, finished_at) VALUES "
            "('PYTEST_FRESH', 'success', 5, now() - interval '2 days'), "
            "('PYTEST_FRESH', 'failed', 0, now())"
        )
        conn.commit()
    yield
    purge()


@pytest.mark.integration
def test_per_source_reads_real_runs(runs):
    row = by_name(freshness.per_source())["PYTEST_FRESH"]
    assert row.last_success_rows == 5
    assert row.last_failure_at is not None
    assert 1.9 < row.age_days < 2.1
