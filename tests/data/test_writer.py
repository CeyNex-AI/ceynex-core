"""Assertions for the UnifiedDatasetWriter (SRS 3.1.8, 3.10.2).

Validation and hashing are pure and tested without a database. The properties
that only exist against real Postgres — upsert idempotency, dq_flag persistence,
ingest_run durability — are marked `integration`.

The idempotency test is the one that matters. Running `make ingest` twice must
produce the same row count; the failure mode is silent duplication that makes
every aggregate figure wrong.
"""

import pandas as pd
import psycopg
import pytest

from ceynex.contracts import DQFlag
from ceynex.data.writer import (
    IDENTITY_COLUMNS,
    UnifiedDatasetWriter,
    WriterError,
    source_hash,
)
from ceynex.settings import postgres_dsn

TEST_SOURCE = "PYTEST_WRITER"


def record(**overrides):
    base = {
        "source_id": TEST_SOURCE,
        "sector": "agriculture",
        "item": "tea",
        "hs_code": "0902",
        "reporter_iso3": "LKA",
        "reporter_m49": 144,
        "partner_iso3": "DEU",
        "partner_m49": 276,
        "period_start": "2023-01-01",
        "period_end": "2023-12-31",
        "frequency": "A",
        "export_volume": 1000.0,
        "volume_unit": "kg",
        "export_value_usd": 8500.0,
        "price": 8.5,
        "price_unit": "USD/kg",
        "fx_usd_lkr": None,
    }
    base.update(overrides)
    return base


def frame(*records):
    return pd.DataFrame(list(records) or [record()])


# --- hashing -------------------------------------------------------------


def test_hash_is_stable_for_the_same_row():
    assert source_hash(record()) == source_hash(record())


def test_hash_changes_when_a_measured_value_changes():
    """A revised Comtrade figure must be seen as an update, not a no-op."""
    assert source_hash(record()) != source_hash(record(export_value_usd=9000.0))


def test_hash_changes_when_identity_changes():
    for column in IDENTITY_COLUMNS:
        altered = record(**{column: "CHANGED"})
        assert source_hash(altered) != source_hash(record()), column


# --- validation ----------------------------------------------------------


def test_missing_non_nullable_column_is_refused_before_anything_is_written():
    writer = UnifiedDatasetWriter(dsn="postgresql://nowhere/nope")
    incomplete = frame().drop(columns=["sector"])
    with pytest.raises(WriterError, match="sector"):
        writer.prepare(incomplete)


def test_a_null_in_a_non_nullable_column_is_refused():
    writer = UnifiedDatasetWriter(dsn="postgresql://nowhere/nope")
    with pytest.raises(WriterError, match="non-nullable"):
        writer.prepare(frame(record(item=None)))


def test_an_invalid_frequency_is_refused():
    writer = UnifiedDatasetWriter(dsn="postgresql://nowhere/nope")
    with pytest.raises(WriterError, match="frequency"):
        writer.prepare(frame(record(frequency="YEARLY")))


def test_prepare_adds_the_source_hash():
    writer = UnifiedDatasetWriter(dsn="postgresql://nowhere/nope")
    prepared = writer.prepare(frame())
    assert prepared["source_hash"].notna().all()


def test_duplicates_within_one_batch_collapse_to_the_last():
    """Postgres refuses to update the same row twice in one statement."""
    writer = UnifiedDatasetWriter(dsn="postgresql://nowhere/nope")
    prepared = writer.prepare(
        frame(record(export_value_usd=1.0), record(export_value_usd=2.0))
    )
    assert len(prepared) == 1
    assert prepared.iloc[0]["export_value_usd"] == 2.0


def test_optional_columns_may_be_absent_entirely():
    writer = UnifiedDatasetWriter(dsn="postgresql://nowhere/nope")
    sparse = frame().drop(columns=["price", "fx_usd_lkr"])
    prepared = writer.prepare(sparse)
    assert prepared["price"].isna().all()


def test_an_empty_frame_is_a_no_op_not_an_error():
    writer = UnifiedDatasetWriter(dsn="postgresql://nowhere/nope")
    result = writer.write(pd.DataFrame())
    assert result.status == "success"
    assert result.rows_written == 0


def test_an_unreachable_database_fails_without_raising():
    """The pipeline reports a failed source; it does not crash on one."""
    writer = UnifiedDatasetWriter(dsn="postgresql://ceynex:x@127.0.0.1:1/ceynex")
    result = writer.write(frame())
    assert result.status == "failed"
    assert result.error


# --- against real postgres ----------------------------------------------


@pytest.fixture
def clean_source():
    """Remove this test's rows before and after, leaving real data alone."""

    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute("DELETE FROM fact_trade WHERE source_id = %s", (TEST_SOURCE,))
            conn.execute("DELETE FROM dq_flag WHERE source_a = %s", (TEST_SOURCE,))
            conn.execute("DELETE FROM ingest_run WHERE source_id = %s", (TEST_SOURCE,))
            conn.commit()

    purge()
    yield
    purge()


def count_rows() -> int:
    with psycopg.connect(postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM fact_trade WHERE source_id = %s", (TEST_SOURCE,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


@pytest.mark.integration
def test_writing_twice_does_not_duplicate(clean_source, tmp_path):
    """`make ingest` run twice must give the same count, not double it."""
    writer = UnifiedDatasetWriter(parquet_root=tmp_path / "parquet")
    records = frame(record(partner_iso3="DEU", partner_m49=276),
                    record(partner_iso3="GBR", partner_m49=826))

    writer.write(records)
    assert count_rows() == 2

    writer.write(records)
    assert count_rows() == 2, "the upsert is not idempotent"


@pytest.mark.integration
def test_a_null_partner_row_also_upserts(clean_source, tmp_path):
    """Deviation D5. NULLS DISTINCT would insert this twice and nothing would raise."""
    writer = UnifiedDatasetWriter(parquet_root=tmp_path / "parquet")
    world = frame(record(partner_iso3=None, partner_m49=None))

    writer.write(world)
    writer.write(world)
    assert count_rows() == 1, "NULL partner rows are duplicating — check fact_trade_upsert_key"


@pytest.mark.integration
def test_a_revised_figure_updates_in_place(clean_source, tmp_path):
    writer = UnifiedDatasetWriter(parquet_root=tmp_path / "parquet")
    writer.write(frame(record(export_value_usd=8500.0)))
    writer.write(frame(record(export_value_usd=9900.0)))

    assert count_rows() == 1
    with psycopg.connect(postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT export_value_usd FROM fact_trade WHERE source_id = %s", (TEST_SOURCE,)
        )
        row = cur.fetchone()
    assert row is not None
    assert float(row[0]) == 9900.0


@pytest.mark.integration
def test_the_run_is_logged_for_the_admin_page(clean_source, tmp_path):
    """SRS 3.5.4 — the admin pipeline status reads ingest_run."""
    writer = UnifiedDatasetWriter(parquet_root=tmp_path / "parquet")
    result = writer.write(frame())

    with psycopg.connect(postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT status, rows_written, finished_at FROM ingest_run WHERE run_id = %s",
            (result.run_id,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "success"
    assert row[1] == 1
    assert row[2] is not None, "finished_at unset — the run looks stuck at 'running'"


@pytest.mark.integration
def test_dq_flags_from_an_injected_validator_are_persisted(clean_source, tmp_path):
    """Deviation D3 — this is the seam M1's CrossValidator plugs into."""

    class StubValidator:
        def cross_validate(self, records):
            return [
                DQFlag(
                    item="tea",
                    metric="export_value_usd",
                    source_a=TEST_SOURCE,
                    value_a=8500.0,
                    source_b="EDB",
                    value_b=9200.0,
                    pct_diff=8.2,
                    severity="material",
                    hs_code="0902",
                    partner_iso3="DEU",
                    period_start="2023-01-01",
                )
            ]

    writer = UnifiedDatasetWriter(
        cross_validator=StubValidator(), parquet_root=tmp_path / "parquet"
    )
    result = writer.write(frame())
    assert result.dq_flags == 1

    with psycopg.connect(postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT severity, pct_diff FROM dq_flag WHERE source_a = %s", (TEST_SOURCE,)
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == "material"


@pytest.mark.integration
def test_parquet_is_partitioned_by_sector_item_year(clean_source, tmp_path):
    parquet_root = tmp_path / "parquet"
    writer = UnifiedDatasetWriter(parquet_root=parquet_root)
    writer.write(frame(record(), record(sector="apparel", item="apparel_knit", hs_code="61",
                                        partner_iso3="USA", partner_m49=842)))

    partitions = {p.relative_to(parquet_root).parts[:3] for p in parquet_root.rglob("*.parquet")}
    assert ("sector=agriculture", "item=tea", "year=2023") in partitions
    assert ("sector=apparel", "item=apparel_knit", "year=2023") in partitions
