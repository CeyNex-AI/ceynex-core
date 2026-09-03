"""Regression coverage for ceynex/data/reader.py.

Two findings live here.

Found live 2026-08-26: every psycopg.connect() call elsewhere in this codebase
(api/history.py, api/admin.py, api/routes/health.py) already passes
connect_timeout=3 -- reader.py's three calls were the one place that didn't,
so an unreachable Postgres hung indefinitely instead of failing fast into the
existing DatasetUnavailableError path.

Found live 2026-09-03: `annual_series` summed every measured column, including
`price`. Cinnamon prices in fact_trade are Comtrade per-partner unit values
already in USD/kg, so the 71 partner rows for 2015 summed to 854.01 and the
system reported "cinnamon rose from 854.01 USD/kg in 2015 to 1,214.01 in 2025".
The mean is 12.03. The aggregation tests below are the fast guard; the
integration test at the bottom is the one that actually proves the number.
"""

from __future__ import annotations

import pandas as pd
import psycopg
import pytest

from ceynex.data import reader
from ceynex.data.writer import UnifiedDatasetWriter
from ceynex.settings import postgres_dsn

TEST_SOURCE = "PYTEST_READER"
OTHER_SOURCE = "PYTEST_READER_OTHER"


class _FakeCursor:
    def __init__(self, recorder=None):
        self._recorder = recorder

    def execute(self, sql, params=None):
        if self._recorder is not None:
            self._recorder.append((sql, params))
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeConn:
    def __init__(self, recorder=None):
        self._recorder = recorder

    def cursor(self, *_args, **_kwargs):
        return _FakeCursor(self._recorder)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _captured_query(monkeypatch, **kwargs):
    """The SQL and bound params `annual_series` would have run."""
    recorder: list[tuple[str, dict]] = []
    monkeypatch.setattr(reader.psycopg, "connect", lambda *a, **kw: _FakeConn(recorder))
    reader.annual_series("cinnamon", dsn="postgresql://x", **kwargs)
    return recorder[0]


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


# --- a price is averaged, a value is summed ------------------------------


def test_a_price_series_is_never_summed(monkeypatch):
    """The defect itself. `sum(price)` over per-partner unit values is a number
    in no unit at all -- 71 rows of 6-16 USD/kg summing to 854.01, served as a
    per-kg price.
    """
    sql, _ = _captured_query(monkeypatch, target="price")

    assert "sum(price)" not in sql.replace(" ", "")
    assert "avg(price)" in sql, "the fallback for rows with no volume is the plain mean"
    assert "sum(price * export_volume) / sum(export_volume)" in sql


@pytest.mark.parametrize("target", ["export_value_usd", "export_volume"])
def test_an_extensive_series_is_still_summed(monkeypatch, target):
    """Partner rows *do* add up to a national total for values and volumes. The
    price fix must not change what was already right.
    """
    sql, _ = _captured_query(monkeypatch, target=target)

    assert f"sum({target})" in sql
    assert "avg(" not in sql


def test_a_source_filter_is_bound_as_a_parameter(monkeypatch):
    """Without it the reader blends every source holding a price for the item, so
    the agriculture agent cited "FAOSTAT annual Sri Lanka cinnamon producer-price
    series" over what was really a Comtrade unit-value blend (live 2026-09-03).
    Parameterized, not interpolated -- same rule the KG client follows.
    """
    sql, params = _captured_query(monkeypatch, target="price", source_id="FAOSTAT")

    assert params["source_id"] == "FAOSTAT"
    assert "FAOSTAT" not in sql, "the value must be bound, never f-strung into the query"


def test_an_unmeasured_column_is_rejected():
    with pytest.raises(ValueError, match="not a measured column"):
        reader.annual_series("tea", target="sector")


# --- against real postgres ------------------------------------------------


@pytest.fixture
def clean_sources():
    """Remove this test's rows before and after, leaving real data alone."""

    def purge():
        with psycopg.connect(postgres_dsn()) as conn:
            conn.execute(
                "DELETE FROM fact_trade WHERE source_id = ANY(%s)",
                ([TEST_SOURCE, OTHER_SOURCE],),
            )
            conn.commit()

    purge()
    yield
    purge()


def _price_row(partner_iso3, m49, price, volume, *, source_id=TEST_SOURCE, year=2015):
    return {
        "source_id": source_id,
        "sector": "agriculture",
        "item": "pytest_spice",
        "hs_code": "0906",
        "reporter_iso3": "LKA",
        "reporter_m49": 144,
        "partner_iso3": partner_iso3,
        "partner_m49": m49,
        "period_start": f"{year}-01-01",
        "period_end": f"{year}-12-31",
        "frequency": "A",
        "export_volume": volume,
        "volume_unit": "kg",
        "export_value_usd": None if price is None else price * (volume or 0),
        "price": price,
        "price_unit": "USD/kg",
        "fx_usd_lkr": None,
    }


@pytest.mark.integration
def test_a_multi_partner_price_year_returns_a_weighted_mean(clean_sources, tmp_path):
    """The live shape: many partners, each with its own unit value. The answer is
    a price a reader can recognise, not the sum of every partner's price.
    """
    writer = UnifiedDatasetWriter(parquet_root=tmp_path / "parquet")
    writer.write(
        pd.DataFrame(
            [
                _price_row("DEU", 276, 10.0, 900.0),
                _price_row("USA", 840, 20.0, 100.0),
            ]
        ),
        source_id=TEST_SOURCE,
    )

    frame = reader.annual_series("pytest_spice", target="price", source_id=TEST_SOURCE)

    assert len(frame) == 1
    # Volume-weighted: (10*900 + 20*100) / 1000 = 11.0. The plain mean would be
    # 15.0 and the old sum would be 30.0.
    assert frame.iloc[0]["value"] == pytest.approx(11.0)


@pytest.mark.integration
def test_a_price_year_with_no_volumes_falls_back_to_the_plain_mean(clean_sources, tmp_path):
    """A source like FAOSTAT carries a producer price and no volume at all.
    Weighting a subset would drop the unweighted rows out of the average.
    """
    writer = UnifiedDatasetWriter(parquet_root=tmp_path / "parquet")
    writer.write(
        pd.DataFrame(
            [
                _price_row("DEU", 276, 10.0, None),
                _price_row("USA", 840, 20.0, None),
            ]
        ),
        source_id=TEST_SOURCE,
    )

    frame = reader.annual_series("pytest_spice", target="price", source_id=TEST_SOURCE)

    assert frame.iloc[0]["value"] == pytest.approx(15.0)


@pytest.mark.integration
def test_the_source_filter_keeps_two_sources_from_blending(clean_sources, tmp_path):
    """A Comtrade unit value and a FAOSTAT producer price measure different
    things. Averaging across them yields a number neither source would recognise,
    attributed to whichever one the caller named.
    """
    writer = UnifiedDatasetWriter(parquet_root=tmp_path / "parquet")
    writer.write(pd.DataFrame([_price_row("DEU", 276, 10.0, None)]), source_id=TEST_SOURCE)
    writer.write(
        pd.DataFrame([_price_row("DEU", 276, 900.0, None, source_id=OTHER_SOURCE)]),
        source_id=OTHER_SOURCE,
    )

    filtered = reader.annual_series("pytest_spice", target="price", source_id=TEST_SOURCE)
    blended = reader.annual_series("pytest_spice", target="price")

    assert filtered.iloc[0]["value"] == pytest.approx(10.0)
    assert blended.iloc[0]["value"] == pytest.approx(455.0), "unfiltered still blends -- hence the filter"
