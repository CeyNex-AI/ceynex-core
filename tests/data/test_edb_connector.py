"""No network access: parsing is exercised against a committed text-layer
fixture (`fixtures/edb_sample.txt`) rather than a real PDF — pdfplumber
consumes bytes decoded from a PDF, but `parse_table_page` itself only ever
sees the page's plain-text layer, so the fixture skips the PDF encoding step
without skipping anything the connector actually does with the text.
"""

from pathlib import Path

import pandas as pd
import pytest

from ceynex.data.connectors.edb import (
    EDBConnector,
    EDBReportSource,
    parse_archive_table_page,
    parse_table_page,
)

FIXTURE = Path(__file__).parent / "fixtures" / "edb_sample.txt"
ARCHIVE_FIXTURE = Path(__file__).parent / "fixtures" / "edb_archive_sample.txt"

_FACT_TRADE_COLUMNS = {
    "source_id",
    "sector",
    "item",
    "hs_code",
    "reporter_iso3",
    "reporter_m49",
    "partner_iso3",
    "partner_m49",
    "period_start",
    "period_end",
    "frequency",
    "export_volume",
    "volume_unit",
    "export_value_usd",
    "price",
    "price_unit",
    "fx_usd_lkr",
    "source_hash",
}

_NON_NULLABLE = {
    "source_id",
    "sector",
    "item",
    "reporter_iso3",
    "reporter_m49",
    "period_start",
    "period_end",
    "frequency",
    "export_value_usd",
    "source_hash",
}


def _raw_df() -> pd.DataFrame:
    text = FIXTURE.read_text()
    rows = parse_table_page(text)
    for row in rows:
        row["table_id"] = "25.79"
        row["product"] = "Knitted Apparel"
        row["edition_year"] = 2024
        row["latest_year"] = 2024
    return pd.DataFrame.from_records(rows)


def test_fixture_parses_to_expected_row_count_and_dtypes():
    rows = parse_table_page(FIXTURE.read_text())
    assert len(rows) == 22
    first = rows[0]
    assert first["rank"] == 1
    assert first["market"] == "United States"
    assert isinstance(first["year_latest"], float)
    assert first["year_latest"] == pytest.approx(301220.9)
    assert first["share_pct"] == pytest.approx(24.5)


def test_to_fact_trade_produces_every_non_nullable_column():
    connector = EDBConnector(sources=[EDBReportSource(edition_year=2024, latest_year=2024)])
    out = connector.to_fact_trade(_raw_df())
    assert set(out.columns) == _FACT_TRADE_COLUMNS
    assert not out[list(_NON_NULLABLE)].isnull().any().any()


def test_to_fact_trade_melts_five_years_per_market():
    connector = EDBConnector(sources=[EDBReportSource(edition_year=2024, latest_year=2024)])
    out = connector.to_fact_trade(_raw_df())
    us_rows = out[(out["partner_iso3"] == "USA")]
    assert len(us_rows) == 5
    assert set(us_rows["period_start"]) == {
        "2020-01-01",
        "2021-01-01",
        "2022-01-01",
        "2023-01-01",
        "2024-01-01",
    }
    latest = us_rows[us_rows["period_start"] == "2024-01-01"].iloc[0]
    assert latest["export_value_usd"] == pytest.approx(301220.9 * 1_000_000.0)


def test_to_fact_trade_drops_markets_the_crosswalk_cannot_resolve():
    connector = EDBConnector(sources=[EDBReportSource(edition_year=2024, latest_year=2024)])
    out = connector.to_fact_trade(_raw_df())
    # "Nowhereistan" (row 22 in the fixture) isn't in the crosswalk, so its 5
    # year-rows must be absent rather than written with a null partner.
    assert len(out) == 21 * 5


def test_idempotency_same_input_yields_same_source_hash():
    connector = EDBConnector(sources=[EDBReportSource(edition_year=2024, latest_year=2024)])
    raw = _raw_df()
    first = connector.to_fact_trade(raw)
    second = connector.to_fact_trade(raw)
    assert list(first["source_hash"]) == list(second["source_hash"])


def test_to_fact_trade_dedupes_overlapping_editions_keeping_the_newer_one():
    # Real-world shape: the 2023 edition's tables span 2019-2023, the 2024
    # edition's span 2020-2024, so 2020-2023 is reported by BOTH editions.
    # Before the fix, concatenating both editions' raw rows made to_fact_trade
    # emit two rows per overlapping (item, market, year) -- confirmed against
    # the real 2023+2024 EDB PDFs, ~37% of all EDB rows were exact duplicates
    # of this shape.
    edition_2023 = pd.DataFrame.from_records(
        [
            {
                "rank": 1,
                "market": "United States",
                "year_minus4": 100.0,
                "year_minus3": 110.0,
                "year_minus2": 120.0,
                "year_minus1": 130.0,
                "year_latest": 140.0,  # 2023 figure, later superseded
                "share_pct": 25.0,
                "avg_growth_pct": 5.0,
                "table_id": "25.79",
                "product": "Apparel",
                "edition_year": 2023,
                "latest_year": 2023,
            }
        ]
    )
    edition_2024 = pd.DataFrame.from_records(
        [
            {
                "rank": 1,
                "market": "United States",
                "year_minus4": 110.0,
                "year_minus3": 120.0,
                "year_minus2": 130.0,
                "year_minus1": 999.0,  # 2023, revised in the newer edition
                "year_latest": 150.0,  # 2024
                "share_pct": 24.0,
                "avg_growth_pct": 6.0,
                "table_id": "25.79",
                "product": "Apparel",
                "edition_year": 2024,
                "latest_year": 2024,
            }
        ]
    )
    raw = pd.concat([edition_2023, edition_2024], ignore_index=True)

    connector = EDBConnector(sources=[EDBReportSource(edition_year=2024, latest_year=2024)])
    out = connector.to_fact_trade(raw)

    us_rows = out[out["partner_iso3"] == "USA"]
    # one row per year, not one row per (year, edition) -- the overlap collapses
    assert sorted(us_rows["period_start"]) == [
        "2019-01-01",
        "2020-01-01",
        "2021-01-01",
        "2022-01-01",
        "2023-01-01",
        "2024-01-01",
    ]
    assert not out.duplicated(subset=["item", "partner_iso3", "period_start"]).any()

    # 2023 is reported by both editions -- the newer (2024) edition's revised
    # figure must win over the older (2023) edition's superseded one.
    us_2023 = us_rows[us_rows["period_start"] == "2023-01-01"].iloc[0]
    assert us_2023["export_value_usd"] == pytest.approx(999.0 * 1_000_000.0)


# --- "archive" layout: older multi-year volumes, interleaved Value/%Share
# columns per year rather than 5 values then one trailing %share. Confirmed
# against a real 2009-2018-titled EDB archive PDF. ---


def _archive_raw_df() -> pd.DataFrame:
    rows = parse_archive_table_page(ARCHIVE_FIXTURE.read_text())
    for row in rows:
        row["table_id"] = "17.82"
        row["product"] = "APPAREL"
        row["edition_year"] = 2018
        row["latest_year"] = 2018
    return pd.DataFrame.from_records(rows)


def test_archive_fixture_parses_to_expected_row_count_and_drops_non_latest_shares():
    rows = parse_archive_table_page(ARCHIVE_FIXTURE.read_text())
    assert len(rows) == 21
    first = rows[0]
    assert first["rank"] == 1
    assert first["market"] == "United States"
    assert first["year_minus4"] == pytest.approx(1989.98)
    assert first["year_latest"] == pytest.approx(2269.47)
    assert first["share_pct"] == pytest.approx(45.75)  # latest year's %share only
    assert first["avg_growth_pct"] == pytest.approx(2.77)


def test_archive_to_fact_trade_produces_the_same_shape_as_annual_layout():
    connector = EDBConnector(
        sources=[EDBReportSource(edition_year=2018, latest_year=2018, layout="archive")]
    )
    out = connector.to_fact_trade(_archive_raw_df())
    assert set(out.columns) == _FACT_TRADE_COLUMNS
    assert not out[list(_NON_NULLABLE)].isnull().any().any()
    # 20 resolvable markets ("Nowhereistan" dropped) x 5 years
    assert len(out) == 20 * 5


def test_archive_to_fact_trade_matches_the_real_pdf_figure():
    connector = EDBConnector(
        sources=[EDBReportSource(edition_year=2018, latest_year=2018, layout="archive")]
    )
    out = connector.to_fact_trade(_archive_raw_df())
    us_2018 = out[(out["partner_iso3"] == "USA") & (out["period_start"] == "2018-01-01")].iloc[0]
    assert us_2018["export_value_usd"] == pytest.approx(2269.47 * 1_000_000.0)
