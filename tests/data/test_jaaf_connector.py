"""No network access: fixtures are pre-saved HTML, matching what fetch() itself
expects on disk (srilankaapparel.com blocks automated fetching)."""

from pathlib import Path

import pandas as pd
import pytest

from ceynex.data.connectors.jaaf import (
    JAAFConnector,
    _annual_tables_to_records,
    extract_annual_exports_html,
    extract_market_wise_html,
)

ANNUAL_FIXTURE = Path(__file__).parent / "fixtures" / "jaaf_annual_exports_sample.html"
MARKET_WISE_FIXTURE = Path(__file__).parent / "fixtures" / "jaaf_market_wise_sample.html"

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


def test_fixture_parses_to_expected_table_count_and_labels():
    tables = extract_annual_exports_html(ANNUAL_FIXTURE.read_text())
    assert [t["market"] for t in tables] == ["total", "us", "eu_bloc_approx", "uk", "other_approx"]
    assert len(tables[0]["rows"]) == 2  # 2 year-rows in the fixture (2024, 2025)


def test_annual_tables_to_records_skips_the_literal_nan_placeholder():
    tables = extract_annual_exports_html(ANNUAL_FIXTURE.read_text())
    uk_records = [r for r in _annual_tables_to_records(tables) if r["market"] == "uk"]
    # Fixture's UK/2025/Feb cell is the literal text "NaN" (matches a real
    # saved page) — it must be dropped, not parsed as float("nan").
    assert not any(r["year"] == 2025 and r["month"] == 2 for r in uk_records)
    assert len(uk_records) == 5  # 2025: Jan, Mar (Feb dropped); 2024: Jan, Feb, Mar


def test_market_wise_fixture_parses_labels_and_values():
    result = extract_market_wise_html(MARKET_WISE_FIXTURE.read_text())
    assert result == [("USA", 38.5), ("UK", 15.2), ("EU", 22.1), ("Other", 24.2)]


def _raw_df() -> pd.DataFrame:
    tables = extract_annual_exports_html(ANNUAL_FIXTURE.read_text())
    return pd.DataFrame.from_records(_annual_tables_to_records(tables))


def test_to_fact_trade_produces_every_non_nullable_column():
    connector = JAAFConnector(annual_exports_path=str(ANNUAL_FIXTURE))
    out = connector.to_fact_trade(_raw_df())
    assert set(out.columns) == _FACT_TRADE_COLUMNS
    assert not out[list(_NON_NULLABLE)].isnull().any().any()


def test_to_fact_trade_keeps_only_total_us_and_uk():
    connector = JAAFConnector(annual_exports_path=str(ANNUAL_FIXTURE))
    out = connector.to_fact_trade(_raw_df())
    # total: 3 months x 2 years = 6; us: 6; uk: 5 (one "NaN" cell dropped).
    # EU/Other excluded entirely.
    assert len(out) == 17
    assert set(out["partner_iso3"].dropna()) == {"USA", "GBR"}
    assert out[out["partner_iso3"].isna()]["item"].nunique() == 1  # the "total"/World rows


def test_to_fact_trade_total_row_is_written_as_world_null_partner():
    connector = JAAFConnector(annual_exports_path=str(ANNUAL_FIXTURE))
    out = connector.to_fact_trade(_raw_df())
    jan_2024_total = out[
        (out["partner_iso3"].isna()) & (out["period_start"] == "2024-01-01")
    ].iloc[0]
    assert jan_2024_total["export_value_usd"] == pytest.approx(380.0 * 1_000_000.0)
    assert jan_2024_total["period_end"] == "2024-01-31"


def test_idempotency_same_input_yields_same_source_hash():
    connector = JAAFConnector(annual_exports_path=str(ANNUAL_FIXTURE))
    raw = _raw_df()
    first = connector.to_fact_trade(raw)
    second = connector.to_fact_trade(raw)
    assert list(first["source_hash"]) == list(second["source_hash"])


def test_fetch_raises_a_clear_error_when_the_page_has_not_been_saved(tmp_path):
    connector = JAAFConnector(annual_exports_path=str(tmp_path / "missing.html"))
    with pytest.raises(FileNotFoundError, match="srilankaapparel.com"):
        connector.fetch()
