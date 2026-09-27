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


def test_annual_tables_to_records_skips_zero_value_future_month_placeholders():
    """Regression: a real saved page had the in-progress year's not-yet-
    reported months rendered as a literal "0" (not "NaN"/blank), which the
    connector ingested as real $0 rows. Found live 2026-08-26 -- this pushed
    JAAF's latest observed year into the future, which in turn corrupted
    every cross-item "latest year" graph query (export_analytics,
    trade_economics both assumed the global max year was real data).
    """
    tables = [
        {
            "market": "us",
            "header": ["", "Jan", "Feb"],
            "rows": [["2026", "165110000", "0"]],
        }
    ]
    records = _annual_tables_to_records(tables)
    assert len(records) == 1
    assert records[0]["month"] == 1
    assert records[0]["value_usd_mn"] == 165110000


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


# --- market-wise per-country rows, wired 2026-09-27 -----------------------


def _market_wise_html(labels_and_values: list[tuple[str, float]]) -> str:
    labels = ", ".join(f'"{label}"' for label, _ in labels_and_values)
    values = ", ".join(str(value) for _, value in labels_and_values)
    return (
        "<html><body><canvas id='c'></canvas><script>"
        f"var c = new Chart(ctx, {{type: 'pie', data: {{labels: [{labels}], "
        f"datasets: [{{data: [{values}]}}]}}}});"
        "</script></body></html>"
    )


def test_market_wise_rows_are_written_as_one_annual_row_per_new_country(tmp_path):
    """The 2025 US total from the fixture's "us" table is 465.0 (Jan+Feb+Mar,
    the only months this fixture carries) -- the pie chart's own US figure
    must match that to resolve 2025 as the snapshot's year."""
    market_wise_path = tmp_path / "market_wise.html"
    market_wise_path.write_text(
        _market_wise_html([("US", 465.0), ("UK", 124.0), ("Italy", 90.5), ("Other Markets", 30.0)])
    )
    connector = JAAFConnector(
        annual_exports_path=str(ANNUAL_FIXTURE),
        market_wise_path=str(market_wise_path),
        data_dir=str(tmp_path / "cache"),
    )

    raw = connector.fetch()
    assert connector.manifest().notes["market_wise_year"] == 2025

    out = connector.to_fact_trade(raw)
    italy = out[(out["partner_iso3"] == "ITA") & (out["frequency"] == "A")]
    assert len(italy) == 1
    assert italy.iloc[0]["period_start"] == "2025-01-01"
    assert italy.iloc[0]["period_end"] == "2025-12-31"
    assert italy.iloc[0]["export_value_usd"] == pytest.approx(90.5 * 1_000_000.0)


def test_market_wise_excludes_us_uk_and_the_other_markets_residual(tmp_path):
    """US/UK are already written monthly from the Annual Exports tables --
    writing them again annually from market-wise would double-count the same
    real trade flows under two frequencies. "Other Markets" cannot be
    geocoded to a real partner at all."""
    market_wise_path = tmp_path / "market_wise.html"
    market_wise_path.write_text(
        _market_wise_html([("US", 465.0), ("UK", 124.0), ("Italy", 90.5), ("Other Markets", 30.0)])
    )
    connector = JAAFConnector(
        annual_exports_path=str(ANNUAL_FIXTURE),
        market_wise_path=str(market_wise_path),
        data_dir=str(tmp_path / "cache"),
    )

    raw = connector.fetch()
    out = connector.to_fact_trade(raw)
    annual_rows = out[out["frequency"] == "A"]
    assert set(annual_rows["partner_iso3"]) == {"ITA"}


def test_an_unrecognized_market_wise_label_is_dropped_not_guessed_at(tmp_path):
    market_wise_path = tmp_path / "market_wise.html"
    market_wise_path.write_text(
        _market_wise_html([("US", 465.0), ("Neverland", 12.0)])
    )
    connector = JAAFConnector(
        annual_exports_path=str(ANNUAL_FIXTURE),
        market_wise_path=str(market_wise_path),
        data_dir=str(tmp_path / "cache"),
    )

    raw = connector.fetch()
    out = connector.to_fact_trade(raw)
    assert not any(out["frequency"] == "A")


def test_an_unresolvable_year_is_flagged_not_silently_mis_dated(tmp_path):
    """The pie chart's US figure matches no real annual "us" total -- table
    identity itself is suspect (module docstring), so no market-wise rows
    are written rather than guessed at under the wrong year."""
    market_wise_path = tmp_path / "market_wise.html"
    market_wise_path.write_text(
        _market_wise_html([("US", 999999.0), ("Italy", 90.5)])
    )
    connector = JAAFConnector(
        annual_exports_path=str(ANNUAL_FIXTURE),
        market_wise_path=str(market_wise_path),
        data_dir=str(tmp_path / "cache"),
    )

    raw = connector.fetch()
    assert connector.manifest().notes["market_wise_year_unresolved"] is True
    out = connector.to_fact_trade(raw)
    assert not any(out["frequency"] == "A")
