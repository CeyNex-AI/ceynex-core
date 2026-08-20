from pathlib import Path

import pandas as pd

from ceynex.data.cleaning import DataCleaner
from ceynex.data.connectors.pinksheet import PinkSheetConnector


def test_pinksheet_connector_is_repeatable_with_offline_workbook_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "pinksheet"
    workbook = raw_dir / "2026-08-15" / "CMO-Historical-Data-Monthly.xlsx"
    workbook.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            ["Period", "Tea, Colombo", "Copper"],
            ["2024M01", 2.10, 8.50],
            ["2024M02", 2.20, 8.60],
            ["Notes", None, None],
        ]
    ).to_excel(workbook, sheet_name="Monthly Prices", header=False, index=False)
    connector = PinkSheetConnector(raw_dir, tmp_path / "staging")

    first = connector.fetch()
    second = connector.fetch()

    assert len(first) == len(second) == 2
    assert first["source_hash"].tolist() == second["source_hash"].tolist()
    first_stage, second_stage = connector.stage(), connector.stage()
    assert first_stage == second_stage
    assert first_stage.exists()
    assert connector.manifest().period_start == "2024M01"
    assert connector.manifest().period_end == "2024M02"

    fact_trade = connector.to_fact_trade(first)
    assert fact_trade["period_start"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-01", "2024-02-01"]
    assert fact_trade["period_end"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-31", "2024-02-29"]
    assert fact_trade["price"].tolist() == [2.10, 2.20]
    assert fact_trade["price_unit"].tolist() == ["USD/kg", "USD/kg"]
    assert fact_trade["reporter_iso3"].tolist() == ["LKA", "LKA"]
    assert fact_trade["reporter_m49"].tolist() == [144, 144]
    assert fact_trade["hs_code"].tolist() == ["0902", "0902"]

    cleaned = DataCleaner().clean(fact_trade)
    assert cleaned["price"].tolist() == [2.10, 2.20]
    assert cleaned["price_unit"].tolist() == ["USD/kg", "USD/kg"]
    assert cleaned["original_price_unit"].tolist() == ["USD/kg", "USD/kg"]
