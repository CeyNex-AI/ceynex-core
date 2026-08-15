from pathlib import Path

import pandas as pd

from ceynex.data.connectors.pinksheet import PinkSheetConnector


def test_pinksheet_connector_is_repeatable_with_offline_workbook_fixture(tmp_path: Path) -> None:
    workbook = tmp_path / "pink_sheet.xlsx"
    pd.DataFrame(
        [
            ["Period", "Tea, Colombo", "Copper"],
            ["2024M01", 2.10, 8.50],
            ["2024M02", 2.20, 8.60],
            ["Notes", None, None],
        ]
    ).to_excel(workbook, sheet_name="Monthly Prices", header=False, index=False)
    connector = PinkSheetConnector(workbook, tmp_path / "staging")

    first = connector.fetch()
    second = connector.fetch()

    assert len(first) == len(second) == 2
    assert first["source_hash"].tolist() == second["source_hash"].tolist()
    first_stage, second_stage = connector.stage(), connector.stage()
    assert first_stage == second_stage
    assert first_stage.exists()
    assert connector.manifest().period_start == "2024M01"
    assert connector.manifest().period_end == "2024M02"
