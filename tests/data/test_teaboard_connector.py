from pathlib import Path

import pandas as pd

from ceynex.data.connectors.teaboard import TeaBoardConnector


def _write_workbook(path: Path) -> None:
    production = pd.DataFrame(
        {
            "year": [2024, 2025],
            "orthodox_mt": [236216, 237079],
            "ctc_mt": [23680, 24709],
            "green_mt": [2261, 2333],
            "total_production_mt": [262159, 264122],
            "source": ["Tea Exporters Association"] * 2,
            "dq_flags": [None, None],
        }
    )
    exports = pd.DataFrame(
        {
            "year": [2024, 2025],
            "bulk_mt": [111074, 106801],
            "tea_in_packets_mt": [101818, 116245],
            "tea_bags_mt": [25584, 26444],
            "instant_tea_mt": [2623, 3026],
            "green_tea_mt": [4687, 4922],
            "total_exports_mt": [245787, 257440],
            "source": ["Tea Exporters Association"] * 2,
            "dq_flags": [None, None],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        production.to_excel(writer, sheet_name="Production", index=False, startrow=3)
        exports.to_excel(writer, sheet_name="Exports", index=False, startrow=3)


def test_teaboard_connector_stages_and_maps_total_exports(tmp_path: Path) -> None:
    raw_dir = tmp_path / "tea_board"
    workbook = raw_dir / "2026-08-15" / "tea_annual_production_exports_2011_2025.xlsx"
    workbook.parent.mkdir(parents=True)
    _write_workbook(workbook)
    connector = TeaBoardConnector(raw_dir, tmp_path / "staging")

    raw = connector.fetch()
    repeated_raw = connector.fetch()
    assert len(raw) == len(repeated_raw) == 20
    assert raw["source_hash"].tolist() == repeated_raw["source_hash"].tolist()
    assert set(raw["metric"]) == {"export", "production"}
    first_stage, second_stage = connector.stage(), connector.stage()
    assert first_stage == second_stage
    assert first_stage.exists()
    assert connector.manifest().period_start == "2024"

    fact_trade = connector.to_fact_trade(raw)
    assert fact_trade["export_volume"].tolist() == [245787000, 257440000]
    assert fact_trade["volume_unit"].tolist() == ["kg", "kg"]
