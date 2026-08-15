from pathlib import Path

import pandas as pd

from ceynex.data.connectors.cinnamon import CinnamonConnector


def _write_workbook(path: Path) -> None:
    columns = [
        "year", "metric", "category", "value", "unit", "source",
        "source_file", "source_url", "flag", "dq_flags",
    ]
    faostat = pd.DataFrame(
        [[2024, "Production", "overall", 20587, "t", "FAOSTAT", "raw.csv", "https://fao.example", "A", "official_value"]],
        columns=columns,
    )
    dea = pd.DataFrame(
        [[2017, "export_volume", "total", 16617.0, "MT", "DEA_EAC", "book.pdf", "https://dea.example", "", "official"]],
        columns=columns,
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        faostat.to_excel(writer, sheet_name="Annual Series", index=False, startrow=3)
        dea.to_excel(writer, sheet_name="DEA EAC Series", index=False, startrow=3)


def test_cinnamon_connector_stages_and_maps_annual_exports(tmp_path: Path) -> None:
    workbook = tmp_path / "cinnamon.xlsx"
    _write_workbook(workbook)
    connector = CinnamonConnector(workbook, tmp_path / "staging")

    raw = connector.fetch()
    repeated_raw = connector.fetch()
    assert len(raw) == len(repeated_raw) == 2
    assert raw["source_hash"].tolist() == repeated_raw["source_hash"].tolist()
    assert set(raw["source"]) == {"FAOSTAT", "DEA_EAC"}
    first_stage, second_stage = connector.stage(), connector.stage()
    assert first_stage == second_stage
    assert first_stage.exists()
    assert connector.manifest().period_start == "2017"

    fact_trade = connector.to_fact_trade(raw)
    assert fact_trade["item"].tolist() == ["cinnamon"]
    assert fact_trade["export_volume"].tolist() == [16617000.0]
    assert fact_trade["hs_code"].tolist() == ["0906"]
