from pathlib import Path

from ceynex.data.connectors.faostat import FAOSTATConnector


def test_faostat_connector_is_repeatable_with_offline_csv_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "faostat"
    snapshot = raw_dir / "2026-08-15"
    snapshot.mkdir(parents=True)
    (snapshot / "producer_prices.csv").write_text(
        "Year,Area,Item,Value\n2024,Sri Lanka,Cinnamon,3023119.4\n",
        encoding="utf-8",
    )
    (snapshot / "production.csv").write_text(
        "Year,Area,Item,Value\n2024,Sri Lanka,Tea,264122\n",
        encoding="utf-8",
    )
    connector = FAOSTATConnector(raw_dir, tmp_path / "staging")

    first = connector.fetch()
    second = connector.fetch()

    assert len(first) == len(second) == 2
    assert first["source_hash"].tolist() == second["source_hash"].tolist()
    first_stage, second_stage = connector.stage(), connector.stage()
    assert first_stage == second_stage
    assert first_stage.exists()
    assert connector.manifest().row_count == 2
    assert connector.manifest().period_start == "2024"
