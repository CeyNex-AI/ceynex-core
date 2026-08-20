from pathlib import Path

from ceynex.data.cleaning import DataCleaner
from ceynex.data.connectors.faostat import FAOSTATConnector


def test_faostat_connector_is_repeatable_with_offline_csv_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "faostat"
    snapshot = raw_dir / "2026-08-15"
    snapshot.mkdir(parents=True)
    (snapshot / "producer_prices.csv").write_text(
        "Area Code (M49),Area,Item,Element,Year,Value\n"
        "144,Sri Lanka,\"Cinnamon and cinnamon-tree flowers, raw\",Producer Price (LCU/tonne),2024,3023119.4\n"
        "144,Sri Lanka,\"Cinnamon and cinnamon-tree flowers, raw\",Producer Price (USD/tonne),2024,9912.3\n"
        "144,Sri Lanka,\"Cinnamon and cinnamon-tree flowers, raw\",Producer Price Index (2014-2016 = 100),2024,130.0\n",
        encoding="utf-8",
    )
    (snapshot / "production.csv").write_text(
        "Year,Area,Item,Value\n2024,Sri Lanka,Tea,264122\n",
        encoding="utf-8",
    )
    connector = FAOSTATConnector(raw_dir, tmp_path / "staging")

    first = connector.fetch()
    second = connector.fetch()

    assert len(first) == len(second) == 4
    assert first["source_hash"].tolist() == second["source_hash"].tolist()
    first_stage, second_stage = connector.stage(), connector.stage()
    assert first_stage == second_stage
    assert first_stage.exists()
    assert connector.manifest().row_count == 4
    assert connector.manifest().period_start == "2024"

    fact_trade = connector.to_fact_trade(first)
    assert len(fact_trade) == 1
    assert fact_trade.loc[0, "item"] == "cinnamon"
    assert fact_trade.loc[0, "hs_code"] == "0906"
    assert fact_trade.loc[0, "price"] == 9912.3
    assert fact_trade.loc[0, "price_unit"] == "USD/tonne"
    assert fact_trade.loc[0, "export_volume"] is None

    cleaned = DataCleaner().clean(fact_trade)
    assert cleaned.loc[0, "price"] == 9.9123
    assert cleaned.loc[0, "price_unit"] == "USD/kg"
    assert cleaned.loc[0, "original_price_unit"] == "USD/tonne"
