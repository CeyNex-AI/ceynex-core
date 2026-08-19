import pandas as pd
import pytest

from ceynex.data.cleaning import DataCleaner


def test_cleaner_standardizes_sri_lanka_and_preserves_original_units() -> None:
    records = pd.DataFrame(
        {
            "reporter_iso3": ["Sri Lanka"],
            "reporter_m49": [None],
            "partner_iso3": ["lka"],
            "partner_m49": [None],
            "export_volume": [2.5],
            "volume_unit": ["tonne"],
            "price": [1500.0],
            "price_unit": ["LKR/tonne"],
        }
    )

    cleaned = DataCleaner().clean(records)
    row = cleaned.iloc[0]
    assert (row["reporter_iso3"], row["reporter_m49"]) == ("LKA", 144)
    assert (row["partner_iso3"], row["partner_m49"]) == ("LKA", 144)
    assert row["export_volume"] == 2500.0
    assert (row["volume_unit"], row["original_volume_unit"]) == ("kg", "tonne")
    assert row["volume_conversion_factor"] == 1000.0
    assert row["price"] == 1.5
    assert (row["price_unit"], row["original_price_unit"]) == ("LKR/kg", "LKR/tonne")


def test_cleaner_only_forward_fills_daily_fx_for_five_days() -> None:
    records = pd.DataFrame(
        {
            "source_id": ["CBSL"] * 7,
            "item": ["tea"] * 7,
            "frequency": ["D"] * 7,
            "period_start": pd.date_range("2025-01-01", periods=7, freq="D"),
            "fx_usd_lkr": [300.0, None, None, None, None, None, None],
            "export_volume": [None, None, None, None, None, None, None],
        }
    )

    cleaned = DataCleaner().clean(records)
    assert cleaned["fx_usd_lkr"].iloc[:6].tolist() == [300.0] * 6
    assert pd.isna(cleaned["fx_usd_lkr"].iloc[6])
    assert cleaned["export_volume"].isna().all()


def test_resample_uses_mean_sum_and_period_end_fx() -> None:
    records = pd.DataFrame(
        {
            "source_id": ["SOURCE"] * 3,
            "item": ["cinnamon"] * 3,
            "frequency": ["D"] * 3,
            "period_start": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "period_end": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "price": [10.0, 13.0, 16.0],
            "price_unit": ["USD/kg"] * 3,
            "export_volume": [2.0, 3.0, 5.0],
            "volume_unit": ["kg"] * 3,
            "fx_usd_lkr": [300.0, None, 305.0],
        }
    )

    cleaned = DataCleaner().clean(records, target_frequency="M")
    assert len(cleaned) == 1
    row = cleaned.iloc[0]
    assert row["price"] == pytest.approx(13.0)
    assert row["export_volume"] == pytest.approx(10.0)
    assert row["fx_usd_lkr"] == pytest.approx(305.0)
    assert row["frequency"] == "M"
    assert row["resampling_rule"] == "price=mean; volume=sum; fx=period-end"


def test_resample_rejects_unknown_frequency() -> None:
    with pytest.raises(ValueError, match="target_frequency"):
        DataCleaner().resample(pd.DataFrame({"period_start": ["2025-01-01"]}), "X")
