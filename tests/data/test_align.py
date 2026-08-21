"""Assertions for SRS 3.1.8 — frequency alignment.

The tests that matter are about the *rule*, not the mechanics: summing a price
series or averaging an exchange rate produces a number that looks fine and is
wrong, and nothing downstream would catch it.
"""

import pandas as pd
import pytest

from ceynex.data.align import AlignmentError, aggregation_for, resample, to_annual

MONTHLY = pd.DataFrame(
    {
        "period_start": pd.date_range("2024-01-01", periods=12, freq="MS"),
        "item": ["tea"] * 12,
        "export_value_usd": [100.0] * 12,
        "export_volume": [10.0] * 12,
        "price": [10.0] * 12,
        "fx_usd_lkr": [300.0 + i for i in range(12)],
    }
)


# --- the aggregation rule ------------------------------------------------


def test_values_and_volumes_are_summed():
    annual = to_annual(MONTHLY, group_by=["item"])
    assert annual["export_value_usd"].iloc[0] == pytest.approx(1200.0)
    assert annual["export_volume"].iloc[0] == pytest.approx(120.0)


def test_prices_are_averaged_not_summed():
    """Summing twelve monthly USD/kg prices gives a number in no unit at all."""
    annual = to_annual(MONTHLY, group_by=["item"])
    assert annual["price"].iloc[0] == pytest.approx(10.0)


def test_exchange_rates_take_the_period_end_value():
    """An FX rate is a level. The annual mean of a depreciating currency is a
    rate that was never observed on any day of the year."""
    annual = to_annual(MONTHLY, group_by=["item"])
    assert annual["fx_usd_lkr"].iloc[0] == pytest.approx(311.0)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("export_value_usd", "sum"),
        ("export_volume", "sum"),
        ("price", "mean"),
        ("unit_value", "mean"),
        ("fx_usd_lkr", "last"),
        ("exchange_rate", "last"),
    ],
)
def test_the_rule_is_chosen_by_what_the_number_is(column, expected):
    assert aggregation_for(column) == expected


def test_an_explicit_rule_overrides_the_default():
    annual = to_annual(MONTHLY, group_by=["item"], rules={"price": "sum"})
    assert annual["price"].iloc[0] == pytest.approx(120.0)


# --- grouping ------------------------------------------------------------


def test_separate_series_are_kept_apart_rather_than_collapsed():
    """Resampling twenty partners without grouping yields one meaningless total."""
    two = pd.concat([MONTHLY, MONTHLY.assign(item="cinnamon", export_value_usd=50.0)])
    annual = to_annual(two, group_by=["item"])

    assert len(annual) == 2
    by_item = dict(zip(annual["item"], annual["export_value_usd"], strict=True))
    assert by_item["tea"] == pytest.approx(1200.0)
    assert by_item["cinnamon"] == pytest.approx(600.0)


# --- refusals ------------------------------------------------------------


def test_upsampling_is_refused_rather_than_interpolated():
    """Annual to monthly invents eleven observations a year that never existed."""
    annual = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01"]),
            "export_value_usd": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(AlignmentError, match="upsample"):
        resample(annual, "M")


def test_an_unknown_target_frequency_is_rejected():
    with pytest.raises(AlignmentError):
        resample(MONTHLY, "H")


def test_a_missing_period_column_is_named_in_the_error():
    with pytest.raises(AlignmentError, match="period_start"):
        resample(MONTHLY.drop(columns=["period_start"]), "A")


def test_a_frame_with_nothing_numeric_is_rejected():
    labels = pd.DataFrame({"period_start": pd.to_datetime(["2024-01-01"]), "item": ["tea"]})
    with pytest.raises(AlignmentError, match="numeric"):
        resample(labels, "A")


def test_an_empty_frame_comes_back_empty_rather_than_raising():
    assert resample(pd.DataFrame(), "A").empty


def test_a_single_period_frame_is_already_aligned():
    """One period has no gap to infer a frequency from; that is not an error."""
    single = pd.DataFrame(
        {"period_start": pd.to_datetime(["2024-01-01"]), "export_value_usd": [5.0]}
    )
    assert len(resample(single, "A")) == 1


def test_quarterly_to_annual_sums_four_quarters():
    quarterly = pd.DataFrame(
        {
            "period_start": pd.date_range("2024-01-01", periods=4, freq="QS"),
            "export_value_usd": [25.0] * 4,
        }
    )
    assert to_annual(quarterly)["export_value_usd"].iloc[0] == pytest.approx(100.0)
