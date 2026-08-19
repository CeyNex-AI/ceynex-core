import pandas as pd

from ceynex.data.cleaning import CrossValidator


def test_cross_validator_flags_conflict_without_dropping_records() -> None:
    records = pd.DataFrame(
        {
            "source_id": ["FAOSTAT", "DEA_EAC", "FAOSTAT"],
            "item": ["cinnamon", "cinnamon", "tea"],
            "hs_code": ["0906", "0906", "0902"],
            "partner_iso3": [None, None, None],
            "period_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "export_volume": [1000.0, 1150.0, 100.0],
        }
    )
    original = records.copy(deep=True)

    flags = CrossValidator().cross_validate(records)

    assert len(records) == 3
    pd.testing.assert_frame_equal(records, original)
    assert len(flags) == 1
    flag = flags[0]
    assert (flag.item, flag.metric, flag.source_a, flag.source_b) == (
        "cinnamon", "export_volume", "FAOSTAT", "DEA_EAC"
    )
    assert flag.pct_diff == 15.0
    assert flag.severity == "material"


def test_cross_validator_uses_required_severity_thresholds() -> None:
    records = pd.DataFrame(
        {
            "source_id": ["A", "B", "C", "D", "E", "F"],
            "item": ["tea"] * 6,
            "period_start": ["2024-01-01"] * 6,
            "price": [100.0, 104.0, 100.0, 120.0, 100.0, 121.0],
        }
    )

    flags = CrossValidator().cross_validate(records)
    by_pair = {(flag.source_a, flag.source_b): flag.severity for flag in flags}
    assert by_pair[("A", "B")] == "minor"
    assert by_pair[("C", "D")] == "material"
    assert by_pair[("E", "F")] == "severe"


def test_cross_validator_exports_dq_flag_table_rows() -> None:
    records = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "item": ["tea", "tea"],
            "period_start": ["2024-01-01", "2024-01-01"],
            "price": [100.0, 105.0],
        }
    )

    rows = CrossValidator.to_frame(CrossValidator().cross_validate(records))
    assert rows.columns.tolist() == [
        "item", "hs_code", "partner_iso3", "period_start", "metric",
        "source_a", "value_a", "source_b", "value_b", "pct_diff", "severity",
    ]
    assert rows.iloc[0]["severity"] == "material"
