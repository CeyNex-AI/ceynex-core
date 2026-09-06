"""Checks for M1's comparable-total validation rules."""

from __future__ import annotations

import pandas as pd

from eval.agriculture_validation import count_comparable_pairs, normalise_annual_totals, validate


def _records(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_validation_flags_a_material_or_severe_tea_total_difference():
    records = _records(
        [
            {
                "source_id": "TEA_BOARD",
                "item": "tea",
                "hs_code": "0902",
                "partner_iso3": None,
                "period_start": "2024-01-01",
                "export_volume": 100.0,
                "volume_unit": "kg",
            },
            {
                "source_id": "UN_COMTRADE",
                "item": "tea",
                "hs_code": "0902",
                "partner_iso3": "GBR",
                "period_start": "2024-01-01",
                "export_volume": 75.0,
                "volume_unit": "kg",
            },
            {
                "source_id": "UN_COMTRADE",
                "item": "tea",
                "hs_code": "0902",
                "partner_iso3": "USA",
                "period_start": "2024-01-01",
                "export_volume": 75.0,
                "volume_unit": "kg",
            },
        ]
    )

    pairs, flags = validate(records)

    assert pairs == 1
    assert len(flags) == 1
    assert flags[0].item == "tea"
    assert flags[0].metric == "export_volume"
    assert flags[0].severity == "severe"


def test_partner_rows_are_used_instead_of_a_comtrade_world_total():
    totals = normalise_annual_totals(
        _records(
            [
                {
                    "source_id": "UN_COMTRADE",
                    "item": "cinnamon",
                    "hs_code": "0906",
                    "partner_iso3": None,
                    "period_start": "2024-01-01",
                    "export_volume": 999.0,
                    "volume_unit": "kg",
                },
                {
                    "source_id": "UN_COMTRADE",
                    "item": "cinnamon",
                    "hs_code": "0906",
                    "partner_iso3": "MEX",
                    "period_start": "2024-01-01",
                    "export_volume": 40.0,
                    "volume_unit": "kg",
                },
                {
                    "source_id": "UN_COMTRADE",
                    "item": "cinnamon",
                    "hs_code": "0906",
                    "partner_iso3": "USA",
                    "period_start": "2024-01-01",
                    "export_volume": 60.0,
                    "volume_unit": "kg",
                },
            ]
        )
    )

    assert totals.loc[0, "export_volume"] == 100.0


def test_unpaired_records_produce_no_comparison_or_flag():
    records = _records(
        [
            {
                "source_id": "TEA_BOARD",
                "item": "tea",
                "hs_code": "0902",
                "partner_iso3": None,
                "period_start": "2024-01-01",
                "export_volume": 100.0,
                "volume_unit": "kg",
            }
        ]
    )

    pairs, flags = validate(records)

    assert pairs == 0
    assert flags == []
    assert count_comparable_pairs(pd.DataFrame()) == 0
