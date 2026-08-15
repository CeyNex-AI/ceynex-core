import pandas as pd

from ceynex.data.cleaning import DataQualityPipeline


def test_pipeline_preserves_cleaned_records_and_emits_separate_dq_rows() -> None:
    records = pd.DataFrame(
        {
            "source_id": ["SOURCE_A", "SOURCE_B"],
            "item": ["cinnamon", "cinnamon"],
            "hs_code": ["0906", "0906"],
            "reporter_iso3": ["Sri Lanka", "LKA"],
            "reporter_m49": [None, 144],
            "partner_iso3": [None, None],
            "period_start": ["2024-01-01", "2024-01-01"],
            "frequency": ["A", "A"],
            "export_volume": [1.0, 1.2],
            "volume_unit": ["tonne", "tonne"],
        }
    )

    result = DataQualityPipeline().prepare(records)

    assert len(result.records) == 2
    assert result.records["reporter_iso3"].tolist() == ["LKA", "LKA"]
    assert result.records["export_volume"].tolist() == [1000.0, 1200.0]
    assert len(result.flags) == 1
    assert result.flags[0].severity == "material"
    assert result.dq_flag_rows()["metric"].tolist() == ["export_volume"]
