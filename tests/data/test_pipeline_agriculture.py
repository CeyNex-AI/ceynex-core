from __future__ import annotations

import pandas as pd

from ceynex.contracts import SourceManifest
from ceynex.data import pipeline
from ceynex.data.cleaning import CrossValidator
from ceynex.data.writer import WriteResult


class _TeaBoardLikeConnector:
    source_id = "TEA_BOARD"

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame({"raw_value": [1]})

    def to_fact_trade(self, _raw: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "source_id": [self.source_id],
                "sector": ["agriculture"],
                "item": ["tea"],
                "hs_code": ["0902"],
                "reporter_iso3": ["Sri Lanka"],
                "reporter_m49": [None],
                "partner_iso3": [None],
                "partner_m49": [None],
                "period_start": ["2025-01-01"],
                "period_end": ["2025-12-31"],
                "frequency": ["A"],
                "export_volume": [2.5],
                "volume_unit": ["tonne"],
                "source_hash": ["raw-hash"],
            }
        )

    def manifest(self) -> SourceManifest:
        return SourceManifest(source_id=self.source_id, fetched_at="2026-01-01T00:00:00+00:00", row_count=1)


class _RecordingWriter:
    def __init__(self) -> None:
        self.records: pd.DataFrame | None = None

    def write(self, records: pd.DataFrame, *, source_id: str) -> WriteResult:
        self.records = records.copy()
        return WriteResult(source_id, None, len(records), len(records), 0, None)


def test_agriculture_connector_output_is_cleaned_before_writing(monkeypatch) -> None:
    monkeypatch.setitem(pipeline.CONNECTORS, "tea_board", lambda **_kwargs: _TeaBoardLikeConnector())
    writer = _RecordingWriter()

    result = pipeline.run_source("tea_board", writer)  # type: ignore[arg-type]

    assert result.status == "success"
    assert writer.records is not None
    assert writer.records.loc[0, "reporter_iso3"] == "LKA"
    assert writer.records.loc[0, "reporter_m49"] == 144
    assert writer.records.loc[0, "export_volume"] == 2500.0
    assert writer.records.loc[0, "volume_unit"] == "kg"
    assert writer.records.loc[0, "original_volume_unit"] == "tonne"


def test_pipeline_injects_m1_cross_validator_into_writer(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def writer_factory(*, cross_validator: object) -> object:
        captured["validator"] = cross_validator
        return object()

    def fake_run_source(_name: str, _writer: object, **_kwargs: object) -> WriteResult:
        return WriteResult("TEA_BOARD", None, 0, 0, 0, None)

    monkeypatch.setattr(pipeline, "UnifiedDatasetWriter", writer_factory)
    monkeypatch.setattr(pipeline, "run_source", fake_run_source)
    monkeypatch.setattr(pipeline, "_print_counts", lambda: 0)

    assert pipeline.main(["--sources", "tea_board"]) == 0
    assert isinstance(captured["validator"], CrossValidator)
