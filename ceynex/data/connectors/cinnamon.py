"""Sri Lanka cinnamon annual fallback connector (SRS 3.1.7)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ceynex.contracts.protocols import DataSourceConnector, SourceManifest


class CinnamonConnector(DataSourceConnector):
    """Load documented annual Sri Lanka cinnamon observations.

    The source workbook deliberately keeps FAOSTAT and Department of Export
    Agriculture (DEA/EAC) series separate.  It is a transparent fallback, not
    a replacement for the unavailable Liyanage--Silva location/grade/date
    purchasing-price data.  All observations are staged; only annual total
    cinnamon exports can map safely to the frozen ``fact_trade`` contract.
    """

    source_id = "CINNAMON"
    refresh_mode = "event-driven"
    _REQUIRED_COLUMNS = (
        "year",
        "metric",
        "category",
        "value",
        "unit",
        "source",
        "source_file",
        "source_url",
        "flag",
        "dq_flags",
    )
    _SHEETS = ("Annual Series", "DEA EAC Series")

    def __init__(self, workbook_path: Path, staging_dir: Path) -> None:
        self.workbook_path = Path(workbook_path)
        self.staging_dir = Path(staging_dir)
        self._last: pd.DataFrame | None = None
        self._fetched_at: datetime | None = None

    def fetch(self) -> pd.DataFrame:
        if not self.workbook_path.exists():
            raise FileNotFoundError(self.workbook_path)

        frames = [self._read_sheet(sheet_name) for sheet_name in self._SHEETS]
        result = pd.concat(frames, ignore_index=True)
        result["source_hash"] = hashlib.sha256(self.workbook_path.read_bytes()).hexdigest()
        result = result.sort_values(
            ["year", "source", "metric", "category", "unit"], ignore_index=True
        )
        self._last, self._fetched_at = result, datetime.now(UTC)
        return result

    def stage(self) -> Path:
        raw = self.fetch()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        output = self.staging_dir / "cinnamon.parquet"
        raw.to_parquet(output, index=False)
        return output

    def manifest(self) -> SourceManifest:
        if self._last is None or self._fetched_at is None:
            raise RuntimeError("Call fetch() before manifest().")
        years = self._last["year"]
        return SourceManifest(
            source_id=self.source_id,
            fetched_at=self._fetched_at.isoformat(),
            row_count=len(self._last),
            period_start=str(int(years.min())),
            period_end=str(int(years.max())),
            frequency="A",
            notes={
                "workbook": self.workbook_path.name,
                "sources": sorted(self._last["source"].unique().tolist()),
                "scope": "fallback annual series; not Liyanage--Silva purchasing-price panel",
                "fact_trade_mapping": "DEA/EAC annual total exports only",
            },
        )

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map DEA/EAC annual total cinnamon exports to ``fact_trade``."""
        required = {"year", "metric", "category", "value", "unit", "source_hash"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"Cinnamon records missing columns: {sorted(missing)}")

        exports = raw[
            (raw["source"] == "DEA_EAC")
            & (raw["metric"] == "export_volume")
            & (raw["category"] == "total")
            & (raw["unit"] == "MT")
        ].copy()
        periods = pd.to_datetime(exports["year"].astype(str) + "-01-01")
        return pd.DataFrame(
            {
                "source_id": self.source_id,
                "sector": "agriculture",
                "item": "cinnamon",
                "hs_code": "0906",
                "reporter_iso3": "LKA",
                "reporter_m49": 144,
                "partner_iso3": None,
                "partner_m49": None,
                "period_start": periods,
                "period_end": periods + pd.offsets.YearEnd(),
                "frequency": "A",
                "export_volume": exports["value"].astype(float) * 1000,
                "volume_unit": "kg",
                "export_value_usd": None,
                "price": None,
                "price_unit": None,
                "fx_usd_lkr": None,
                "source_hash": exports["source_hash"].to_numpy(),
            }
        )

    def _read_sheet(self, sheet_name: str) -> pd.DataFrame:
        frame = pd.read_excel(self.workbook_path, sheet_name=sheet_name, header=3)
        missing = set(self._REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{sheet_name} sheet missing columns: {sorted(missing)}")
        frame = frame.loc[:, list(self._REQUIRED_COLUMNS)].copy()
        frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype("int64")
        frame["value"] = pd.to_numeric(frame["value"], errors="raise")
        frame["source_file"] = self.workbook_path.name
        return frame
