"""Sri Lanka Tea Board annual production and export connector (SRS 3.1.7)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ceynex.contracts.protocols import DataSourceConnector, SourceManifest
from ceynex.data.connectors._snapshots import resolve_snapshot_file


class TeaBoardConnector(DataSourceConnector):
    """Load the curated annual Sri Lanka Tea production and export workbook.

    The source workbook has ``Production`` and ``Exports`` sheets with one row
    per year. ``fetch`` reshapes them into auditable long-form observations;
    ``to_fact_trade`` publishes only total export volume because the frozen
    ``fact_trade`` schema has no production-volume field.
    """

    source_id = "TEA_BOARD"
    refresh_mode = "event-driven"

    _PRODUCTION_COLUMNS = (
        "orthodox_mt",
        "ctc_mt",
        "green_mt",
        "total_production_mt",
    )
    _EXPORT_COLUMNS = (
        "bulk_mt",
        "tea_in_packets_mt",
        "tea_bags_mt",
        "instant_tea_mt",
        "green_tea_mt",
        "total_exports_mt",
    )
    _EXPECTED_YEARS = range(2011, 2026)
    _HEADER_ROW = 3

    def __init__(self, workbook_path: Path, staging_dir: Path) -> None:
        self.workbook_path = Path(workbook_path)
        self.staging_dir = Path(staging_dir)
        self._last: pd.DataFrame | None = None
        self._fetched_at: datetime | None = None

    def fetch(self) -> pd.DataFrame:
        workbook = resolve_snapshot_file(
            self.workbook_path, "tea_annual_production_exports_2011_2025.xlsx"
        )
        if not workbook.exists():
            raise FileNotFoundError(workbook)

        production = self._read_sheet("Production", self._PRODUCTION_COLUMNS)
        exports = self._read_sheet("Exports", self._EXPORT_COLUMNS)
        workbook_sha256 = hashlib.sha256(workbook.read_bytes()).hexdigest()

        frames = [
            self._to_long(production, "production", self._PRODUCTION_COLUMNS),
            self._to_long(exports, "export", self._EXPORT_COLUMNS),
        ]
        result = pd.concat(frames, ignore_index=True)
        result["workbook_file"] = workbook.name
        result["workbook_sha256"] = workbook_sha256
        result["source_hash"] = workbook_sha256
        result = result.sort_values(["year", "metric", "category"], ignore_index=True)
        self._last, self._fetched_at = result, datetime.now(UTC)
        return result

    def stage(self) -> Path:
        raw = self.fetch()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        output = self.staging_dir / "teaboard.parquet"
        raw.to_parquet(output, index=False)
        return output

    def manifest(self) -> SourceManifest:
        if self._last is None or self._fetched_at is None:
            raise RuntimeError("Call fetch() before manifest().")
        years = self._last["year"]
        available_years = {int(year) for year in years.unique()}
        return SourceManifest(
            source_id=self.source_id,
            fetched_at=self._fetched_at.isoformat(),
            row_count=len(self._last),
            period_start=str(int(years.min())),
            period_end=str(int(years.max())),
            frequency="A",
            notes={
                "workbook": resolve_snapshot_file(
                    self.workbook_path, "tea_annual_production_exports_2011_2025.xlsx"
                ).name,
                "metrics": sorted(self._last["metric"].unique().tolist()),
                "production_mapping": "staged only; fact_trade lacks production_volume",
                "missing_years": [
                    year for year in self._EXPECTED_YEARS if year not in available_years
                ],
            },
        )

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map annual total tea exports to the frozen fact_trade schema."""
        required = {
            "year",
            "metric",
            "category",
            "value_mt",
            "source",
            "source_hash",
            "workbook_file",
            "workbook_sha256",
            "source_sheet",
            "source_row",
        }
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"Tea Board records missing columns: {sorted(missing)}")

        exports = raw[
            (raw["metric"] == "export")
            & (raw["category"] == "total")
            & raw["value_mt"].notna()
        ].copy()
        periods = pd.to_datetime(exports["year"].astype(str) + "-01-01")
        return pd.DataFrame(
            {
                "source_id": self.source_id,
                "sector": "agriculture",
                "item": "tea",
                "hs_code": "0902",
                "reporter_iso3": "LKA",
                "reporter_m49": 144,
                "partner_iso3": None,
                "partner_m49": None,
                "period_start": periods,
                "period_end": periods + pd.offsets.YearEnd(),
                "frequency": "A",
                "export_volume": exports["value_mt"] * 1000,
                "volume_unit": "kg",
                "export_value_usd": None,
                "price": None,
                "price_unit": None,
                "fx_usd_lkr": None,
                "source_hash": exports["source_hash"].to_numpy(),
                "publisher": exports["source"].to_numpy(),
                "source_file": None,
                "source_url": None,
                "workbook_file": exports["workbook_file"].to_numpy(),
                "workbook_sha256": exports["workbook_sha256"].to_numpy(),
                "source_sheet": exports["source_sheet"].to_numpy(),
                "source_row": exports["source_row"].to_numpy(),
            }
        )

    def _read_sheet(self, sheet_name: str, value_columns: tuple[str, ...]) -> pd.DataFrame:
        workbook = resolve_snapshot_file(
            self.workbook_path, "tea_annual_production_exports_2011_2025.xlsx"
        )
        frame = pd.read_excel(workbook, sheet_name=sheet_name, header=self._HEADER_ROW)
        required = {"year", *value_columns, "source", "dq_flags"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{sheet_name} sheet missing columns: {sorted(missing)}")
        frame = frame[["year", *value_columns, "source", "dq_flags"]].copy()
        frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype("int64")
        # Excel is 1-indexed; header=3 means the first data row is row 5.
        frame["source_sheet"] = sheet_name
        frame["source_row"] = frame.index.to_series().add(self._HEADER_ROW + 2).to_numpy()
        return frame

    @staticmethod
    def _to_long(frame: pd.DataFrame, metric: str, value_columns: tuple[str, ...]) -> pd.DataFrame:
        long = frame.melt(
            id_vars=["year", "source", "dq_flags", "source_sheet", "source_row"],
            value_vars=list(value_columns),
            var_name="category",
            value_name="value_mt",
        )
        long = long.dropna(subset=["value_mt"]).copy()
        long["category"] = (
            long["category"]
            .str.removesuffix("_production_mt")
            .str.removesuffix("_exports_mt")
            .str.removesuffix("_mt")
            .replace({"total_production": "total", "total_exports": "total"})
        )
        long.insert(1, "metric", metric)
        return long
