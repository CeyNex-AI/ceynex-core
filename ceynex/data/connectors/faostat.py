"""FAOSTAT agriculture connector (SRS 3.1.7)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ceynex.contracts.protocols import DataSourceConnector, SourceManifest


class FAOSTATConnector(DataSourceConnector):
    """Load locally cached Sri Lanka FAOSTAT production and producer-price CSVs."""

    source_id = "FAOSTAT"
    refresh_mode = "scheduled"

    def __init__(self, raw_dir: Path, staging_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.staging_dir = Path(staging_dir)
        self._last: pd.DataFrame | None = None
        self._fetched_at: datetime | None = None

    def fetch(self) -> pd.DataFrame:
        files = sorted(self.raw_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No FAOSTAT CSV files in {self.raw_dir}")
        frames = []
        for path in files:
            frame = pd.read_csv(path, encoding="latin1")
            frame["raw_file"] = path.name
            frame["source_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        self._last, self._fetched_at = result, datetime.now(UTC)
        return result

    def stage(self) -> Path:
        raw = self.fetch()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        output = self.staging_dir / "faostat.parquet"
        raw.to_parquet(output, index=False)
        return output

    def manifest(self) -> SourceManifest:
        if self._last is None or self._fetched_at is None:
            raise RuntimeError("Call fetch() before manifest().")
        years = pd.to_numeric(self._last.get("Year"), errors="coerce")
        return SourceManifest(
            source_id=self.source_id,
            fetched_at=self._fetched_at.isoformat(),
            row_count=len(self._last),
            period_start=str(int(years.min())) if years.notna().any() else None,
            period_end=str(int(years.max())) if years.notna().any() else None,
            frequency="mixed",
            notes={"raw_files": sorted(self._last["raw_file"].unique().tolist())},
        )
