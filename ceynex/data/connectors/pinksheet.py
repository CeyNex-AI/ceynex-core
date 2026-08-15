"""World Bank Pink Sheet connector (SRS 3.1.7)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ceynex.contracts.protocols import DataSourceConnector, SourceManifest
from ceynex.data.connectors._snapshots import resolve_snapshot_file


class PinkSheetConnector(DataSourceConnector):
    """Load the monthly World Bank Pink Sheet price table from a cached workbook."""

    source_id = "PINK_SHEET"
    refresh_mode = "scheduled"

    def __init__(self, workbook_path: Path, staging_dir: Path) -> None:
        self.workbook_path = Path(workbook_path)
        self.staging_dir = Path(staging_dir)
        self._last: pd.DataFrame | None = None
        self._fetched_at: datetime | None = None

    def fetch(self) -> pd.DataFrame:
        workbook = resolve_snapshot_file(self.workbook_path, "CMO-Historical-Data-Monthly.xlsx")
        if not workbook.exists():
            raise FileNotFoundError(workbook)
        sheet = pd.read_excel(workbook, sheet_name="Monthly Prices", header=None)
        header_row = next(i for i, row in sheet.iterrows() if "Tea, Colombo" in row.astype(str).tolist())
        data = sheet.iloc[header_row + 1 :].copy()
        data.columns = sheet.iloc[header_row].astype(str).str.strip()
        data = data.rename(columns={data.columns[0]: "period"}).dropna(subset=["period"])
        data = data[data["period"].astype(str).str.match(r"^\d{4}M\d{2}$")].copy()
        # The workbook mixes numeric cells with ellipsis markers for missing
        # observations. Convert every commodity column to one numeric dtype so
        # Arrow can write a stable Parquet schema.
        price_columns = data.columns.drop("period")
        data[price_columns] = data[price_columns].apply(pd.to_numeric, errors="coerce")
        data = data.replace({"…": pd.NA, "â€¦": pd.NA})
        data["source_hash"] = hashlib.sha256(workbook.read_bytes()).hexdigest()
        self._last, self._fetched_at = data.reset_index(drop=True), datetime.now(UTC)
        return self._last

    def stage(self) -> Path:
        raw = self.fetch()
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        output = self.staging_dir / "pinksheet.parquet"
        raw.to_parquet(output, index=False)
        return output

    def manifest(self) -> SourceManifest:
        if self._last is None or self._fetched_at is None:
            raise RuntimeError("Call fetch() before manifest().")
        workbook = resolve_snapshot_file(self.workbook_path, "CMO-Historical-Data-Monthly.xlsx")
        return SourceManifest(self.source_id, self._fetched_at.isoformat(), len(self._last), str(self._last["period"].min()), str(self._last["period"].max()), "M", {"workbook": workbook.name})
