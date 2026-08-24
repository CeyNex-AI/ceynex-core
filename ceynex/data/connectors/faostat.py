"""FAOSTAT agriculture connector (SRS 3.1.7)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ceynex.contracts.protocols import DataSourceConnector, SourceManifest
from ceynex.data.connectors._snapshots import latest_snapshot_dir


class FAOSTATConnector(DataSourceConnector):
    """Load locally cached Sri Lanka FAOSTAT production and producer-price CSVs."""

    source_id = "FAOSTAT"
    refresh_mode = "scheduled"

    _ITEMS = {
        "Tea leaves": ("tea", "0902"),
        "Cinnamon and cinnamon-tree flowers, raw": ("cinnamon", "0906"),
        "Natural rubber in primary forms": ("rubber", "4001"),
        "Coconuts, in shell": ("coconut", "0801"),
    }

    def __init__(self, raw_dir: Path, staging_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.staging_dir = Path(staging_dir)
        self._last: pd.DataFrame | None = None
        self._fetched_at: datetime | None = None

    def fetch(self) -> pd.DataFrame:
        source_dir = latest_snapshot_dir(self.raw_dir)
        files = sorted(source_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No FAOSTAT CSV files in {source_dir}")
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

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map annual FAOSTAT producer prices; leave production staged-only.

        The frozen ``fact_trade`` contract has no production-volume measure.
        Only the USD-denominated producer-price observations are mapped.  The
        same FAOSTAT item/year also carries an LCU price, but ``fact_trade`` has
        no currency field in its identity key; mapping both would silently
        collapse one of the two.  LCU and production observations remain
        staged.  The original ``USD/tonne`` unit is retained for
        :class:`DataCleaner` to normalize to ``USD/kg``.
        """
        required = {"Area", "Area Code (M49)", "Item", "Element", "Year", "Value", "source_hash"}
        missing = required.difference(raw.columns)
        if missing:
            raise ValueError(f"FAOSTAT records missing columns: {sorted(missing)}")

        annual = (
            raw["Months"].astype("string").str.strip().eq("Annual value")
            if "Months" in raw.columns
            else pd.Series(True, index=raw.index)
        )
        prices = raw.loc[
            raw["Area"].astype("string").str.strip().eq("Sri Lanka")
            & pd.to_numeric(raw["Area Code (M49)"], errors="coerce").eq(144)
            & raw["Item"].isin(self._ITEMS)
            & raw["Element"].astype("string").str.fullmatch(r"Producer Price \(USD/tonne\)")
            & annual,
        ].copy()
        prices["year"] = pd.to_numeric(prices["Year"], errors="raise").astype("int64")
        prices["price"] = pd.to_numeric(prices["Value"], errors="raise")
        prices = prices.dropna(subset=["price"]).reset_index(drop=True)
        prices["item"] = prices["Item"].map(lambda item: self._ITEMS[str(item)][0])
        prices["hs_code"] = prices["Item"].map(lambda item: self._ITEMS[str(item)][1])
        prices["price_unit"] = prices["Element"].str.extract(r"\(([^)]+)\)", expand=False)
        periods = pd.to_datetime(prices["year"].astype(str) + "-01-01")

        return pd.DataFrame(
            {
                "source_id": self.source_id,
                "sector": "agriculture",
                "item": prices["item"].to_numpy(),
                "hs_code": prices["hs_code"].to_numpy(),
                "reporter_iso3": "LKA",
                "reporter_m49": 144,
                "partner_iso3": None,
                "partner_m49": None,
                "period_start": periods,
                "period_end": periods + pd.offsets.YearEnd(),
                "frequency": "A",
                "export_volume": None,
                "volume_unit": None,
                "export_value_usd": None,
                "price": prices["price"].to_numpy(),
                "price_unit": prices["price_unit"].to_numpy(),
                "fx_usd_lkr": None,
                "source_hash": prices["source_hash"].to_numpy(),
            }
        )
