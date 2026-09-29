"""Audit the curated agriculture workbooks selected by the ingestion pipeline.

This is read-only. The workbooks themselves live outside version control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ceynex.data.connectors._snapshots import resolve_snapshot_file
from ceynex.data.connectors.cinnamon import CinnamonConnector
from ceynex.data.connectors.teaboard import TeaBoardConnector
from ceynex.data.pipeline import _agriculture_raw_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(raw_dir: Path) -> dict[str, object]:
    tea_path = resolve_snapshot_file(
        raw_dir / "tea_board", "tea_annual_production_exports_2011_2025.xlsx"
    )
    cinnamon_path = resolve_snapshot_file(
        raw_dir / "cinnamon", "cinnamon_annual_fallback_2011_2025.xlsx"
    )
    tea = TeaBoardConnector(tea_path, raw_dir / "_unused_staging")
    cinnamon = CinnamonConnector(cinnamon_path, raw_dir / "_unused_staging")
    tea_raw = tea.fetch()
    cinnamon_raw = cinnamon.fetch()
    tea_facts = tea.to_fact_trade(tea_raw)
    cinnamon_facts = cinnamon.to_fact_trade(cinnamon_raw)

    panel = pd.read_excel(tea_path, sheet_name="Annual Panel", header=3)
    exports = pd.read_excel(tea_path, sheet_name="Exports", header=3)
    production = pd.read_excel(tea_path, sheet_name="Production", header=3)
    years = set(range(2011, 2026))
    for name, frame in (("panel", panel), ("exports", exports), ("production", production)):
        if set(frame["year"]) != years or frame["year"].duplicated().any():
            raise ValueError(f"tea {name} must contain one row per year from 2011 to 2025")
    for name, sheet, column in (
        ("exports", exports, "total_exports_mt"),
        ("production", production, "total_production_mt"),
    ):
        joined = panel[["year", column]].merge(
            sheet[["year", column]], on="year", suffixes=("_panel", "_sheet"), validate="one_to_one"
        )
        if not joined[f"{column}_panel"].equals(joined[f"{column}_sheet"]):
            raise ValueError(f"tea {name} totals disagree with Annual Panel")
    source_notes = pd.read_excel(tea_path, sheet_name="Sources and Notes", header=3)
    if source_notes["Source URL"].dropna().empty:
        raise ValueError("tea source URLs are missing")
    if panel[["production_source", "export_source"]].isna().any().any():
        raise ValueError("tea annual panel has missing source attribution")

    dea = pd.read_excel(cinnamon_path, sheet_name="DEA EAC Series", header=3)
    mapped_dea = dea.loc[
        dea["metric"].eq("export_volume") & dea["category"].eq("total")
    ]
    if set(mapped_dea["year"]) != set(range(2013, 2018)):
        raise ValueError("cinnamon export years must be 2013 through 2017")
    if mapped_dea[["source", "source_file", "source_url"]].isna().any().any():
        raise ValueError("cinnamon export provenance is incomplete")
    if set(mapped_dea["source"]) != {"DEA_EAC"}:
        raise ValueError("cinnamon export source must remain DEA_EAC")
    if len(tea_facts) != 15 or len(cinnamon_facts) != 5:
        raise ValueError("unexpected number of mapped agriculture source rows")

    return {
        "tea": {
            "path": str(tea_path),
            "sha256": _sha256(tea_path),
            "years": [int(panel["year"].min()), int(panel["year"].max())],
            "annual_rows": len(panel),
            "staged_rows": len(tea_raw),
            "mapped_export_rows": len(tea_facts),
            "export_publishers": sorted(panel["export_source"].unique().tolist()),
            "source_urls": sorted(source_notes["Source URL"].dropna().unique().tolist()),
        },
        "cinnamon": {
            "path": str(cinnamon_path),
            "sha256": _sha256(cinnamon_path),
            "staged_rows": len(cinnamon_raw),
            "mapped_export_rows": len(cinnamon_facts),
            "export_years": sorted(mapped_dea["year"].astype(int).tolist()),
            "source_urls": sorted(mapped_dea["source_url"].unique().tolist()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.raw_dir or _agriculture_raw_dir())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
