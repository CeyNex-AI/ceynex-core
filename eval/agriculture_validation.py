"""Validate comparable M1 agriculture trade totals without changing source facts.

Only annual export-volume totals with the same commodity definition are compared:
Tea Board tea against UN Comtrade tea, and DEA/EAC cinnamon against UN Comtrade
cinnamon. The input facts are already standardised to kilograms by the
connectors and cleaner. FAOSTAT producer prices and Pink Sheet auction prices
are deliberately excluded because their price concepts differ from export
volumes; EDB currently has no agriculture fact rows; and WITS is deferred.

    python -m eval.agriculture_validation
    python -m eval.agriculture_validation --write-flags
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

from ceynex.contracts.protocols import DQFlag
from ceynex.data.cleaning import CrossValidator
from ceynex.settings import postgres_dsn

LOCAL_SOURCES = {"tea": "TEA_BOARD", "cinnamon": "CINNAMON"}
COMTRADE = "UN_COMTRADE"
REQUESTED_SOURCES = ("FAOSTAT", "TEA_BOARD", "CINNAMON", "EDB", COMTRADE, "WITS")
DEFERRED_SOURCES = {
    "WITS": "WITS tariff ingestion is deliberately deferred; no WITS facts are compared or claimed.",
}
NON_COMPARABLE_SOURCES = {
    "FAOSTAT": "Current fact_trade rows are producer prices, not export-volume totals.",
    "EDB": "The configured EDB connector currently contributes apparel data, not agriculture fact rows.",
}


@dataclass(frozen=True)
class ValidationResult:
    """The reproducible result of an agriculture cross-source validation run."""

    inventory: dict[str, int]
    comparable_pairs: int
    flags: list[DQFlag]
    inserted_flags: int

    def as_dict(self) -> dict[str, Any]:
        by_severity = {severity: 0 for severity in ("minor", "material", "severe")}
        for flag in self.flags:
            by_severity[flag.severity] = by_severity.get(flag.severity, 0) + 1
        return {
            "source_inventory": self.inventory,
            "comparable_annual_pairs": self.comparable_pairs,
            "flags": len(self.flags),
            "flags_by_severity": by_severity,
            "material_or_severe_flags_written": self.inserted_flags,
            "deferred_sources": DEFERRED_SOURCES,
            "non_comparable_sources": NON_COMPARABLE_SOURCES,
        }


def load_annual_export_volumes(dsn: str | None = None) -> pd.DataFrame:
    """Read only normalised annual agriculture export-volume facts from PostgreSQL."""
    statement = """
        SELECT source_id, item, hs_code, partner_iso3, period_start, export_volume, volume_unit
        FROM fact_trade
        WHERE lower(sector) = 'agriculture'
          AND frequency = 'A'
          AND export_volume IS NOT NULL
          AND source_id = ANY(%s)
        ORDER BY source_id, item, period_start, partner_iso3 NULLS LAST
    """
    source_ids = [*LOCAL_SOURCES.values(), COMTRADE]
    with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute(statement, (source_ids,))
        rows = cur.fetchall()
    return pd.DataFrame(
        rows,
        columns=[
            "source_id",
            "item",
            "hs_code",
            "partner_iso3",
            "period_start",
            "export_volume",
            "volume_unit",
        ],
    )


def source_inventory(dsn: str | None = None) -> dict[str, int]:
    """Count agriculture fact rows for every requested source, including absent ones."""
    statement = """
        SELECT source_id, count(*)
        FROM fact_trade
        WHERE lower(sector) = 'agriculture'
          AND source_id = ANY(%s)
        GROUP BY source_id
    """
    with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
        cur.execute(statement, (list(REQUESTED_SOURCES),))
        present = {str(source): int(count) for source, count in cur.fetchall()}
    return {source: present.get(source, 0) for source in REQUESTED_SOURCES}


def normalise_annual_totals(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate partner rows to national totals while avoiding world-row double counts.

    A source with partner-level rows (Comtrade) is summed over partners. A source
    containing only a null partner is already a national total (Tea Board and
    DEA/EAC cinnamon). All inputs must already be in kilograms; conversion is
    owned by the connector/cleaner and is never guessed here.
    """
    columns = ["source_id", "item", "hs_code", "partner_iso3", "period_start", "export_volume", "volume_unit"]
    if records.empty:
        return pd.DataFrame(columns=["source_id", "item", "hs_code", "partner_iso3", "period_start", "export_volume"])
    missing = set(columns).difference(records.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")

    frame = records.copy()
    frame["period_start"] = pd.to_datetime(frame["period_start"], errors="raise")
    frame["export_volume"] = pd.to_numeric(frame["export_volume"], errors="raise")
    units = set(frame["volume_unit"].dropna().astype(str))
    if units != {"kg"}:
        raise ValueError(f"validation expects connector-normalised kg volumes; found {sorted(units)}")
    frame["year"] = frame["period_start"].dt.year

    totals: list[dict[str, Any]] = []
    for (source_id, item, hs_code, year), group in frame.groupby(
        ["source_id", "item", "hs_code", "year"], dropna=False, sort=True
    ):
        partner_rows = group.loc[group["partner_iso3"].notna()]
        usable = partner_rows if not partner_rows.empty else group
        totals.append(
            {
                "source_id": str(source_id),
                "item": str(item),
                "hs_code": None if pd.isna(hs_code) else str(hs_code),
                "partner_iso3": None,
                "period_start": pd.Timestamp(year=int(year), month=1, day=1),
                "export_volume": float(usable["export_volume"].sum()),
            }
        )
    return pd.DataFrame(totals)


def comparable_records(totals: pd.DataFrame) -> pd.DataFrame:
    """Keep only the two documented local-source/Comtrade comparison pairs."""
    selected: list[pd.DataFrame] = []
    for item, local_source in LOCAL_SOURCES.items():
        subset = totals.loc[
            (totals["item"] == item) & (totals["source_id"].isin([local_source, COMTRADE]))
        ]
        if subset["source_id"].nunique() == 2:
            selected.append(subset)
    if not selected:
        return pd.DataFrame(columns=totals.columns)
    return pd.concat(selected, ignore_index=True)


def count_comparable_pairs(records: pd.DataFrame) -> int:
    """Count annual item-year pairs for which both approved sources are present."""
    if records.empty:
        return 0
    return sum(
        group["source_id"].nunique() == 2
        for _, group in records.groupby(["item", "hs_code", "period_start"], dropna=False)
    )


def validate(records: pd.DataFrame) -> tuple[int, list[DQFlag]]:
    """Return every disagreement; callers persist only material/severe flags."""
    totals = normalise_annual_totals(records)
    comparable = comparable_records(totals)
    if comparable.empty:
        return 0, []
    pairs = count_comparable_pairs(comparable)
    flags = CrossValidator(metrics=("export_volume",)).cross_validate(comparable)
    return pairs, flags


def persist_material_or_severe(flags: list[DQFlag], dsn: str | None = None) -> int:
    """Insert new material/severe flags once, preserving facts and prior findings."""
    candidates = [flag for flag in flags if flag.severity in {"material", "severe"}]
    if not candidates:
        return 0
    inserted = 0
    duplicate_check = """
        SELECT 1 FROM dq_flag
         WHERE item = %s AND metric = %s AND source_a = %s AND source_b = %s
           AND period_start = %s
           AND round(value_a, 6) = round(%s::numeric, 6)
           AND round(value_b, 6) = round(%s::numeric, 6)
         LIMIT 1
    """
    insert = """
        INSERT INTO dq_flag (
            item, hs_code, partner_iso3, period_start, metric,
            source_a, value_a, source_b, value_b, pct_diff, severity
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with psycopg.connect(dsn or postgres_dsn()) as conn, conn.cursor() as cur:
        for flag in candidates:
            identity = (
                flag.item,
                flag.metric,
                flag.source_a,
                flag.source_b,
                flag.period_start,
                flag.value_a,
                flag.value_b,
            )
            cur.execute(duplicate_check, identity)
            if cur.fetchone() is not None:
                continue
            cur.execute(
                insert,
                (
                    flag.item,
                    flag.hs_code,
                    flag.partner_iso3,
                    flag.period_start,
                    flag.metric,
                    flag.source_a,
                    flag.value_a,
                    flag.source_b,
                    flag.value_b,
                    flag.pct_diff,
                    flag.severity,
                ),
            )
            inserted += 1
        conn.commit()
    return inserted


def run(*, write_flags: bool, dsn: str | None = None) -> ValidationResult:
    """Run the validation and optionally persist only material/severe findings."""
    inventory = source_inventory(dsn)
    pairs, flags = validate(load_annual_export_volumes(dsn))
    inserted = persist_material_or_severe(flags, dsn) if write_flags else 0
    return ValidationResult(inventory=inventory, comparable_pairs=pairs, flags=flags, inserted_flags=inserted)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate comparable M1 agriculture trade totals.")
    parser.add_argument("--write-flags", action="store_true", help="Persist only material/severe dq_flag rows.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/results/agriculture_validation.json"),
        help="Local JSON record path; eval/results is gitignored.",
    )
    args = parser.parse_args(argv)

    result = run(write_flags=args.write_flags)
    payload = result.as_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
