"""Implements SRS 3.1.8 and 3.10.2 — the only writer into the unified dataset.

SAD §8 Data Layer: Postgres and Parquet are reached only through this class. No
agent and no connector opens a database connection itself.

    from ceynex.data.writer import UnifiedDatasetWriter

    writer = UnifiedDatasetWriter()
    result = writer.write(records, source_id="FAOSTAT")
    print(result.rows_written, result.dq_flags)

`records` is a DataFrame with the `fact_trade` columns; anything extra is
ignored, anything missing and non-nullable raises before a single row is written.

**Idempotency is the property that matters.** Every row carries a `source_hash`
over its identity columns, and the upsert targets `fact_trade_upsert_key`
(deviation D5 — the declared UNIQUE spans nullable columns, and Postgres treats
NULLs as distinct, so it would never match the world-partner rows). Re-running
`make ingest` produces the same row count, not double.

Cross-validation is injected (deviation D3): the writer depends on
`CrossValidatorProtocol` and defaults to `NullCrossValidator`, so M1's real
validator drops in without the writer changing.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

from ceynex.contracts import CrossValidatorProtocol, DQFlag, NullCrossValidator
from ceynex.settings import data_dir, postgres_dsn, redacted_dsn

log = logging.getLogger(__name__)

# The contract's identity columns. A row is "the same row" iff these match.
IDENTITY_COLUMNS = (
    "source_id",
    "item",
    "hs_code",
    "reporter_iso3",
    "partner_iso3",
    "period_start",
    "frequency",
)

REQUIRED_COLUMNS = (
    "source_id",
    "sector",
    "item",
    "reporter_iso3",
    "reporter_m49",
    "period_start",
    "period_end",
    "frequency",
)

WRITABLE_COLUMNS = (
    "source_id",
    "sector",
    "item",
    "hs_code",
    "reporter_iso3",
    "reporter_m49",
    "partner_iso3",
    "partner_m49",
    "period_start",
    "period_end",
    "frequency",
    "export_volume",
    "volume_unit",
    "export_value_usd",
    "price",
    "price_unit",
    "fx_usd_lkr",
    "source_hash",
)

VALID_FREQUENCIES = {"D", "W", "M", "Q", "A"}


class WriterError(RuntimeError):
    """The records did not conform to the contract. Nothing was written."""


@dataclass
class WriteResult:
    """What one `write()` did. Mirrors the `ingest_run` row it produced."""

    source_id: str
    run_id: int | None
    rows_in: int
    rows_written: int
    dq_flags: int
    parquet_path: Path | None
    status: str = "success"
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def source_hash(row: dict[str, Any]) -> str:
    """Stable digest over a row's identity plus its measured values.

    Identity alone is not enough: a revised Comtrade figure for a year we already
    hold has the same identity and a different value, and we want the update to
    be visible rather than a no-op.
    """
    parts = [str(row.get(column, "")) for column in IDENTITY_COLUMNS]
    parts += [
        f"{row.get('export_volume')}",
        f"{row.get('export_value_usd')}",
        f"{row.get('price')}",
        f"{row.get('fx_usd_lkr')}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:40]


class UnifiedDatasetWriter:
    """Writes validated records to Postgres, mirrors them to Parquet, logs the run."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        cross_validator: CrossValidatorProtocol | None = None,
        parquet_root: Path | None = None,
    ) -> None:
        self.dsn = dsn or postgres_dsn()
        # D3: M1's CrossValidator drops in here without this class changing.
        self.cross_validator: CrossValidatorProtocol = cross_validator or NullCrossValidator()
        self.parquet_root = parquet_root or (data_dir() / "parquet")

    # --- validation ------------------------------------------------------

    def prepare(self, records: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize. Raises before writing anything, never halfway."""
        if records.empty:
            return records

        missing = [c for c in REQUIRED_COLUMNS if c not in records.columns]
        if missing:
            raise WriterError(f"records are missing non-nullable columns: {missing}")

        frame = records.copy()
        for column in WRITABLE_COLUMNS:
            if column not in frame.columns:
                frame[column] = None

        for column in REQUIRED_COLUMNS:
            if frame[column].isna().any():
                count = int(frame[column].isna().sum())
                raise WriterError(f"{column} is non-nullable but {count} rows have no value")

        bad = set(frame["frequency"].unique()) - VALID_FREQUENCIES
        if bad:
            raise WriterError(f"frequency must be one of {sorted(VALID_FREQUENCIES)}, got {bad}")

        frame["source_hash"] = [source_hash(row) for row in frame.to_dict("records")]

        # Two rows with the same identity inside one batch would make the upsert
        # non-deterministic — "ON CONFLICT DO UPDATE command cannot affect row a
        # second time". Last one wins, and we say so.
        before = len(frame)
        frame = frame.drop_duplicates(subset=list(IDENTITY_COLUMNS), keep="last")
        if len(frame) < before:
            log.warning(
                "dropped %d duplicate rows within the batch (same identity, kept last)",
                before - len(frame),
            )

        return frame[list(WRITABLE_COLUMNS)]

    # --- writing ---------------------------------------------------------

    def write(self, records: pd.DataFrame, source_id: str | None = None) -> WriteResult:
        """Upsert records, persist dq_flags, mirror to Parquet, close the ingest_run."""
        resolved_source = source_id or _single_source(records)
        run_id: int | None = None
        warnings: list[str] = []

        try:
            prepared = self.prepare(records)
        except WriterError as exc:
            log.error("refusing to write: %s", exc)
            return WriteResult(resolved_source, None, len(records), 0, 0, None, "failed", str(exc))

        if prepared.empty:
            log.info("%s: nothing to write", resolved_source)
            return WriteResult(resolved_source, None, 0, 0, 0, None, "success")

        flags = list(self.cross_validator.cross_validate(prepared))

        try:
            with psycopg.connect(self.dsn) as conn:
                run_id = self._start_run(conn, resolved_source)
                # Committed immediately and on its own: if the upsert below fails
                # and rolls back, the run row must survive so the admin pipeline
                # status (SRS 3.5.4) can show a failure rather than showing
                # nothing at all.
                conn.commit()

                written = self._upsert(conn, prepared)
                self._write_flags(conn, flags)
                self._finish_run(conn, run_id, "success", written, None)
                conn.commit()
        except psycopg.Error as exc:
            log.error("write failed against %s: %s", redacted_dsn(self.dsn), exc)
            self._record_failure(run_id, resolved_source, str(exc))
            return WriteResult(
                resolved_source, run_id, len(prepared), 0, 0, None, "failed", str(exc)
            )

        parquet_path = self._mirror_to_parquet(prepared)

        log.info(
            "%s: %d rows upserted, %d dq flags, parquet -> %s",
            resolved_source,
            written,
            len(flags),
            parquet_path,
        )
        return WriteResult(
            source_id=resolved_source,
            run_id=run_id,
            rows_in=len(records),
            rows_written=written,
            dq_flags=len(flags),
            parquet_path=parquet_path,
            warnings=warnings,
        )

    # --- postgres --------------------------------------------------------

    def _start_run(self, conn: psycopg.Connection, source_id: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_run (source_id, status) VALUES (%s, 'running') RETURNING run_id",
                (source_id,),
            )
            row = cur.fetchone()
        if row is None:  # pragma: no cover - RETURNING always yields a row
            raise WriterError("could not open an ingest_run")
        return int(row[0])

    def _finish_run(
        self,
        conn: psycopg.Connection,
        run_id: int | None,
        status: str,
        rows: int,
        error: str | None,
    ) -> None:
        if run_id is None:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingest_run
                   SET finished_at = %s, status = %s, rows_written = %s, error = %s
                 WHERE run_id = %s
                """,
                (datetime.now(UTC), status, rows, error, run_id),
            )

    def _record_failure(self, run_id: int | None, source_id: str, error: str) -> None:
        """Close a failed run in its own connection — the first one is unusable.

        SRS 3.5.4 puts pipeline status on the admin page. A run stuck at
        'running' forever is worse than one that says it failed, and a run that
        vanished entirely is worse than both.
        """
        if run_id is None:
            return
        try:
            with psycopg.connect(self.dsn) as conn:
                self._finish_run(conn, run_id, "failed", 0, error[:500])
                conn.commit()
        except psycopg.Error:
            log.exception("could not record the failure of ingest_run %s", run_id)

    def _upsert(self, conn: psycopg.Connection, frame: pd.DataFrame) -> int:
        """Upsert on `fact_trade_upsert_key` (D5), skipping rows whose hash is unchanged."""
        columns = list(WRITABLE_COLUMNS)
        placeholders = ", ".join(["%s"] * len(columns))
        updatable = [c for c in columns if c not in IDENTITY_COLUMNS]
        assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in updatable)

        statement = f"""
            INSERT INTO fact_trade ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT ON CONSTRAINT fact_trade_upsert_key DO UPDATE
               SET {assignments}, ingested_at = now()
             WHERE fact_trade.source_hash IS DISTINCT FROM EXCLUDED.source_hash
        """  # noqa: S608 - column names are a module constant, values are parameters

        rows = [
            tuple(None if pd.isna(value) else value for value in record)
            for record in frame[columns].itertuples(index=False, name=None)
        ]
        with conn.cursor() as cur:
            cur.executemany(statement, rows)
        return len(rows)

    def _write_flags(self, conn: psycopg.Connection, flags: list[DQFlag]) -> None:
        """Persist cross-validation flags. SRS 3.1.8: flag, never silently discard."""
        if not flags:
            return
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO dq_flag (
                    item, hs_code, partner_iso3, period_start, metric,
                    source_a, value_a, source_b, value_b, pct_diff, severity
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
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
                    )
                    for flag in flags
                ],
            )

    # --- parquet ---------------------------------------------------------

    def _mirror_to_parquet(self, frame: pd.DataFrame) -> Path | None:
        """Partitioned by sector/item/year, per the plan.

        Postgres is the queryable store; Parquet is what the forecasting models
        read, and partitioning means a model for one item does not scan the rest.
        """
        try:
            mirror = frame.copy()
            mirror["year"] = pd.to_datetime(mirror["period_start"]).dt.year
            self.parquet_root.mkdir(parents=True, exist_ok=True)
            mirror.to_parquet(
                self.parquet_root,
                partition_cols=["sector", "item", "year"],
                index=False,
                existing_data_behavior="delete_matching",
            )
        except Exception as exc:  # noqa: BLE001 - the mirror is derived, Postgres is the record
            log.warning("parquet mirror failed (postgres write stands): %s", exc)
            return None
        return self.parquet_root


def _single_source(records: pd.DataFrame) -> str:
    if records.empty or "source_id" not in records.columns:
        return "UNKNOWN"
    sources = records["source_id"].dropna().unique()
    if len(sources) == 1:
        return str(sources[0])
    return "MIXED"
