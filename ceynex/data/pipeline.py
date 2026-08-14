"""Provisional data-ingestion pipeline entrypoint (SRS 3.1.7/3.1.8).

STUB — same pattern as `NullCrossValidator` (`docs/ARCHITECTURE_DELTA.md` D3):
a placeholder until the real multi-sector pipeline harness (core-systems
scope) lands, written so swapping it out later is a small diff, not a
rewrite. It only knows about apparel sources right now — `--sources all` is
currently an alias for `--sources apparel` until agriculture/macro sources
are registered somewhere and wired in here too.

Writes `fact_trade` rows to a staging parquet file
(`data/staging/fact_trade_apparel.parquet`) rather than Postgres directly —
the docker stack (`make up`) isn't assumed to be running. The connector ->
`to_fact_trade()` -> write boundary below is exactly where a Postgres writer
(`ceynex.contracts.protocols.CrossValidatorProtocol`-style injection, or a
plain `psycopg` upsert on `fact_trade`'s natural key) slots in once that's
ready — nothing upstream of that boundary should need to change.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from ceynex.data.connectors.apparel_sources import APPAREL_SOURCES

STAGING_DIR = Path("data/staging")


def run_apparel_sources() -> pd.DataFrame:
    """Run every registered apparel connector end to end and concatenate the output."""
    frames = []
    for connector in APPAREL_SOURCES:
        raw = connector.fetch()
        fact = connector.to_fact_trade(raw)
        print(f"{connector.source_id}: {len(raw)} raw rows -> {len(fact)} fact_trade rows")
        frames.append(fact)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered data source connectors.")
    parser.add_argument(
        "--sources",
        default="apparel",
        choices=["apparel", "all"],
        help="'all' currently aliases 'apparel' — no other sector is registered here yet.",
    )
    parser.parse_args()

    fact = run_apparel_sources()
    if fact.empty:
        print("No rows produced.")
        sys.exit(1)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STAGING_DIR / "fact_trade_apparel.parquet"
    fact.to_parquet(out_path, index=False)
    print(f"\nWrote {len(fact)} fact_trade rows to {out_path}")
    print(fact.groupby("source_id").size().to_string())


if __name__ == "__main__":
    main()
