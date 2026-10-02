# Fact provenance handoff

## Scope

`fact_trade` remains the frozen, normalized store for numeric observations.
`fact_provenance` records the source evidence for a fact without duplicating
numeric values or overwriting a competing source.  It is currently populated
by the `TEA_BOARD` and `CINNAMON` connectors.

Each provenance row links one `fact_trade.record_id` to:

- the publisher and, where available, source file and URL;
- the containing workbook filename and SHA-256 digest;
- the worksheet and 1-indexed spreadsheet row.

The relation is idempotent.  Re-ingesting an unchanged workbook keeps one
provenance row per fact/workbook/sheet/row locator.  A revised workbook refreshes
the publisher/file/URL/hash at that locator.

## Required coordinated deployment

This is a shared contract change.  Obtain the required M1/M2/M3 approval, merge
the `ceynex-contracts` change first, then merge and deploy the `ceynex-core`
change.  Do not run ad-hoc DDL on production.

On the deployed API container, after rebuilding it with both merged revisions:

```bash
docker compose exec -T api python -m ceynex.data.bootstrap
docker compose exec -T api python -m ceynex.data.pipeline --sources tea_board cinnamon
docker compose exec -T api python -m ceynex.data.pipeline --verify
docker compose exec -T api python -m eval.agriculture_validation --write-flags
```

The raw workbooks remain Git-ignored.  Transfer them through the approved
raw-data process before the pipeline command; record their SHA-256 values in
the deployment evidence.

## Verification query

Run from a PostgreSQL client connected to the deployed database:

```sql
SELECT f.source_id,
       COUNT(*) AS facts,
       COUNT(p.provenance_id) AS provenance_rows,
       COUNT(DISTINCT p.workbook_sha256) AS workbook_hashes
  FROM fact_trade AS f
  LEFT JOIN fact_provenance AS p ON p.record_id = f.record_id
 WHERE f.source_id IN ('TEA_BOARD', 'CINNAMON')
 GROUP BY f.source_id
 ORDER BY f.source_id;
```

For the current curated workbooks, expect five `CINNAMON` facts and 15
`TEA_BOARD` facts, with a matching number of provenance rows for each source.

## Neo4j boundary

This change intentionally does not create Neo4j provenance nodes.  Adding
`Observation` and `SourceDocument` labels changes the frozen graph contract and
requires separate team approval, a Cypher schema update, loader implementation,
and graph-specific tests.  PostgreSQL is authoritative for source lineage.
