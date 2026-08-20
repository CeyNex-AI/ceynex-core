# Agriculture data-quality handoff

## Status

`DataQualityPipeline` is integrated at the ingestion entrypoint with M2's
`UnifiedDatasetWriter` and the shared country crosswalk.  No frozen contract or
schema was changed.

## Exact writer integration

After every connector has mapped data to the frozen `fact_trade` columns, M2's
writer should use this sequence:

```python
from ceynex.data.cleaning import DataQualityPipeline

result = DataQualityPipeline().prepare(mapped_records)
writer.write_fact_trade(result.records)       # all cleaned source rows
writer.write_dq_flags(result.dq_flag_rows())  # discrepancies; never deletes rows
```

`result.records` keeps every input observation.  `result.dq_flag_rows()` has
the columns required by the `dq_flag` table except database-owned fields
(`flag_id`, `detected_at`, `resolved`).

## Country crosswalk interface

The cleaner guarantees Sri Lanka mappings only:

| Accepted reporter/partner value | Output ISO-3 | Output M49 |
| --- | --- | --- |
| `Sri Lanka`, `LK`, `LKA`, or `144` | `LKA` | `144` |

M2's general crosswalk must run before this pipeline for non-Sri-Lankan
partners and must emit `reporter_iso3`, `reporter_m49`, `partner_iso3`, and
`partner_m49` using the frozen `fact_trade` column names.

## Confirmed connector mapping

| Connector | Maps to `fact_trade` | Deliberately staged only |
| --- | --- | --- |
| Tea Board | annual total tea export volume, HS `0902`, converted MT → kg | production categories |
| Cinnamon | DEA/EAC annual total cinnamon export volume, HS `0906`, converted MT → kg | FAOSTAT production and price series; DEA/EAC price series |
| FAOSTAT | annual USD producer prices for tea, cinnamon, rubber and coconut | production, price indices, and LCU prices (the fact-trade identity cannot hold both currencies for one item/year) |
| Pink Sheet | monthly `Tea, Colombo` price in USD/kg, HS `0902` | other commodity series |

## Schema issue requiring three-way approval

The frozen `fact_trade` schema has `export_volume`, but no `production_volume`
or generic `observation_type`/`measurement_value` field.  Consequently, tea
and cinnamon production must remain in staging Parquet and cannot be written
faithfully to `fact_trade`.  Do **not** overload `export_volume` with production.

Proposed decision for M1/M2/M3: add a nullable `production_volume NUMERIC` and
`production_unit TEXT`, or add a separate `fact_production` table.  This is a
schema-contract change and must be approved by all three members before any
migration is written.
