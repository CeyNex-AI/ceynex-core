# Agriculture data pipeline

This document describes the M1 agriculture ingestion pipeline: its source
snapshots, canonical fields, transformations, quality controls, and known
limitations.  Raw source files are intentionally kept outside version control;
the relevant snapshot profiles live alongside the raw data.

## Source inventory

| Source | Cadence and available range | Raw unit(s) | Canonical unit(s) | Target fields represented in `fact_trade` | Connector and staging output |
| --- | --- | --- | --- | --- | --- |
| FAOSTAT | Annual crop observations, 1961–2024; producer-price observations, 1991–2025 (annual and monthly records may be present) | tonnes; LCU/tonne; USD/tonne; price index | kg for applicable volumes; USD/kg for mapped USD prices; LKR/kg where an LKR price is cleaned | `producer_price` only: annual `Producer Price (USD/tonne)` observations for tea, cinnamon, rubber, and coconut | `FAOSTATConnector` → `data/staging/faostat.parquet` |
| World Bank Pink Sheet | Monthly, 1960-01–2026-07 in the supplied historical workbook | USD/kg for `Tea, Colombo`; other workbook columns retain their source units | USD/kg for mapped Tea, Colombo prices | Monthly tea `producer_price` (`Tea, Colombo`) | `PinkSheetConnector` → `data/staging/pinksheet.parquet` |
| Sri Lanka Tea Board | Annual, 2011–2025 | metric tonnes (MT) for production and exports | kg for mapped export volume | Annual total tea `export_volume` (HS 0902) | `TeaBoardConnector` → `data/staging/teaboard.parquet` |
| Cinnamon / DEA-EAC | Annual fallback workbook, 2011–2025; DEA/EAC export series, 2013–2017 | tonnes/MT, hectares, kg/ha, LKR/kg, USD/tonne, and price indexes | kg for mapped export volume; LKR/kg or USD/kg for cleaned price fields | Annual DEA/EAC total cinnamon `export_volume` (HS 0906) | `CinnamonConnector` → `data/staging/cinnamon.parquet` |

Each connector stages the full useful raw series with its source metadata. Its
`to_fact_trade()` mapping is deliberately narrower: only observations that fit
the current trade-fact schema are loaded into `fact_trade`. In particular,
FAOSTAT crop-production and Tea Board production data remain staged for later
use rather than being incorrectly labelled as exports.

## Raw snapshots and staging

The expected raw-data layout is:

```text
data/raw/
  faostat/2026-08-15/
  pinksheet/2026-08-15/
  tea_board/2026-08-15/
  cinnamon/2026-08-15/
```

The dated directory is part of source provenance. Connector manifests record
the source identifier, fetch time, source files, source hash, row count,
period range, and frequency. The pipeline writes reproducible intermediate
Parquet files to `data/staging/` as listed above.

## Country standardization

All recognised forms of Sri Lanka are standardised to:

| Source country value | ISO alpha-3 | UN M49 |
| --- | --- | --- |
| Sri Lanka / Sri Lanka (ex-Ceilon) / LKA | `LKA` | `144` |

The original country values are retained where supplied, while the standardised
codes are used for integration and comparison.

## Units and conversion provenance

The cleaner preserves `original_volume_unit`, `original_price_unit`, and the
associated conversion-factor columns. This means a user can always trace a
canonical value to its original reported unit.

| Input measurement | Canonical measurement | Conversion factor | Rule |
| --- | --- | ---: | --- |
| tonnes, tonnes (`t`), or metric tonnes (`MT`) | kg | 1,000 | `kg = tonnes × 1000` |
| kg | kg | 1 | no conversion |
| USD/tonne | USD/kg | 0.001 | `USD/kg = USD/tonne ÷ 1000` |
| USD/kg | USD/kg | 1 | no conversion |
| LKR/tonne or LCU/tonne | LKR/kg | 0.001 | per-tonne price divided by 1,000 |
| LKR/kg | LKR/kg | 1 | no conversion |

The cleaner does **not** convert LKR-denominated prices to USD: it normalises
the mass denominator to kg while preserving currency. Any currency conversion
must use a documented FX series rather than an implicit assumption.

## Resampling rules

When a target frequency differs from a source frequency, the pipeline uses the
following metric-specific rules:

| Data type | Resampling rule |
| --- | --- |
| Prices | arithmetic mean within the target period |
| Volumes | sum within the target period |
| FX | last non-null observation in the period (period-end) |

The output records the applied `resampling_rule` so the aggregation remains
auditable.

## Missing-value policy

- Volume values are never interpolated or fabricated; a missing component keeps
  the aggregate missing and is flagged for follow-up.
- FX may be forward-filled for at most five daily observations. Longer gaps are
  not filled and are flagged.
- Price gaps are retained as missing unless a documented source-specific rule
  says otherwise.

## Cross-source validation and DQ flags

`CrossValidator` compares overlapping records by item, partner, period, and
metric. It preserves both records and creates a `DQFlag`; it never deletes or
replaces a source value because another source disagrees.

| Absolute percentage difference | Severity |
| --- | --- |
| less than 5% | minor |
| 5% through 20% | material |
| greater than 20% | severe |

The Agriculture Agent checks the requested item, metric, and period for
material or severe flags. When one applies, it keeps the reported figure,
adds an explicit discrepancy assumption, gives readable evidence naming both
sources and the percentage difference, and reduces confidence using the
existing DQ penalty. Minor flags remain available in the data-quality store but
do not add a material/severe warning to an answer.

## Known limitation: production schema

`fact_trade` has no dedicated `production_volume` field. Production records
from FAOSTAT, Tea Board, and the cinnamon fallback are therefore staged only.
They must not be inserted into `export_volume`; a schema extension is required
before production can be represented faithfully in the unified fact table.

## Related documentation

- [Agriculture pipeline handoff](AGRICULTURE_PIPELINE_HANDOFF.md) — raw
  snapshot availability, connector mappings, and deployment prerequisites.
- [M1 model deployment handoff](https://github.com/CeyNex-AI/ceynex-core/blob/0c0ffcd/docs/M1_MODEL_DEPLOYMENT_HANDOFF.md)
  — ignored model artifacts, ignored raw snapshots, and deployment steps. The
  linked handoff is commit `0c0ffcd`; merge that documentation commit into this
  branch before changing this permanent link back to a repository-relative one.
- [Evaluation](EVALUATION.md) — agricultural model data sufficiency,
  rolling-origin backtests, and interval evaluation.
