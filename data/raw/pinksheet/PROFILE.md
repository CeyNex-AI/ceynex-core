# World Bank Pink Sheet — monthly commodity prices

**Source**: World Bank Commodity Markets Outlook, free, no auth, updated monthly.

```
https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx
```

Already named exactly what `ceynex/data/connectors/pinksheet.py` expects
(`CMO-Historical-Data-Monthly.xlsx`) — no renaming needed.

## Verified against the real file (checked 2026-09-24)

Ran `PinkSheetConnector` against the real downloaded workbook: `fetch()` → 800
rows, `to_fact_trade()` → 800 rows, no errors, no code changes needed. `Tea,
Colombo` monthly USD/kg price series runs 1960-01 through 2026-08 (the latest
month in the file at download time).

## To ingest

Drop the downloaded workbook into `data/raw/pinksheet/<YYYY-MM-DD>/`, then
`python -m ceynex.data.pipeline --sources pink_sheet`.
