# FAOSTAT — Producer Prices (tea, cinnamon, rubber, coconut)

**Source**: FAOSTAT bulk download, free, no auth.

```
https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip
```

Use the **`(Normalized)`** file. The default `Prices_E_All_Data.zip` is
wide-format (one column per year, `Y1991, Y1992, ...`) and this connector
cannot read it. The normalized file is long-format (`Year`, `Value`, `Months`
columns) and matches what `ceynex/data/connectors/faostat.py` expects.

## Real data confirmed present (checked 2026-09-24)

Filtering the normalized CSV to `Area == "Sri Lanka"`, `Element == "Producer
Price (USD/tonne)"`, `Months == "Annual value"`:

| Item | Years covered | Gaps |
|---|---|---|
| Cinnamon and cinnamon-tree flowers, raw | 1991-2024 | none |
| Coconuts, in shell | 1991-2024 | none |
| Tea leaves | 1991-2009, 2016-2024 | 2010-2015 missing |
| Natural rubber in primary forms | 1993-2009, 2017-2024 | 2010-2016 missing |

## The one thing that will bite you

FAOSTAT's real bulk CSV encodes `Area Code (M49)` with a **leading
apostrophe** (`'144`, not `144`) — its standard Excel-safe text-forcing
convention for numeric-looking codes. `to_fact_trade()` strips it
(`str.lstrip("'")`) before `pd.to_numeric()`; without that, the Sri Lanka
filter silently matches zero rows even with the correct file in place. See
`ceynex-core/docs/HANDOFF_faostat_pinksheet_ingestion_2026-09-24.md` for how
this was found (ran the connector against the real downloaded file).

## To ingest

Drop the downloaded CSV (or the full normalized file — the connector filters
internally) into `data/raw/faostat/<YYYY-MM-DD>/`, then
`python -m ceynex.data.pipeline --sources faostat`.
