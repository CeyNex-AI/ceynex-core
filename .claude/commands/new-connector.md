---
description: Scaffold a DataSourceConnector subclass, its fixture-backed test, and its manifest writer
argument-hint: <SOURCE_ID> (e.g. FAOSTAT, JAAF, CBSL)
---

Scaffold a new data source connector for `$1`.

Read `ceynex/contracts/protocols.py` first — `DataSourceConnector` is a frozen
contract and the new class subclasses it without modifying it.

Create:

1. `ceynex/data/connectors/<source_lower>.py` — a `<Source>Connector(DataSourceConnector)` with:
   - `source_id = "$1"` and `refresh_mode` set to `"scheduled"` for programmatic/bulk
     sources or `"event-driven"` for sources that publish on their own cadence (SRS 3.1.7)
   - `fetch() -> pd.DataFrame` that **caches the raw response to
     `data/raw/<source_lower>/<YYYY-MM-DD>/`** before parsing. These get re-run
     dozens of times; never re-hit the network for data already on disk
   - `manifest() -> SourceManifest` with row count, period range, and frequency
   - `to_fact_trade(raw)` mapping the source's own columns onto the `fact_trade`
     schema in `ceynex/data/schema.sql`, using `ceynex.data.crosswalk` for country
     codes rather than hand-rolled mapping
   - `tenacity` retry with exponential backoff on transport errors
   - a module docstring naming the SRS section it implements

2. `tests/data/test_<source_lower>_connector.py` — asserts against a committed
   ~20-row fixture at `tests/data/fixtures/<source_lower>_sample.<ext>`. **No
   network access in tests.** Cover at minimum:
   - the fixture parses to the expected row count and dtypes
   - `to_fact_trade` produces every non-nullable `fact_trade` column
   - idempotency: the same input twice yields the same `source_hash` values

3. A `data/raw/<source_lower>/PROFILE.md` stub with headings for row count, date
   range, frequency, units, column names, null rate, three sample rows, and
   "the one thing that will bite you".

Then run `make lint && pytest tests/data/test_<source_lower>_connector.py`.
