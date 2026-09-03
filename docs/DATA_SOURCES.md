# Where CeyNex's data comes from, and what we do to it

Written for someone who has not read the code. If you only need one sentence:
**we download Sri Lanka's export figures from the United Nations, clean up the
country and product codes, store them in PostgreSQL, and copy the useful parts
into a Neo4j graph so the agents can ask questions about them.**

Related documents: [DEFERRED.md](DEFERRED.md) for what we chose not to build,
[ARCHITECTURE_DELTA.md](ARCHITECTURE_DELTA.md) for where we departed from the
design and why.

---

## 1. The short version

```
UN Comtrade (the internet)
        │  download
        ▼
data/raw/comtrade/<date>/*.json        ← saved exactly as received, never edited
        │  translate columns, fix codes, drop bad rows
        ▼
PostgreSQL  fact_trade                 ← the single source of truth
        │  copy the parts a graph is good at
        ▼
Neo4j  (Country)-[EXPORTS_TO]-(Commodity)
        │
        ▼
the agents answer questions from here
```

Three commands do the whole thing:

```bash
make db-init    # create the tables (safe to run twice)
make ingest     # download from the UN and fill fact_trade
make kg-load    # build the graph from what fact_trade now holds
```

---

## 2. Where the data actually comes from

### UN Comtrade — the only external source M2 downloads

The United Nations Comtrade database is the official record of what every
country exports to every other country. Governments report to it; it is the
same data the World Bank and WTO build on.

**You do not need an account.** Comtrade offers a free "preview" endpoint that
needs no registration and returns the same fields as the paid one, just capped
at 500 rows per request. Our requests are far smaller than that, so for this
project the two are equivalent. If someone does register and sets
`COMTRADE_API_KEY` in `.env`, the connector automatically switches to the
subscription endpoint. Nothing else changes.

| | Address | Limit |
|---|---|---|
| Free, no key | `comtradeapi.un.org/public/v1/preview/C/A/HS` | 500 rows per request |
| With a key | `comtradeapi.un.org/data/v1/get/C/A/HS` | 500 requests/day |

**What we ask it for.** Sri Lanka's exports (Sri Lanka is country number **144**
in the UN's numbering), one year at a time, for the last ten years, for five
product codes:

| Code | Product | Sector |
|---|---|---|
| 0902 | Tea | agriculture |
| 0906 | Cinnamon | agriculture |
| 4001 | Natural rubber | agriculture |
| 61 | Knitted apparel | apparel |
| 62 | Woven apparel | apparel |

Those codes are **HS codes** — an international product numbering system where
the first two digits are the broad chapter (61 = knitted clothing) and more
digits mean more specific (0902 = tea specifically, inside chapter 09 "coffee,
tea and spices").

A fresh local ingest gives **4,625 rows covering 2015–2024**. The deployed
backend was re-ingested after that and covers **2015–2025** — verified live
2026-09-03, where every answer's latest year is 2025 — so it holds more rows than
this. The window is what to trust; the count depends on when the host last ran
`make ingest`:

```sql
SELECT count(*), min(period_start), max(period_start) FROM fact_trade;
```

Coconut (HS 0801 and 1513) joined the pull on 2026-08-26 and is loaded on the
deployed backend.

### Other sources belong to teammates

The design names six data sources. M2 built one of them. The rest:

| Source | Who | Status |
|---|---|---|
| UN Comtrade | M2 | **built** |
| FAOSTAT, Central Bank of Sri Lanka | M1 | theirs |
| JAAF, EDB | M3 | theirs |
| WITS (tariff rates) | M2 | **cut** — see [DEFERRED.md](DEFERRED.md) |

Everyone writes through the same `UnifiedDatasetWriter`, so their data lands in
the same table as ours and adds to it rather than overwriting it.

### Tables we maintain by hand

Not everything can be downloaded. Six small CSV files under
`ceynex/data/reference/` are typed out and committed, because they change once a
decade and an API call for them would be a needless thing to break:

| File | Rows | What it holds |
|---|---:|---|
| `countries.csv` | 255 | Every country's UN number, three-letter code, and name |
| `hs_codes.csv` | 49 | Product codes and which sector they belong to |
| `partner_aggregates.csv` | 20 | Codes that are *groups* of countries, not countries |
| `trade_agreement_coverage.csv` | 13 | Which products each deal covers |
| `partner_aliases.csv` | 8 | Comtrade codes that don't match the standard (see §5) |
| `trade_agreements.csv` | 6 | GSP+ and the trade deals Sri Lanka is in |

> **The two trade-agreement files are marked `unverified`.** Nobody has yet
> checked them against the official EU regulation. Under our team rule, no
> number derived from them goes into a report until a human has. The graph
> stores that status on each node, so an answer using them can say so.

---

## 3. What we do to the data, step by step

### Step 1 — Download, and keep the original

Every response is saved to `data/raw/comtrade/<date>/` **before anything reads
it**. Raw files are never edited.

This matters more than it looks. If a number in a final answer looks wrong, you
can go back to exactly what the UN sent that day and see whether the problem is
theirs or ours. Without it, "the data changed" and "our code changed" are
indistinguishable. It also means re-running the pipeline doesn't re-download —
useful, because we ran it dozens of times.

### Step 2 — Rename the columns

Comtrade's field names are its own. We map them onto our table:

| Comtrade calls it | We call it |
|---|---|
| `partnerCode` | `partner_m49` → `partner_iso3` |
| `cmdCode` | `hs_code` |
| `primaryValue` | `export_value_usd` |
| `netWgt` | `export_volume` (kg) |
| `refYear` | `period_start` / `period_end` |

### Step 3 — Fix the codes

Three fixes, each for a real problem (details in §5):

1. **Put back missing leading zeros.** Comtrade sends HS codes as numbers, so
   tea's `0902` arrives as `902`. We restore the zero.
2. **Remove group rows.** Comtrade includes rows for "World" and "European
   Union" *alongside* the individual countries. Adding them up would count the
   same exports twice.
3. **Translate odd country codes.** Comtrade uses `842` for the USA where the
   international standard says `840`. We map them.

### Step 4 — Check before writing anything

The writer validates the whole batch first — required columns present, values
are numbers, dates make sense. If anything fails, **nothing** is written. A
half-loaded table is worse than an empty one, because it looks fine.

### Step 5 — Write to PostgreSQL, safely repeatable

Each row gets a `source_hash` — a fingerprint of its identity and its values.
On write we use "insert, or update if this row already exists":

- Row is new → inserted.
- Row exists and the numbers are identical → **left alone**.
- Row exists and the numbers changed → updated (the UN does revise figures).

So **running `make ingest` twice gives the same row count, not double.** You can
re-run it safely any time.

Five tables:

| Table | What's in it |
|---|---|
| `fact_trade` | The actual trade records — one row per product/partner/year |
| `dim_country` | Country reference |
| `dim_hs` | Product code reference |
| `dq_flag` | Suspected data problems, **flagged not deleted** |
| `ingest_run` | A log of every run: when, how many rows, success or failure |

`dq_flag` is worth understanding. When two sources disagree about the same
number, we **record the disagreement and keep both**. Deleting the one you trust
less is a decision that hides itself; a flag is a decision someone can review.

### Step 6 — Also save as Parquet

The same data is written to `data/parquet/`, organised by sector/product/year.
Parquet is a file format that pandas and other tools read quickly without a
database running. Postgres is the source of truth; Parquet is for convenience.

### Step 7 — Build the knowledge graph

`make kg-load` does three things:

1. Applies the graph's rules (each country appears once, each product once).
2. Creates the trade-agreement nodes (GSP+ and the rest).
3. Turns each `fact_trade` row into a relationship:
   `(Tea)-[EXPORTS_TO {year, value}]->(Germany)`.

**Why have both a database and a graph?** They answer different shapes of
question. Postgres is good at "add up tea exports for 2024". The graph is good
at "which products are covered by an agreement that also covers the market
growing fastest" — questions that hop between things. Every write uses `MERGE`,
Neo4j's "create only if missing", so M1's and M3's loaders add to the graph
rather than colliding with ours.

---

## 4. Running it yourself

```bash
make install                        # dependencies
cp .env.example .env                # database settings; API keys can stay blank
make up                             # start PostgreSQL and Neo4j locally
make db-init                        # create the tables
make ingest                         # download and load (a few minutes)
make kg-load                        # build the graph
```

Useful variations:

```bash
python -m ceynex.data.pipeline --sources comtrade --years 2020 2024
python -m ceynex.data.pipeline --offline    # use only what's already downloaded
python -m ceynex.data.pipeline --verify     # print row counts per source
```

### Checking it worked

```sql
SELECT source_id, count(*) FROM fact_trade GROUP BY 1;
```

Expect roughly 4,625 rows from `UN_COMTRADE` on a fresh local ingest, and more on
a host re-ingested since 2026-08-28. Then a number you can check against the real
world:

```sql
SELECT sum(export_value_usd) FROM fact_trade
WHERE item = 'tea' AND extract(year FROM period_start) = 2023;
```

This returns about **USD 1.27 billion**. Sri Lanka's published tea export
earnings for 2023 were about **USD 1.3 billion**. Close enough to believe, and
different enough to remember these are Comtrade's mirror figures rather than the
Sri Lanka Tea Board's own.

> **Team rule: every number gets checked against the real world once before it
> goes in a document.** Not because the code is untrustworthy, but because a
> plausible wrong number is the hardest kind of error to notice later.

---

## 5. Problems in the data you should know about

### 2018 is missing, and it is not our fault

**Sri Lanka reported nothing to Comtrade for 2018.** Every product returns zero
rows for that year, on both endpoints. It is a gap in the source.

This matters more than a missing year usually would. A growth rate measured from
2018 has no starting point, and a forecast trained across it has a hole in its
history. The pipeline prints a warning naming the year rather than quietly
filling it in — an interpolated 2018 would look exactly like real data.

### Country codes that aren't what you'd expect

This one silently destroyed USD 7.23 billion of exports before it was caught.

Comtrade does **not** always use the international standard country numbers. For
countries reported together with their overseas territories, it uses its own:

| Comtrade | Standard | Country | Value that vanished |
|---|---|---|---|
| 842 | 840 | **United States** | USD 6.66 bn |
| 251 | 250 | France | USD 0.34 bn |
| 699 | 356 | India | |
| 757 | 756 | Switzerland | |
| 579 | 578 | Norway | |

Looked up against a standard country table, these match nothing, so they were
dropped as "unknown partner" — with a log line and no error. The USA is Sri
Lanka's **largest apparel market**, so every market-share figure was wrong while
looking entirely reasonable.

Fixed by `partner_aliases.csv`, and there is a test that fails if anyone removes
it.

### Group rows that double-count

The opposite trap, in the same field. Comtrade includes:

- `0` = **World** (the total of everything)
- `97` = **European Union** (reported *as well as* Germany, France, Italy…)

Include either while also counting the individual countries and your totals are
inflated. Removed by `drop_aggregate_partners()`.

**The two traps pull in opposite directions**, which is what makes this field
dangerous: the instinct that protects you from one exposes you to the other.

### Leading zeros

HS codes are text that looks like numbers. Tea is `0902`, and JSON turns that
into `902`. Left alone, it stops matching the product tables. Restored by
`normalize_hs()`.

---

## 6. What we chose not to do

- **No tariff rates.** The WITS connector was cut. Preference-loss simulations
  use a documented constant from `config/elasticities.yaml` and say so in their
  assumptions. See [DEFERRED.md](DEFERRED.md).
- **No made-up data anywhere.** Every figure the system reports traces to a
  Comtrade row or a committed reference table. Where data is missing, the answer
  says it is missing.
- **No deleting suspicious rows.** They get a `dq_flag` and stay.
- **No filling gaps by interpolation.** 2018 stays empty.

---

## 7. File map

| Path | What it does |
|---|---|
| `ceynex/data/connectors/comtrade.py` | Downloads from the UN |
| `ceynex/data/crosswalk.py` | Country and product code translation |
| `ceynex/data/align.py` | Puts monthly/quarterly/annual data on one time base |
| `ceynex/data/writer.py` | Validates and writes to Postgres + Parquet |
| `ceynex/data/reader.py` | Reads series back out |
| `ceynex/data/pipeline.py` | The `make ingest` entry point |
| `ceynex/data/bootstrap.py` | The `make db-init` entry point |
| `ceynex/kg/load.py` | The `make kg-load` entry point |
| `ceynex/data/reference/` | The hand-maintained CSVs |
| `data/raw/comtrade/PROFILE.md` | Technical detail on the Comtrade source |
