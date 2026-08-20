# EDB — Export Performance Indicators (Apparel & Textiles)

Two distinct PDF layouts exist and are both supported, selected via
`EDBReportSource.layout`. Verified against three real files:
`export-performance-indicators-of-sri-lanka-2023.pdf` and `...-2024.pdf`
(`layout="annual"`), and `export-performance-indicators-2009-2018.pdf`
(`layout="archive"`).

## `"annual"` layout (2023/2024 editions)

- **Row count**: 1,359 raw rows (17 tables x ~40-80 markets, 2 editions combined); 6,183 `fact_trade` rows after the 5-year melt and crosswalk resolution (98/99 real market names in these two editions resolve via `ceynex.data.crosswalk`; only `"Not Specified"` is correctly left unmapped)
- **Date range**: 2023 edition covers 2019-2023; 2024 edition covers 2020-2024. **`latest_year == edition_year`** for both — do not assume a publication lag; pass it explicitly per `EDBReportSource`
- **Frequency**: Annual (A)
- **Section**: 25.78-25.94 (17 apparel/textile sub-categories)
- **Units**: USD Millions (`"Value in US$ Millions"` header, confirmed on the real PDFs). `to_fact_trade` multiplies by 1,000,000 to get `export_value_usd` — an earlier version of this connector assumed USD '000s and multiplied by 1,000 instead; caught and fixed once the real PDF text was checked.
- **Column names**: `rank`, `market`, `year_minus4..year_latest`, `share_pct`, `avg_growth_pct`, plus `table_id`/`product`/`edition_year`/`latest_year` added during parsing
- **Null rate**: 0% for parsed numeric fields on real data (no `-`/blank cells observed in the tables checked)
- **Three sample rows** (2024 edition, table 25.79 "APPREL"): `United States: 2019=2338.33 2020=1649.27 2021=2082.60 2022=2300.24 2023=1782.72`; `United Kingdom: 2019=747.25 ... 2023=614.62`; `Italy: 2019=416.20 ... 2023=562.39` (all USD Mn)
- **The one thing that will bite you**: two things, both confirmed on real data:
  1. `pdfplumber`'s default `extract_table()` silently corrupts the 2023 edition (rows collapse into 3 mangled cells) while parsing 2024 cleanly — this connector never calls it, it parses the raw text layer via `parse_table_page` instead.
  2. **`table_id` is not a stable product key across editions.** Table `25.89` is "Made-Up Textile Articles" in the 2023 edition and "Made-Up Clothing Accessories" in the 2024 edition (confirmed); table `25.94` similarly swaps meaning. `to_fact_trade` keys `source_hash` on `product` text, not `table_id`, for exactly this reason — never join across editions on `table_id` alone. Product-name wording also drifts slightly edition to edition for what's otherwise the same category (punctuation only) — not normalized here, flagged as a follow-up (a controlled product vocabulary, not something to invent unilaterally in one connector).

## `"archive"` layout (older multi-year volumes, e.g. the 2009-2018-titled PDF)

- **Row count**: 160 raw rows (4 tables x ~40 markets) for the single file checked; 769 `fact_trade` rows after the melt (67 unique markets, all but 0 resolve after the crosswalk was extended — the 2 gaps found, Honduras and Puerto Rico, were added)
- **Date range**: the file is titled "2009-2018" but its Apparel & Textiles tables only span **one 5-year trailing window, 2014-2018** — the wider title describes the bound volume/series, not this table's actual range. If earlier years (2009-2013) are needed, they were not located in this file; would need a different EDB archive volume.
- **Frequency**: Annual (A)
- **Section**: 17.80-17.83 — only **4** apparel/textile sub-categories (Apparel & Textiles-Total, Woven Fabrics, Apparel, Made-Up Textile Articles), far coarser than the annual layout's 17. No attempt is made here to reconcile this coarser taxonomy against the annual layout's finer one — they'll currently coexist in `fact_trade` as distinct `item` values even where they cover overlapping concepts.
- **Units**: also USD Millions (same header text as the annual layout)
- **Column layout**: rows interleave a %share after every year's value — `<rank> <market> <v1> <s1%> ... <v5> <s5%> <avg growth%>` (11 numbers) — not the annual layout's 5-values-then-one-trailing-%share (7 numbers). Confirmed the wrong pattern silently returns zero rows rather than erroring, so this was easy to miss without checking real data.
- **The one thing that will bite you**: the page-range constant for this layout (`ARCHIVE_APPAREL_TEXTILE_PAGES`, 195-220) is deliberately padded wider than the real table location (206-209 in the file checked) to survive minor drift in a different archive volume — but that padding also swept in unrelated sections (fish, food, tobacco tables were present in the padded range on the real file). `parse_pdf_bytes` filters on `table_id ∈ ARCHIVE_TARGET_TABLE_IDS` as the real safeguard; the page range alone is not sufficient to scope this to Apparel & Textiles.
