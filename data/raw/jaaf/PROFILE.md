# JAAF — Annual Exports / Market Wise Exports

Verified against real saved pages (`Annual Exports – Sri Lanka Apparel.html`,
`Market Wise Exports – Sri Lanka Apparel.html`, saved 2026-08-07).

- **Row count**: 1,196 raw rows (5 tables x 23 years x up to 12 months, 2004-2026,
  minus genuinely missing cells); 658 `fact_trade` rows after keeping only
  Total/US/UK and dropping EU-bloc/Other
- **Date range**: 2004-2026 (2026 partial year, in progress as of the saved page)
- **Frequency**: Monthly (M)
- **Units**: USD Millions — confirmed via the page's own heading text,
  `"TEXTILE AND APPAREL EXPORTS IN US$ Mn"`
- **Column names**: real header is `["", "Jan", ..., "Dec", "Total"]`, one row
  per year (most recent first) — **not** month-rows/year-columns as originally
  assumed; `_annual_tables_to_records` melts this into `market`, `month`,
  `year`, `value_usd_mn` and skips the trailing "Total" column (a derived sum)
- **Null rate**: near-zero once the literal-"NaN"-text quirk below is handled;
  a handful of genuinely missing cells (blank, not "NaN") for partial years
- **Three sample rows** (2025, Total table): `Jan=425.44` (2026, partial-year,
  Jun-Dec are `0.00` placeholders not blanks), `Jan=437.07`/`Total=5019.20` (2025,
  matches the market-wise pie chart's grand total), `Jan=357.73` (2024)
- **The one thing that will bite you**: two things, both confirmed on the real
  page:
  1. `srilankaapparel.com` disallows automated fetching (robots.txt).
     `fetch()` never makes an HTTP request — it reads a page you saved
     manually (Ctrl+S). Re-running this connector on stale saved HTML
     silently returns stale data; there is no way for the connector itself to
     detect staleness.
  2. **The page itself renders some missing cells as the literal text
     `"NaN"`**, not blank (e.g. UK's Feb/Mar in several years 2004-2016).
     `float("nan")` doesn't raise, so a naive parser would silently write a
     `NaN` `export_value_usd` instead of dropping the row — `_num` checks for
     this explicitly. Also confirmed by real data: only 3 of the 5 tables
     (Total, US, UK) are confidently labeled by cross-matching against the
     market-wise pie chart (`US`: 1947.37 ≈ pie's 1947.38; `UK`: 679.66 exact
     match). EU-bloc and Other are excluded from `to_fact_trade` entirely, not
     just flagged, and brute-force testing which named markets they'd sum to
     found no coherent geographic grouping — treat their composition as
     genuinely unknown, not "probably EU vs. the rest".
