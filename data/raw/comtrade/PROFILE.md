# UN Comtrade — source profile

**Feeds:** Export Analytics, Trade Economics, Forecast · **Refresh:** scheduled
**Access:** REST. Subscription key optional — see below.
**Connector:** `ceynex/data/connectors/comtrade.py`

## What we pull

Sri Lanka (reporter M49 **144**) exports, annual, HS `0902` tea · `0906`
cinnamon · `4001` rubber · `61` knit apparel · `62` woven apparel, last ten
complete years. One call per (HS code, year).

## Endpoints

| | URL | Cap |
|---|---|---|
| With `COMTRADE_API_KEY` | `/data/v1/get/C/A/HS` | 500 calls/day, 100k records/call |
| Without | `/public/v1/preview/C/A/HS` | 500 records/call |

The preview endpoint returns the same fields and requires no registration, which
is why this connector works with an empty key and why the demo runs on real UN
figures rather than invented ones. Our per-call result sets are well under 500
rows, so for this scope the two endpoints are equivalent.

## Field mapping

| Comtrade | `fact_trade` | Note |
|---|---|---|
| `partnerCode` | `partner_m49` → `partner_iso3` | via `crosswalk.to_iso3` |
| `cmdCode` | `hs_code` | **integer in the response** — leading zero restored |
| `primaryValue` (fallback `fobvalue`) | `export_value_usd` | |
| `netWgt` (fallback `qty`) | `export_volume` | kg |
| `refYear` | `period_start` / `period_end` | 1 Jan – 31 Dec |

## Known gaps

**Sri Lanka reported nothing for 2018.** Every HS code returns `count: 0` for
that year, on both endpoints — a genuine gap in the source, not a connector bug.
The pipeline prints a warning naming the year. Any CAGR whose endpoint lands on
2018, and any forecast window spanning it, has a hole in it and must say so.

## The one thing that will bite you

**Partner codes are not all ISO 3166-1 numeric, and the wrong ones fail in
opposite directions.**

- `0` is **World** and `97` is the **EU reported alongside its own member
  states**. Include either and every total double-counts.
- `842` is the **USA**, `251` **France**, `699` **India**, `757` Switzerland,
  `579` Norway — Comtrade's own codes for territories reported with their
  dependencies. Drop these as "unknown" and you lose your largest market. On a
  2021–2023 pull that was **USD 7.23bn**, USD 6.66bn of it the USA alone.

Both cases are handled in `ceynex/data/crosswalk.py` — `drop_aggregate_partners()`
for the first, `partner_aliases.csv` for the second — and both are asserted in
`tests/data/test_comtrade_connector.py` against a fixture that deliberately
contains all of them.
