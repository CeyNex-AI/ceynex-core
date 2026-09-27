"""Implements SRS 3.1.7 — the JAAF Annual/Market-Wise Exports connector.

srilankaapparel.com/data-center/ disallows automated fetching (robots.txt), so
`fetch()` never makes an HTTP request: it reads HTML pages the caller has
already saved locally (Ctrl+S / "Save As" in a browser). `refresh_mode` is
"event-driven" (SRS 3.1.7), not "scheduled" — there is nothing to poll.

Two page types:
  - Annual Exports: standard `<table>` elements -> BeautifulSoup.
  - Market Wise Exports: NOT a `<table>`; the pie-chart values live inside an
    inline `<script>` block as a Chart.js `data` array with a matching
    `labels` array -> parsed by regex.

The Annual Exports page's 5 tables are, in fixed order, Total / US /
EU-bloc(approx) / UK / Other(approx) — confirmed by cross-matching each
table's latest-year total against the Market Wise pie-chart's named entries.
Only Total, US and UK match a pie-chart label exactly; EU-bloc and Other are
unofficial groupings JAAF does not itself label.

`to_fact_trade` writes only Total (as the `partner_iso3 IS NULL` "World" row),
US and UK from the Annual Exports tables. EU-bloc(approx) and Other(approx) are
excluded, not just flagged: written alongside US/UK/Total they would
double-count in exactly the way `docs/ARCHITECTURE_DELTA.md` D4 excludes
Comtrade's World and EU-aggregate partner rows for the same reason.

**Market Wise's per-country breakdown, wired in 2026-09-27.** The pie chart
itself names real individual countries the Annual Exports tables never do
(confirmed on a real saved page: Italy, Germany, Netherlands, Canada, France,
Belgium, Australia, India, China, Sweden, UAE, Hong Kong, Mexico, Ireland,
alongside US/UK/an unlabeled "Other Markets" residual) — until now, only used
to cross-check the Annual Exports table order, then discarded. It is one
annual snapshot, not a monthly time series like the Annual Exports tables
(`PROFILE.md`: the pie chart's US figure matches the Annual Exports "us"
table's most recent *complete* year, summed across its 12 months) — `fetch`
works out which year that is by finding the annual "us" total the pie chart's
own US value matches most closely, the same cross-check the module already
relies on to confirm table order, rather than assuming "latest" is always
correct on a partial in-progress year.

US, UK and the "Other Markets" residual are excluded from this half too:
US/UK would double-count the Annual Exports tables' own monthly data for the
same countries (this connector writes both at different `frequency`s, and
nothing downstream is guaranteed to filter by frequency before summing), and
"Other Markets" cannot be geocoded to a real partner. Only the 14 markets with
no existing coverage are new rows.
"""

from __future__ import annotations

import calendar
import hashlib
import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from ceynex.contracts.protocols import DataSourceConnector, SourceManifest
from ceynex.data.crosswalk import market_to_iso3

log = logging.getLogger(__name__)

TABLE_LABELS = ["total", "us", "eu_bloc_approx", "uk", "other_approx"]

# label -> (iso3, m49); labels absent here are excluded from to_fact_trade.
# "total" maps to (None, None): partner_iso3 = NULL is this schema's "World".
_FACT_TRADE_PARTNERS: dict[str, tuple[str | None, int | None]] = {
    "total": (None, None),
    "us": ("USA", 842),
    "uk": ("GBR", 826),
}


def _num(s: str) -> float | None:
    # The live page itself renders some missing cells as the literal text
    # "NaN" (confirmed in a saved page — e.g. UK Feb/Mar in several older
    # years), which must be checked explicitly: float("nan") doesn't raise.
    s = s.replace(",", "").strip()
    if s == "" or s.lower() in ("-", "...", "..", "n/a", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_annual_exports_html(html: str) -> list[dict]:
    """Extract the 5 monthly-by-year export tables from a saved Annual Exports page."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    results = []
    for i, t in enumerate(tables):
        rows = t.find_all("tr")
        if not rows:
            continue
        header = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])]
        data_rows = []
        for r in rows[1:]:
            cells = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
            if cells and cells[0].strip().isdigit():
                data_rows.append(cells)
        label = TABLE_LABELS[i] if i < len(TABLE_LABELS) else f"table_{i}"
        results.append({"market": label, "header": header, "rows": data_rows})
    return results


def extract_market_wise_html(html: str) -> list[tuple[str, float]] | None:
    """Extract the market-wise pie chart data (labels + values) via regex.

    Doubles as the offline cross-check that the Annual Exports table
    order/labels still hold (see module docstring), and as the source of the
    per-country rows `to_fact_trade` writes for markets the Annual Exports
    tables never break out on their own.
    """
    data_match = re.search(r"data:\s*\[\s*([\d\.\,\s]+)\]", html)
    labels_match = re.search(r'labels:\s*\[\s*((?:"[^"]*",?\s*)+)\]', html)
    if not (data_match and labels_match):
        return None
    values = [float(v) for v in data_match.group(1).split(",") if v.strip()]
    labels = re.findall(r'"([^"]*)"', labels_match.group(1))
    return list(zip(labels, values, strict=False))


# Excluded from the market-wise -> fact_trade rows: US/UK would double-count
# the Annual Exports tables' own monthly series for the same two countries
# (written at a different frequency, which nothing downstream is guaranteed
# to filter by before summing), and "Other Markets" is an unlabeled residual
# with no real geography to write it under.
_MARKET_WISE_EXCLUDED = {"us", "usa", "uk", "other markets"}

# How close the pie chart's own US/USA figure must land to an annual "us"
# total before that year is trusted as the snapshot's period. JAAF's numbers
# are given to 2 decimal places in USD millions; a few cents of rounding
# drift is expected, a whole different year's total is not.
_YEAR_MATCH_TOLERANCE_USD_MN = 1.0


def _market_wise_year(annual_records: list[dict], market_wise: list[tuple[str, float]]) -> int | None:
    """Which year the market-wise snapshot represents.

    Not "whatever year is latest": a partial, still-in-progress year's total
    would never match the pie chart's own (necessarily complete) figure, and
    a hardcoded "always the year before latest" would break the day the
    connector is next run against a saved page from a full year later.
    Matches on the US total specifically because it is the one already
    trusted to confirm table identity (module docstring) -- if this fails to
    resolve, the whole page pairing is suspect, not just the year.
    """
    us_pie_value = next(
        (value for label, value in market_wise if label.strip().lower() in ("us", "usa")), None
    )
    if us_pie_value is None:
        return None
    annual_us_totals: dict[int, float] = {}
    for record in annual_records:
        if record["market"] == "us":
            annual_us_totals[record["year"]] = annual_us_totals.get(record["year"], 0.0) + record["value_usd_mn"]
    best_year, best_diff = None, None
    for year, total in annual_us_totals.items():
        diff = abs(total - us_pie_value)
        if best_diff is None or diff < best_diff:
            best_year, best_diff = year, diff
    if best_year is None or best_diff > _YEAR_MATCH_TOLERANCE_USD_MN:
        return None
    return best_year


def _market_wise_to_records(market_wise: list[tuple[str, float]], year: int) -> list[dict]:
    """The market-wise pie chart's new-market entries, as one annual record
    each -- same record shape `_annual_tables_to_records` produces, with
    `month=None` marking it annual rather than monthly."""
    records = []
    for label, value in market_wise:
        if label.strip().lower() in _MARKET_WISE_EXCLUDED:
            continue
        records.append(
            {"market": f"market_wise:{label}", "month": None, "year": year, "value_usd_mn": value}
        )
    return records


_MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip


def _annual_tables_to_records(tables: list[dict]) -> list[dict]:
    """Melt each table's (year-row x month-column) grid into long records.

    Real header shape (confirmed against a saved page): `["", "Jan", ...,
    "Dec", "Total"]`, one row per year, most recent year first. Month columns
    are detected from the header rather than assumed to be a fixed 12, and the
    trailing "Total" column is a derived sum, not a 13th month — it's excluded
    by construction since "tot" isn't a month key. Rows grow annually (a new
    year row, not a new column), which this handles for free by not assuming
    a row count.
    """
    records = []
    for table in tables:
        header = table["header"]
        month_cols = [
            (idx, _MONTH_NUMBERS[h.strip().lower()[:3]])
            for idx, h in enumerate(header)
            if h.strip().lower()[:3] in _MONTH_NUMBERS
        ]
        for row in table["rows"]:
            year_cell = row[0].strip()
            if not (year_cell.isdigit() and len(year_cell) == 4):
                continue
            year = int(year_cell)
            for idx, month in month_cols:
                if idx >= len(row):
                    continue
                value = _num(row[idx])
                if value is None:
                    continue
                if value == 0:
                    # The live page pre-renders the current year's remaining
                    # months as a literal "0" placeholder before they're
                    # reported, not a real observation -- confirmed live
                    # 2026-08-26: a saved page's June-December cells for the
                    # in-progress year were all exactly 0, while every real
                    # month (this connector's monthly totals run in the
                    # hundreds of millions of USD) is never actually zero.
                    # Left unfiltered, these got ingested as real rows and
                    # pushed this item's latest observed year into the
                    # future, which corrupted every cross-item "latest year"
                    # query (export_analytics, trade_economics) that assumed
                    # the graph's global max year was real data.
                    continue
                records.append(
                    {
                        "market": table["market"],
                        "month": month,
                        "year": year,
                        "value_usd_mn": value,
                    }
                )
    return records


def _month_end(period_start: date) -> date:
    return period_start.replace(day=calendar.monthrange(period_start.year, period_start.month)[1])


def _source_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


class JAAFConnector(DataSourceConnector):
    """Monthly apparel export values by market from JAAF's data center (SRS 3.1.7)."""

    source_id = "JAAF"
    refresh_mode = "event-driven"

    def __init__(
        self,
        annual_exports_path: str,
        market_wise_path: str | None = None,
        data_dir: str = "data/raw/jaaf",
    ):
        self.annual_exports_path = Path(annual_exports_path)
        self.market_wise_path = Path(market_wise_path) if market_wise_path else None
        self.data_dir = Path(data_dir)
        self._last_manifest: SourceManifest | None = None

    def fetch(self) -> pd.DataFrame:
        if not self.annual_exports_path.exists():
            raise FileNotFoundError(
                f"{self.annual_exports_path} not found — srilankaapparel.com blocks "
                "automated fetching, so save the Annual Exports page manually "
                "(Ctrl+S) before running this connector."
            )
        html = self.annual_exports_path.read_text(encoding="utf-8", errors="ignore")
        cache_dir = self.data_dir / datetime.now(UTC).strftime("%Y-%m-%d")
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "annual_exports.html").write_text(html, encoding="utf-8")

        tables = extract_annual_exports_html(html)
        raw = pd.DataFrame.from_records(_annual_tables_to_records(tables))

        notes: dict = {}
        if self.market_wise_path and self.market_wise_path.exists():
            mw_html = self.market_wise_path.read_text(encoding="utf-8", errors="ignore")
            (cache_dir / "market_wise.html").write_text(mw_html, encoding="utf-8")
            market_wise = extract_market_wise_html(mw_html)
            notes["market_wise_labels"] = [label for label, _ in (market_wise or [])]
            if market_wise:
                annual_records = _annual_tables_to_records(tables)
                year = _market_wise_year(annual_records, market_wise)
                if year is None:
                    # Table identity itself is now suspect (module docstring's
                    # cross-check) -- said in the manifest rather than
                    # guessed at, so a stale or mismatched page pairing is
                    # visible in ingest_run rather than silently mis-dated.
                    notes["market_wise_year_unresolved"] = True
                else:
                    notes["market_wise_year"] = year
                    mw_rows = _market_wise_to_records(market_wise, year)
                    if mw_rows:
                        raw = pd.concat(
                            [raw, pd.DataFrame.from_records(mw_rows)], ignore_index=True
                        )

        self._last_manifest = SourceManifest(
            source_id=self.source_id,
            fetched_at=datetime.now(UTC).isoformat(),
            row_count=len(raw),
            period_start=str(int(raw["year"].min())) if not raw.empty else None,
            period_end=str(int(raw["year"].max())) if not raw.empty else None,
            frequency="M",
            notes=notes,
        )
        return raw

    def manifest(self) -> SourceManifest:
        if self._last_manifest is None:
            raise RuntimeError("fetch() must run before manifest()")
        return self._last_manifest

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """`fact_trade` rows: Total/US/UK monthly from the Annual Exports
        tables, plus one annual row per market-wise-only country — see the
        module docstring for why each half excludes what it does."""
        if raw.empty:
            return pd.DataFrame()
        out_rows = []
        for _, r in raw.iterrows():
            market = str(r["market"])
            if market.startswith("market_wise:"):
                label = market.removeprefix("market_wise:")
                iso3, m49 = market_to_iso3(label)
                if iso3 is None:
                    # A market-wise label this project's crosswalk does not
                    # yet recognize -- dropped, not guessed at, same
                    # discipline as an item mismatch anywhere else in this
                    # codebase. `ingest_run`'s row count will simply be one
                    # short of `len(market_wise)` for the year it happens.
                    log.warning("jaaf: market-wise label %r not in the country crosswalk; dropped", label)
                    continue
                year = int(r["year"])
                out_rows.append(
                    {
                        "source_id": self.source_id,
                        "sector": "apparel",
                        "item": "apparel_textiles",
                        "hs_code": None,
                        "reporter_iso3": "LKA",
                        "reporter_m49": 144,
                        "partner_iso3": iso3,
                        "partner_m49": m49,
                        "period_start": date(year, 1, 1).isoformat(),
                        "period_end": date(year, 12, 31).isoformat(),
                        "frequency": "A",
                        "export_volume": None,
                        "volume_unit": None,
                        "export_value_usd": float(r["value_usd_mn"]) * 1_000_000.0,
                        "price": None,
                        "price_unit": None,
                        "fx_usd_lkr": None,
                        "source_hash": _source_hash(self.source_id, market, year),
                    }
                )
                continue
            if market not in _FACT_TRADE_PARTNERS:
                continue
            iso3, m49 = _FACT_TRADE_PARTNERS[market]
            period_start = date(int(r["year"]), int(r["month"]), 1)
            out_rows.append(
                {
                    "source_id": self.source_id,
                    "sector": "apparel",
                    "item": "apparel_textiles",
                    "hs_code": None,
                    "reporter_iso3": "LKA",
                    "reporter_m49": 144,
                    "partner_iso3": iso3,
                    "partner_m49": m49,
                    "period_start": period_start.isoformat(),
                    "period_end": _month_end(period_start).isoformat(),
                    "frequency": "M",
                    "export_volume": None,
                    "volume_unit": None,
                    "export_value_usd": float(r["value_usd_mn"]) * 1_000_000.0,
                    "price": None,
                    "price_unit": None,
                    "fx_usd_lkr": None,
                    "source_hash": _source_hash(
                        self.source_id, market, r["year"], r["month"]
                    ),
                }
            )
        return pd.DataFrame.from_records(out_rows)
