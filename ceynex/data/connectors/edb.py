"""Implements SRS 3.1.7 — the EDB Export Performance Indicators connector.

Sri Lanka EDB's "Export Performance Indicators" (EPI) PDF is the national
reference for exporter performance by destination market. Only the Apparel &
Textiles sub-sector tables are in scope for the Apparel & Manufacturing Agent;
every other section is skipped.

`refresh_mode` is "event-driven" (SRS 3.1.7): EDB publishes on its own
schedule, not on a pollable cadence.

pdfplumber's line-based `extract_table()` silently corrupts the 2023 annual
edition (rows collapse into 3 mangled cells) while parsing the 2024 edition
fine. Fixed by parsing the raw text layer with a regex line-matcher instead,
which works identically on both years — tested against both editions.

Two distinct EDB PDF layouts are supported, selected via `EDBReportSource.layout`:

- `"annual"` (default): the 2023/2024-style single-edition report. Apparel &
  Textiles is Section 25.78-25.94 (17 sub-categories), each market row is
  `<rank> <market> <5 year values> <latest-year %share> <avg growth%>`.
- `"archive"`: older multi-year retrospective volumes (confirmed against a
  real 2009-2018-titled archive PDF). Apparel & Textiles is Section
  17.80-17.83 — only 4 sub-categories, coarser than the annual layout's 17 —
  and each row interleaves a %share after every year's value:
  `<rank> <market> <v1> <s1%> <v2> <s2%> <v3> <s3%> <v4> <s4%> <v5> <s5%> <avg growth%>`.
  Confirmed the underlying data table itself only spans one 5-year trailing
  window (2014-2018 in the file checked) despite the file's "2009-2018" name;
  the wider title describes the bound volume, not this table's actual range.
  Both layouts parse down to the same row shape (`year_minus4..year_latest`,
  latest-year `share_pct`, `avg_growth_pct`) so `to_fact_trade` needs no
  layout-specific logic — only the per-year %share values from "archive"
  layout other than the latest year are discarded (unused downstream, same as
  "annual" layout never captures them at all).
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pdfplumber
from tenacity import retry, stop_after_attempt, wait_exponential

from ceynex.contracts.protocols import DataSourceConnector, SourceManifest
from ceynex.data.crosswalk import canonical_item, market_to_iso3

# "annual" layout — matches: <rank> <market> <5 numeric year columns> <% share> <% avg growth>
ROW_PATTERN = re.compile(
    r"^(\d+)\s+([A-Za-z][A-Za-z\.\'\(\), ]*?)\s+"
    r"([\-\d,\.]+)\s+([\-\d,\.]+)\s+([\-\d,\.]+)\s+([\-\d,\.]+)\s+([\-\d,\.]+)\s+"
    r"([\-\d\.]+)\s+([\-\d\.]+)$"
)

# "archive" layout — matches: <rank> <market> then 5x (<value> <% share>) then <% avg growth>
ARCHIVE_ROW_PATTERN = re.compile(
    r"^(\d+)\s+([A-Za-z][A-Za-z\.\'\(\), ]*?)\s+"
    r"([\-\d,\.]+)\s+([\-\d\.]+)\s+"
    r"([\-\d,\.]+)\s+([\-\d\.]+)\s+"
    r"([\-\d,\.]+)\s+([\-\d\.]+)\s+"
    r"([\-\d,\.]+)\s+([\-\d\.]+)\s+"
    r"([\-\d,\.]+)\s+([\-\d\.]+)\s+"
    r"([\-\d\.]+)$"
)

# Section 25.78 (APPAREL & TEXTILES total) through 25.94 (Made-up Textile
# Articles) sit on a fixed, consistent page range in the 2023/2024 "annual"
# editions. Re-verify page offsets before trusting this range for a new edition.
APPAREL_TEXTILE_PAGES = range(244, 261)  # 0-indexed pdfplumber page numbers
ANNUAL_TARGET_TABLE_IDS = {f"25.{n}" for n in range(78, 95)}

# Section 17.80 (APPAREL AND TEXTILES total) through 17.83 (Made-Up Textile
# Articles) in the "archive" layout — confirmed at pdfplumber pages 206-209 in
# the 2009-2018-titled volume checked. Padded to allow for a page or two of
# drift in a different archive volume without falling back to a full scan —
# the padding also sweeps in unrelated sections (fish, food, tobacco tables
# were observed in the padded range on the real file), so `ARCHIVE_TARGET_TABLE_IDS`
# below is what actually keeps this connector Apparel-&-Textiles-only, not the
# page range by itself.
ARCHIVE_APPAREL_TEXTILE_PAGES = range(195, 220)  # 0-indexed pdfplumber page numbers
ARCHIVE_TARGET_TABLE_IDS = {"17.80", "17.81", "17.82", "17.83"}

_YEAR_COLUMNS = ["year_minus4", "year_minus3", "year_minus2", "year_minus1", "year_latest"]
_YEAR_OFFSETS = [4, 3, 2, 1, 0]


@dataclass(frozen=True)
class EDBReportSource:
    """One EPI PDF edition/volume to pull.

    `latest_year` is the calendar year of the rightmost data column in that
    edition's tables. EPI editions report the prior year's full-year data, but
    this is not asserted here — pass the value confirmed against the report's
    own cover page, don't guess it from `edition_year`.

    `layout` selects the row/page format — see the module docstring. Defaults
    to `"annual"` (the 2023/2024-style single-edition report).
    """

    edition_year: int
    latest_year: int
    url: str | None = None
    path: str | None = None
    layout: str = "annual"  # "annual" | "archive"


def _num(s: str) -> float | None:
    s = s.replace(",", "")
    if s in ("-", "", "...", ".."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_table_page(text: str) -> list[dict]:
    """Parse one "annual"-layout EDB table page's text layer into structured rows."""
    rows = []
    for line in text.split("\n"):
        m = ROW_PATTERN.match(line.strip())
        if m:
            rank, market, y1, y2, y3, y4, y5, share, growth = m.groups()
            rows.append(
                {
                    "rank": int(rank),
                    "market": market.strip(),
                    "year_minus4": _num(y1),
                    "year_minus3": _num(y2),
                    "year_minus2": _num(y3),
                    "year_minus1": _num(y4),
                    "year_latest": _num(y5),
                    "share_pct": _num(share),
                    "avg_growth_pct": _num(growth),
                }
            )
    return rows


def parse_archive_table_page(text: str) -> list[dict]:
    """Parse one "archive"-layout EDB table page's text layer into structured rows.

    Per-year %share values other than the latest year are discarded — not
    captured by `to_fact_trade` in the "annual" layout either, so dropping
    them here keeps both layouts producing the same row shape.
    """
    rows = []
    for line in text.split("\n"):
        m = ARCHIVE_ROW_PATTERN.match(line.strip())
        if m:
            rank, market, y1, s1, y2, s2, y3, s3, y4, s4, y5, s5, growth = m.groups()
            rows.append(
                {
                    "rank": int(rank),
                    "market": market.strip(),
                    "year_minus4": _num(y1),
                    "year_minus3": _num(y2),
                    "year_minus2": _num(y3),
                    "year_minus1": _num(y4),
                    "year_latest": _num(y5),
                    "share_pct": _num(s5),
                    "avg_growth_pct": _num(growth),
                }
            )
    return rows


def parse_pdf_bytes(
    data: bytes, edition_year: int, latest_year: int, layout: str = "annual"
) -> pd.DataFrame:
    """Parse the apparel/textile sub-sector tables out of one EPI PDF's bytes.

    Filters on `table_id` membership in the layout's target set, not just the
    page range — the "archive" layout's page range is deliberately padded for
    edition drift, and that padding also sweeps in unrelated sections (fish,
    food, tobacco tables were observed there on the real file checked). The
    table_id filter is what actually keeps this Apparel-&-Textiles-only.
    """
    page_range = APPAREL_TEXTILE_PAGES if layout == "annual" else ARCHIVE_APPAREL_TEXTILE_PAGES
    row_parser = parse_table_page if layout == "annual" else parse_archive_table_page
    target_table_ids = ANNUAL_TARGET_TABLE_IDS if layout == "annual" else ARCHIVE_TARGET_TABLE_IDS
    records = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i in page_range:
            if i >= len(pdf.pages):
                break
            text = pdf.pages[i].extract_text() or ""
            title_match = re.search(r"Table\s*:\s*([\d\.]+)", text)
            product_match = re.search(r"Product\s*:\s*(.+)", text)
            if not (title_match and product_match):
                continue
            table_id = title_match.group(1)
            if table_id not in target_table_ids:
                continue
            product = product_match.group(1).strip()
            for row in row_parser(text):
                row["table_id"] = table_id
                row["product"] = product
                row["edition_year"] = edition_year
                row["latest_year"] = latest_year
                records.append(row)
    return pd.DataFrame.from_records(records)


def _source_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


class EDBConnector(DataSourceConnector):
    """Apparel & Textiles sub-sector tables from EDB's annual EPI report (SRS 3.1.7)."""

    source_id = "EDB"
    refresh_mode = "event-driven"

    def __init__(self, sources: list[EDBReportSource], data_dir: str = "data/raw/edb"):
        self.sources = sources
        self.data_dir = Path(data_dir)
        self._last_manifest: SourceManifest | None = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    def _download(self, url: str) -> bytes:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    def _cache_path(self, source: EDBReportSource) -> Path:
        day_dir = self.data_dir / datetime.now(UTC).strftime("%Y-%m-%d")
        return day_dir / f"epi_{source.edition_year}.pdf"

    def _bytes_for(self, source: EDBReportSource) -> bytes:
        if source.path and Path(source.path).exists():
            return Path(source.path).read_bytes()
        cache_path = self._cache_path(source)
        if cache_path.exists():
            return cache_path.read_bytes()
        if not source.url:
            raise ValueError(
                f"EDB source for edition {source.edition_year} has neither a local "
                "path nor a url to download from"
            )
        data = self._download(source.url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return data

    def fetch(self) -> pd.DataFrame:
        frames = [
            parse_pdf_bytes(
                self._bytes_for(source), source.edition_year, source.latest_year, source.layout
            )
            for source in self.sources
        ]
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        latest_years = [s.latest_year for s in self.sources]
        self._last_manifest = SourceManifest(
            source_id=self.source_id,
            fetched_at=datetime.now(UTC).isoformat(),
            row_count=len(raw),
            period_start=str(min(latest_years) - 4) if latest_years else None,
            period_end=str(max(latest_years)) if latest_years else None,
            frequency="A",
        )
        return raw

    def manifest(self) -> SourceManifest:
        if self._last_manifest is None:
            raise RuntimeError("fetch() must run before manifest()")
        return self._last_manifest

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Long-format `fact_trade` rows: one per (table, market, year).

        Markets `ceynex.data.crosswalk` can't resolve are dropped, not written
        with `partner_iso3 = NULL` — that specific value means "World" in
        `ceynex/data/schema.sql`, and a dropped market must never be confused
        with that aggregate.

        `item` is `product` (the table's own free-text label), not `table_id`
        — confirmed against the real 2023/2024 PDFs that EDB renumbers tables
        between editions (2023's "25.89" is "Made-Up Textile Articles"; 2024's
        "25.89" is "Made-Up Clothing Accessories"), so `table_id` is only a
        page locator for the edition it came from, never a stable product key.
        Wording also drifts between editions for what is otherwise the same
        category — "&" against "AND", hyphen against en-dash, the parenthetical
        gloss present or absent, and outright typos ("APPREL", "SPORTSWERA").
        Since `item` is part of `fact_trade_upsert_key`, each variant is a
        separate identity that inserts instead of updating, silently splitting
        one series in two. Measured in production 2026-08-30: `APPAREL` held
        2014-2018 and `APPREL` 2019-2024 — the same series, and the reason the
        registered apparel model was fitted on 5 rows.

        `canonical_item()` (`data/crosswalk.py`, backed by
        `reference/item_vocabulary.csv`) now resolves the label before it is
        written, the same way `market_to_iso3` resolves the market name one
        line above. It raises on a label it does not know rather than passing
        it through, because passing through is what created the split.

        EDB editions overlap in the years they cover — the 2023 edition's
        tables span 2019-2023, the 2024 edition's span 2020-2024 — so
        `fetch()` concatenating both raw editions means the same (item,
        market, year) triple gets melted twice, once from each edition.
        Confirmed against real 2023+2024 data (`data/raw/edb/manual/`):
        2020-2023 rows were emitted twice, ~37% of all EDB rows, silently
        doubling any downstream `groupby(year).sum()`. Resolved below by
        keeping only the highest-`edition_year` row per (item, market,
        year) — the most recently published figure — rather than letting
        both survive into the output.
        """
        if raw.empty:
            return pd.DataFrame()
        out_rows = []
        for _, r in raw.iterrows():
            iso3, m49 = market_to_iso3(r["market"])
            if iso3 is None:
                continue
            for col, offset in zip(_YEAR_COLUMNS, _YEAR_OFFSETS, strict=True):
                value = r[col]
                if value is None or pd.isna(value):
                    continue
                year = int(r["latest_year"]) - offset
                out_rows.append(
                    {
                        "source_id": self.source_id,
                        "sector": "apparel",
                        "item": canonical_item(r["product"]),
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
                        # EPI tables are USD Millions ("Value in US$ Millions"
                        # header, confirmed on the real 2023/2024 PDFs) — not
                        # '000s.
                        "export_value_usd": value * 1_000_000.0,
                        "price": None,
                        "price_unit": None,
                        "fx_usd_lkr": None,
                        # Keyed on product text, not table_id: EDB renumbers
                        # tables between editions (confirmed against the real
                        # 2023/2024 PDFs — table 25.89 is a different product
                        # in each), so table_id is an edition-local page
                        # locator, not a stable product identifier.
                        "source_hash": _source_hash(
                            self.source_id, r["product"], iso3, year
                        ),
                        # dedup helper only, dropped before returning
                        "_edition_year": int(r["edition_year"]),
                    }
                )
        out = pd.DataFrame.from_records(out_rows)
        if out.empty:
            return out
        out = (
            out.sort_values("_edition_year", kind="stable")
            .drop_duplicates(subset=["item", "partner_iso3", "period_start"], keep="last")
            .drop(columns="_edition_year")
            .sort_index()
            .reset_index(drop=True)
        )
        return out
