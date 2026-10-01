"""The snapshot fetcher: find the current link, keep only a new, parseable file.

No network: an `httpx.MockTransport` serves the page and the workbook. The
workbook is generated in the layout `PinkSheetConnector` parses (a "Monthly
Prices" sheet with a "Tea, Colombo" header row), so validation runs the real
connector code.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd
import pytest

from ceynex.data import fetch_snapshots
from ceynex.data.fetch_snapshots import FetchError, discover, fetch

PAGE = "https://www.worldbank.org/en/research/commodity-markets"
WORKBOOK_URL = (
    "https://thedocs.worldbank.org/en/doc/abc-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
)


def workbook(tea_prices: tuple[float, ...] = (3.1, 3.2)) -> bytes:
    """Build once per test and reuse the bytes: an xlsx records when it was
    written, so two builds a second apart are different files."""
    rows = [["World Bank Commodity Price Data (The Pink Sheet)", None, None],
            [None, "Tea, Colombo", "Rubber, TSR20"],
            [None, "($/kg)", "($/kg)"]]
    rows += [[f"2026M{month:02d}", price, 1.7] for month, price in enumerate(tea_prices, start=7)]
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Monthly Prices", header=False, index=False)
    return buffer.getvalue()


def client(page_html: str, payload: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == PAGE:
            return httpx.Response(200, text=page_html)
        if str(request.url) == WORKBOOK_URL:
            return httpx.Response(200, content=payload)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


LINKED = f'<p><a href="{WORKBOOK_URL}">Monthly prices</a></p>'


def test_discover_finds_the_current_edition_and_makes_it_absolute():
    html = '<a href="/en/doc/x/related/CMO-Historical-Data-Monthly.xlsx?download=1">x</a>'
    found = discover(html, "https://thedocs.worldbank.org/page", "CMO-Historical-Data-Monthly.xlsx")
    assert found == "https://thedocs.worldbank.org/en/doc/x/related/CMO-Historical-Data-Monthly.xlsx?download=1"


def test_a_page_without_the_link_is_a_loud_failure():
    with pytest.raises(FetchError, match="may have moved"):
        discover("<a href='/other.xlsx'>", PAGE, "CMO-Historical-Data-Monthly.xlsx")


def test_a_new_edition_becomes_a_dated_snapshot(tmp_path: Path):
    published = workbook()
    result = fetch("pink_sheet", tmp_path, client=client(LINKED, published), today="2026-10-01")

    assert result.status == "new"
    assert result.path == tmp_path / "pinksheet" / "2026-10-01" / "CMO-Historical-Data-Monthly.xlsx"
    assert result.path.read_bytes() == published
    assert result.url == WORKBOOK_URL


def test_an_unchanged_edition_adds_no_folder(tmp_path: Path):
    existing = tmp_path / "pinksheet" / "2026-09-01" / "CMO-Historical-Data-Monthly.xlsx"
    existing.parent.mkdir(parents=True)
    published = workbook()
    existing.write_bytes(published)

    result = fetch("pink_sheet", tmp_path, client=client(LINKED, published), today="2026-10-01")

    assert result.status == "unchanged"
    assert result.path == existing
    assert sorted(p.name for p in (tmp_path / "pinksheet").iterdir()) == ["2026-09-01"]


def test_a_revised_edition_is_kept_beside_the_old_one(tmp_path: Path):
    existing = tmp_path / "pinksheet" / "2026-09-01" / "CMO-Historical-Data-Monthly.xlsx"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(workbook((3.1, 3.2)))

    result = fetch("pink_sheet", tmp_path, client=client(LINKED, workbook((3.1, 3.3))),
                   today="2026-10-01")

    assert result.status == "new"
    assert sorted(p.name for p in (tmp_path / "pinksheet").iterdir()) == ["2026-09-01", "2026-10-01"]


def test_a_download_that_does_not_parse_is_refused_and_nothing_is_saved(tmp_path: Path):
    with pytest.raises(FetchError, match="does not parse"):
        fetch("pink_sheet", tmp_path, client=client(LINKED, b"<html>not a workbook</html>"),
              today="2026-10-01")
    assert not (tmp_path / "pinksheet").exists()


def test_dry_run_saves_nothing(tmp_path: Path):
    result = fetch("pink_sheet", tmp_path, client=client(LINKED, workbook()),
                   today="2026-10-01", dry_run=True)
    assert result.status == "new (dry run)"
    assert not (tmp_path / "pinksheet").exists()


def test_every_configured_source_has_a_validator():
    for name in fetch_snapshots.sources():
        with pytest.raises(FetchError) as caught:
            fetch_snapshots._validate(name, Path("/nonexistent.xlsx"))
        assert "no validator" not in str(caught.value)
