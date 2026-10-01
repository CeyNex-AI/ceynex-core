"""Implements SRS 3.1.7 / FR-DAT-03: save a new dated snapshot of a source that
is published as one downloadable file, so the scheduled refresh has something
new to ingest.

    python -m ceynex.data.fetch_snapshots --sources pink_sheet
    python -m ceynex.data.fetch_snapshots --sources pink_sheet --dry-run

Pink Sheet's connector reads the newest `<raw>/pinksheet/<YYYY-MM-DD>/` folder
(`connectors/_snapshots.py`). Until this existed, a person saved that workbook
by hand, so a monthly cron would have re-ingested the same bytes every month.
The World Bank's download URL carries a per-edition document id, so the link is
found on the commodity-markets page each time (config/sources.yaml) rather than
pinned to an address that stops working at the next edition.

A download is kept only if it parses through the connector that will read it,
and only if it differs from the newest snapshot already there: an unchanged
edition adds no folder, and a page that moved or a workbook that changed shape
fails here, loudly, rather than in the ingest that follows.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx

from ceynex import settings
from ceynex.data.connectors._snapshots import latest_snapshot_dir

log = logging.getLogger(__name__)

USER_AGENT = "CeyNex data pipeline (CS3501 research project, University of Moratuwa)"
TIMEOUT_S = 60.0


class FetchError(RuntimeError):
    """The source could not be fetched, or what came back is not usable."""


@dataclass(frozen=True)
class FetchResult:
    name: str
    #: "new", "unchanged", or "new (dry run)"
    status: str
    path: Path
    sha256: str
    url: str


def sources() -> dict[str, dict]:
    return settings.load_config("sources").get("snapshot_fetch", {})


def discover(page_html: str, page_url: str, filename: str) -> str:
    """The first link on the page whose path ends in `filename`, made absolute."""
    for href in re.findall(r"""href\s*=\s*["']([^"']+)["']""", page_html):
        path = href.split("?", 1)[0].split("#", 1)[0]
        if path == filename or path.endswith("/" + filename):
            return urljoin(page_url, href)
    raise FetchError(f"{page_url} links no {filename}; the page may have moved")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(name: str, path: Path) -> None:
    """Parse the file exactly as its connector will, and refuse an empty result."""
    try:
        if name == "pink_sheet":
            from ceynex.data.connectors.pinksheet import PinkSheetConnector  # noqa: PLC0415

            with tempfile.TemporaryDirectory() as staging:
                connector = PinkSheetConnector(path, Path(staging))
                rows = connector.to_fact_trade(connector.fetch())
        else:
            raise FetchError(f"no validator for {name!r}")
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001 - any parse failure means "not usable"
        raise FetchError(f"{name}: the download does not parse ({exc})") from exc
    if rows.empty:
        raise FetchError(f"{name}: the download parsed but holds no rows to ingest")


def _latest(root: Path, filename: str) -> Path | None:
    if not root.is_dir():
        return None
    candidate = latest_snapshot_dir(root) / filename
    return candidate if candidate.is_file() else None


def fetch(
    name: str,
    raw_dir: Path,
    *,
    client: httpx.Client,
    today: str | None = None,
    dry_run: bool = False,
) -> FetchResult:
    config = sources().get(name)
    if config is None:
        raise FetchError(f"no snapshot_fetch entry for {name!r} in config/sources.yaml")
    filename = config["filename"]
    root = Path(raw_dir) / config["raw_subdir"]

    page = client.get(config["page"])
    page.raise_for_status()
    url = discover(page.text, str(page.url), filename)

    with tempfile.TemporaryDirectory() as tmp:
        download = Path(tmp) / filename
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with download.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        _validate(name, download)
        digest = _sha256(download)

        latest = _latest(root, filename)
        if latest is not None and _sha256(latest) == digest:
            return FetchResult(name, "unchanged", latest, digest, url)

        day = today or datetime.now(UTC).date().isoformat()
        destination = root / day / filename
        if dry_run:
            return FetchResult(name, "new (dry run)", destination, digest, url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(download), destination)
    return FetchResult(name, "new", destination, digest, url)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", required=True, choices=sorted(sources()))
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="defaults to the agriculture raw directory the pipeline reads")
    parser.add_argument("--dry-run", action="store_true", help="download and check, save nothing")
    args = parser.parse_args(argv)

    if args.raw_dir is None:
        from ceynex.data.pipeline import _agriculture_raw_dir  # noqa: PLC0415

        args.raw_dir = _agriculture_raw_dir()

    failed = 0
    with httpx.Client(timeout=TIMEOUT_S, follow_redirects=True,
                      headers={"User-Agent": USER_AGENT}) as client:
        for name in args.sources:
            try:
                result = fetch(name, args.raw_dir, client=client, dry_run=args.dry_run)
            except (FetchError, httpx.HTTPError) as exc:
                failed += 1
                print(f"  FAIL {name}: {exc}")
                continue
            print(f"  ok   {name}: {result.status}  {result.path}  sha256 {result.sha256[:12]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
