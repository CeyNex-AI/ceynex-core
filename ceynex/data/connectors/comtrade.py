"""Implements SRS 3.1.7 — UN Comtrade bilateral trade flows for Sri Lanka.

Sri Lanka (reporter M49 144) exports, annual, for the HS codes CeyNex covers:
tea 0902, cinnamon 0906, rubber 4001, and apparel chapters 61 and 62.

Two endpoints, chosen by whether a key is present:

- **subscription** (`COMTRADE_API_KEY` set) — 500 calls/day, 100k records/call
- **public preview** (no key) — same data, capped at 500 records per call

The preview endpoint is why this connector works with an empty key, and why the
demo is built on real UN figures rather than invented ones. It is a genuine
fallback, not a stub: R2 in the risk register is "Comtrade registration blocks
the team for a day", and this is the mitigation.

Raw responses are written to `data/raw/comtrade/<YYYY-MM-DD>/` **before** parsing.
When a figure looks wrong three weeks later, the question is always "what did the
API actually return", and the only way to answer it is to have kept the bytes.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ceynex.contracts import DataSourceConnector, SourceManifest
from ceynex.data.crosswalk import (
    CrosswalkError,
    drop_aggregate_partners,
    hs_sector,
    is_aggregate_partner,
    normalize_hs,
    to_iso3,
)
from ceynex.settings import comtrade_api_key, data_dir

log = logging.getLogger(__name__)

SOURCE_ID = "UN_COMTRADE"
REPORTER_M49 = 144  # Sri Lanka
REPORTER_ISO3 = "LKA"

SUBSCRIPTION_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
PREVIEW_URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

# SRS 2.4 scope. Chapter-level 61/62 pulls come back as their own aggregate rows,
# which is what the apparel agent wants; M3 layers his 6-digit view over the same
# extract rather than issuing a second set of calls (settled at the Day 2 standup).
DEFAULT_HS_CODES = ("0902", "0906", "4001", "61", "62")

ITEM_BY_HS = {
    "0902": "tea",
    "0906": "cinnamon",
    "4001": "rubber",
    "61": "apparel_knit",
    "62": "apparel_woven",
}


class ComtradeConnector(DataSourceConnector):
    """Pulls Sri Lankan export flows and maps them onto `fact_trade`."""

    source_id = SOURCE_ID
    refresh_mode = "scheduled"

    #: (hs_code, year) pairs that returned no rows on the last fetch.
    empty_pulls: list[tuple[str, int]]

    def __init__(
        self,
        hs_codes: tuple[str, ...] = DEFAULT_HS_CODES,
        years: tuple[int, ...] | None = None,
        *,
        api_key: str | None = None,
        cache_root: Path | None = None,
        timeout_s: float = 30.0,
        offline: bool = False,
    ) -> None:
        self.hs_codes = hs_codes
        self.years = years or _default_years()
        self.api_key = api_key if api_key is not None else comtrade_api_key()
        self.cache_root = cache_root or (data_dir() / "raw" / "comtrade")
        self.timeout_s = timeout_s
        self.offline = offline
        self.empty_pulls = []
        self._manifest: SourceManifest | None = None

    # --- provenance ------------------------------------------------------

    @property
    def uses_subscription(self) -> bool:
        return bool(self.api_key)

    @property
    def endpoint(self) -> str:
        return SUBSCRIPTION_URL if self.uses_subscription else PREVIEW_URL

    def cache_dir(self, when: str | None = None) -> Path:
        stamp = when or datetime.now(UTC).strftime("%Y-%m-%d")
        return self.cache_root / stamp

    # --- fetching --------------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """One call per (hs_code, year). Cached responses are reused, not re-fetched."""
        frames: list[pd.DataFrame] = []
        self.empty_pulls: list[tuple[str, int]] = []
        for hs_code in self.hs_codes:
            for year in self.years:
                payload = self._load_or_fetch(hs_code, year)
                rows = payload.get("data") or []
                if not rows:
                    # Not an error, and not nothing either: Sri Lanka has years
                    # it did not report to Comtrade at all. A gap year silently
                    # breaks CAGR endpoints and leaves a hole in every forecast
                    # window, so it is recorded rather than logged and forgotten.
                    log.info("comtrade: no rows for hs=%s year=%s", hs_code, year)
                    self.empty_pulls.append((hs_code, year))
                    continue
                frames.append(pd.DataFrame(rows))

        if not frames:
            log.warning("comtrade returned nothing for any (hs, year) pair")
            self._manifest = SourceManifest(
                source_id=self.source_id,
                fetched_at=datetime.now(UTC).isoformat(),
                row_count=0,
                frequency="A",
                notes={"endpoint": self.endpoint, "reason": "no rows"},
            )
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        self._manifest = SourceManifest(
            source_id=self.source_id,
            fetched_at=datetime.now(UTC).isoformat(),
            row_count=len(raw),
            period_start=f"{min(self.years)}-01-01",
            period_end=f"{max(self.years)}-12-31",
            frequency="A",
            notes={
                "endpoint": self.endpoint,
                "subscription": self.uses_subscription,
                "hs_codes": list(self.hs_codes),
                "years": list(self.years),
                "empty_pulls": [f"{hs}:{year}" for hs, year in self.empty_pulls],
                "missing_years": sorted(_fully_missing_years(self.empty_pulls, self.hs_codes)),
            },
        )
        return raw

    def _load_or_fetch(self, hs_code: str, year: int) -> dict[str, Any]:
        cached = self._newest_cached(hs_code, year)
        if cached is not None:
            log.debug("comtrade: cache hit %s", cached.name)
            return json.loads(cached.read_text(encoding="utf-8"))

        if self.offline:
            log.warning("comtrade offline and nothing cached for hs=%s year=%s", hs_code, year)
            return {"data": []}

        payload = self._request(hs_code, year)

        # Written before parsing, deliberately.
        directory = self.cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{hs_code}_{year}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return payload

    def _newest_cached(self, hs_code: str, year: int) -> Path | None:
        """Most recent cached pull for this (hs, year), across all dated folders."""
        if not self.cache_root.is_dir():
            return None
        matches = sorted(self.cache_root.glob(f"*/{hs_code}_{year}.json"))
        return matches[-1] if matches else None

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _request(self, hs_code: str, year: int) -> dict[str, Any]:
        params: dict[str, Any] = {
            "reporterCode": str(REPORTER_M49),
            "period": str(year),
            "cmdCode": hs_code,
            "flowCode": "X",  # exports
        }
        headers = {}
        if self.uses_subscription:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key or ""

        log.info("comtrade: GET hs=%s year=%s (%s)", hs_code, year,
                 "subscription" if self.uses_subscription else "preview")
        response = httpx.get(self.endpoint, params=params, headers=headers, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def manifest(self) -> SourceManifest:
        if self._manifest is None:
            raise RuntimeError("call fetch() before manifest()")
        return self._manifest

    # --- mapping onto the contract --------------------------------------

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map Comtrade's response onto the frozen `fact_trade` columns.

        Two things happen here that everything downstream depends on:

        1. **Aggregate partners are dropped.** `partnerCode` 0 is World and 97 is
           the EU reported alongside its own members. Keeping either
           double-counts, and nothing raises when it happens.
        2. **HS codes are normalized as strings.** Comtrade returns them as
           integers, so tea's `0902` arrives as `902` and joins against nothing.
        """
        if raw.empty:
            return pd.DataFrame(columns=FACT_TRADE_COLUMNS)

        frame = raw.copy()
        frame["partner_m49"] = pd.to_numeric(frame["partnerCode"], errors="coerce")

        before = len(frame)
        frame = drop_aggregate_partners(frame, column="partner_m49")
        dropped = before - len(frame)
        if dropped:
            log.info("comtrade: dropped %d aggregate-partner rows (World, EU, nes)", dropped)

        records = []
        unresolved: set[Any] = set()
        for row in frame.to_dict("records"):
            hs_code = normalize_hs(row["cmdCode"], digits=_digits_of(row["cmdCode"]))
            try:
                partner_iso3 = to_iso3(int(row["partner_m49"]))
            except (CrosswalkError, KeyError, ValueError):
                unresolved.add(row["partner_m49"])
                continue

            year = int(row["refYear"])
            value = _number(row.get("primaryValue")) or _number(row.get("fobvalue"))
            volume = _number(row.get("netWgt")) or _number(row.get("qty"))

            records.append(
                {
                    "source_id": SOURCE_ID,
                    "sector": _sector_of(hs_code),
                    "item": _item_of(hs_code),
                    "hs_code": hs_code,
                    "reporter_iso3": REPORTER_ISO3,
                    "reporter_m49": REPORTER_M49,
                    "partner_iso3": partner_iso3,
                    "partner_m49": int(row["partner_m49"]),
                    "period_start": f"{year}-01-01",
                    "period_end": f"{year}-12-31",
                    "frequency": "A",
                    "export_volume": volume,
                    "volume_unit": "kg",
                    "export_value_usd": value,
                    "price": _unit_price(value, volume),
                    "price_unit": "USD/kg",
                    "fx_usd_lkr": None,
                }
            )

        if unresolved:
            log.warning(
                "comtrade: %d partner codes had no ISO-3 and were skipped: %s",
                len(unresolved),
                sorted(unresolved)[:10],
            )

        return pd.DataFrame(records, columns=FACT_TRADE_COLUMNS)


FACT_TRADE_COLUMNS = [
    "source_id",
    "sector",
    "item",
    "hs_code",
    "reporter_iso3",
    "reporter_m49",
    "partner_iso3",
    "partner_m49",
    "period_start",
    "period_end",
    "frequency",
    "export_volume",
    "volume_unit",
    "export_value_usd",
    "price",
    "price_unit",
    "fx_usd_lkr",
]


def _fully_missing_years(
    empty: list[tuple[str, int]], hs_codes: tuple[str, ...]
) -> set[int]:
    """Years where *every* HS code came back empty — a reporting gap, not a
    commodity that simply was not traded."""
    by_year: dict[int, set[str]] = {}
    for hs_code, year in empty:
        by_year.setdefault(year, set()).add(hs_code)
    return {year for year, codes in by_year.items() if codes >= set(hs_codes)}


def _default_years() -> tuple[int, ...]:
    """Last ten complete years. Comtrade lags, so the current year is excluded."""
    latest = datetime.now(UTC).year - 1
    return tuple(range(latest - 9, latest + 1))


def _digits_of(code: Any) -> int:
    """The granularity Comtrade reported at: 2, 4 or 6 digits."""
    text = str(code).strip()
    padded = text if len(text) % 2 == 0 else "0" + text
    return min(max(len(padded), 2), 6)


def _sector_of(hs_code: str) -> str:
    try:
        return hs_sector(hs_code)
    except CrosswalkError:
        return "unknown"


def _item_of(hs_code: str) -> str:
    for level in (hs_code, hs_code[:4], hs_code[:2]):
        if level in ITEM_BY_HS:
            return ITEM_BY_HS[level]
    return hs_code


def _number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number != 0 else None


def _unit_price(value: float | None, volume: float | None) -> float | None:
    """USD per kg. None rather than a division-by-zero surprise."""
    if not value or not volume:
        return None
    return value / volume


__all__ = [
    "DEFAULT_HS_CODES",
    "FACT_TRADE_COLUMNS",
    "REPORTER_ISO3",
    "REPORTER_M49",
    "SOURCE_ID",
    "ComtradeConnector",
    "is_aggregate_partner",
]
