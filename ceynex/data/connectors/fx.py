"""World Bank official exchange rate connector — Sri Lanka USD/LKR (SRS 3.1.7).

`DataSourceConnector`'s own docstring names the Central Bank of Sri Lanka
(CBSL) as the anticipated FX source. Checked 2026-09-24: cbsl.gov.lk's daily
and historical exchange-rate pages are interactive charts/tables with no
confirmed bulk-download or API endpoint. The World Bank's official exchange
rate indicator (`PA.NUS.FCRF`, LCU per US$, period average — sourced from IMF
International Financial Statistics) is a real, free, no-auth substitute: one
HTTP GET, no key, JSON. It lags about two years and is annual, not daily, so
it answers "what was the rate in a given year", not "today's rate".

Cached like Comtrade: the raw response is written to
`data/raw/fx/<YYYY-MM-DD>/` before parsing, and `--offline` reuses the newest
cached response instead of calling the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ceynex.contracts import DataSourceConnector, SourceManifest
from ceynex.settings import data_dir

log = logging.getLogger(__name__)

SOURCE_ID = "WB_FX"
REPORTER_ISO3 = "LKA"
REPORTER_M49 = 144
ITEM = "usd_lkr"
INDICATOR = "PA.NUS.FCRF"
API_URL = f"https://api.worldbank.org/v2/country/LKA/indicator/{INDICATOR}"


class FXConnector(DataSourceConnector):
    """Sri Lanka's official annual USD/LKR rate, from the World Bank."""

    source_id = SOURCE_ID
    refresh_mode = "scheduled"

    def __init__(
        self,
        years: tuple[int, ...] | None = None,
        *,
        cache_root: Path | None = None,
        timeout_s: float = 30.0,
        offline: bool = False,
    ) -> None:
        # years is accepted for CLI-flag compatibility (`--years FROM TO`
        # applies to every source); the API always returns full history in one
        # call, so this only trims the result rather than shaping the request.
        self.years = years
        self.cache_root = cache_root or (data_dir() / "raw" / "fx")
        self.timeout_s = timeout_s
        self.offline = offline
        self._last: pd.DataFrame | None = None
        self._fetched_at: datetime | None = None
        self._source_hash: str | None = None

    def cache_dir(self, when: str | None = None) -> Path:
        stamp = when or datetime.now(UTC).strftime("%Y-%m-%d")
        return self.cache_root / stamp

    def fetch(self) -> pd.DataFrame:
        payload_bytes = self._load_or_fetch()
        self._source_hash = hashlib.sha256(payload_bytes).hexdigest()
        payload = json.loads(payload_bytes)
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        frame = pd.DataFrame(rows)
        if self.years and not frame.empty:
            year = pd.to_numeric(frame["date"], errors="coerce")
            frame = frame[year.between(min(self.years), max(self.years))].copy()
        self._last, self._fetched_at = frame, datetime.now(UTC)
        return frame

    def _load_or_fetch(self) -> bytes:
        cached = self._newest_cached()
        if cached is not None:
            log.debug("fx: cache hit %s", cached.name)
            return cached.read_bytes()
        if self.offline:
            log.warning("fx: offline and nothing cached")
            return b"[]"
        payload_bytes = self._request()
        directory = self.cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "wb_fx.json").write_bytes(payload_bytes)
        return payload_bytes

    def _newest_cached(self) -> Path | None:
        if not self.cache_root.is_dir():
            return None
        matches = sorted(self.cache_root.glob("*/wb_fx.json"))
        return matches[-1] if matches else None

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _request(self) -> bytes:
        log.info("fx: GET %s", API_URL)
        response = httpx.get(
            API_URL, params={"format": "json", "per_page": "100"}, timeout=self.timeout_s
        )
        response.raise_for_status()
        return response.content

    def manifest(self) -> SourceManifest:
        if self._last is None or self._fetched_at is None:
            raise RuntimeError("call fetch() before manifest().")
        years = pd.to_numeric(self._last.get("date"), errors="coerce")
        return SourceManifest(
            source_id=self.source_id,
            fetched_at=self._fetched_at.isoformat(),
            row_count=len(self._last),
            period_start=str(int(years.min())) if years.notna().any() else None,
            period_end=str(int(years.max())) if years.notna().any() else None,
            frequency="A",
            notes={"indicator": INDICATOR, "endpoint": API_URL},
        )

    def to_fact_trade(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map the World Bank's annual USD/LKR rate onto `fact_trade`.

        This is a macro exchange-rate observation, not a trade flow: every
        commodity/HS/partner column stays null, and only `fx_usd_lkr` carries
        a value. `item="usd_lkr"`, `sector="macro"` keep it out of every real
        commodity's identity key (SRS 2.4's four agriculture items plus
        apparel_knit/apparel_woven) so it can never be mistaken for one.
        """
        if raw.empty or "value" not in raw.columns:
            return pd.DataFrame()
        frame = raw.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["value", "date"]).copy()
        if frame.empty:
            return pd.DataFrame()
        frame["year"] = pd.to_numeric(frame["date"], errors="raise").astype("int64")
        periods = pd.to_datetime(frame["year"].astype(str) + "-01-01")
        return pd.DataFrame(
            {
                "source_id": self.source_id,
                "sector": "macro",
                "item": ITEM,
                "hs_code": None,
                "reporter_iso3": REPORTER_ISO3,
                "reporter_m49": REPORTER_M49,
                "partner_iso3": None,
                "partner_m49": None,
                "period_start": periods,
                "period_end": periods + pd.offsets.YearEnd(),
                "frequency": "A",
                "export_volume": None,
                "volume_unit": None,
                "export_value_usd": None,
                "price": None,
                "price_unit": None,
                "fx_usd_lkr": frame["value"].to_numpy(),
                "source_hash": self._source_hash,
            }
        )


__all__ = ["API_URL", "INDICATOR", "ITEM", "REPORTER_ISO3", "REPORTER_M49", "SOURCE_ID", "FXConnector"]
