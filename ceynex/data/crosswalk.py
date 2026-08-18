"""Implements SRS 3.1.8 and 3.6.1 — reconciling UN M49, ISO 3166-1 alpha-3 and HS codes.

Source data arrives in both country coding conventions and neither may be
treated as authoritative on its own (SRS 3.6.1), so everything entering
`fact_trade` or the knowledge graph passes through here first.

**Built from a committed reference table, never an API call.** `reference/`
holds `countries.csv`, `hs_codes.csv` and `partner_aggregates.csv`. A crosswalk
that reaches the network is a crosswalk that behaves differently in CI, in the
demo, and on the marker's machine.

The trap this module exists to prevent: Comtrade reports bloc and residual
partners in the same result set as real countries. `partner = 97` (EU) is
reported *alongside* Germany, France and Italy, and `partner = 0` is the World
total. Summing them together double-counts, silently, and every market-share
figure downstream comes out wrong with nothing raising. Use
`is_aggregate_partner()` or `drop_aggregate_partners()` before aggregating
anything.
"""

from __future__ import annotations

import csv
import functools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).parent / "reference"


class CrosswalkError(KeyError):
    """An identifier that is not in the reference table.

    Deliberately loud. A silent None here becomes a NULL partner in fact_trade
    and an orphan node in the graph, and neither is noticed until a figure looks
    wrong weeks later.
    """


@dataclass(frozen=True)
class Country:
    iso3: str
    m49: int
    name: str
    status: str  # "current" | "historical"


@functools.lru_cache(maxsize=1)
def _countries() -> tuple[Country, ...]:
    with (REFERENCE_DIR / "countries.csv").open(encoding="utf-8") as fh:
        return tuple(
            Country(row["iso3"], int(row["m49"]), row["name"], row["status"])
            for row in csv.DictReader(fh)
        )


@functools.lru_cache(maxsize=1)
def _by_iso3() -> dict[str, Country]:
    return {c.iso3: c for c in _countries()}


@functools.lru_cache(maxsize=1)
def _by_m49() -> dict[int, Country]:
    return {c.m49: c for c in _countries()}


@functools.lru_cache(maxsize=1)
def _aggregate_partners() -> dict[int, str]:
    """M49 codes Comtrade reports that are not countries. See the module docstring."""
    path = REFERENCE_DIR / "partner_aggregates.csv"
    with path.open(encoding="utf-8") as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        return {int(row["m49"]): row["label"] for row in rows}


@functools.lru_cache(maxsize=1)
def _partner_aliases() -> dict[int, str]:
    """Comtrade partner codes that are countries but not their ISO 3166-1 numeric.

    Comtrade reports the USA as 842 rather than 840, France as 251 rather than
    250, India as 699 rather than 356. These are real partners; treating them as
    unknown drops them from the dataset silently.
    """
    path = REFERENCE_DIR / "partner_aliases.csv"
    with path.open(encoding="utf-8") as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        return {int(row["source_code"]): row["iso3"] for row in rows}


@functools.lru_cache(maxsize=1)
def _hs_codes() -> dict[str, tuple[str, str]]:
    """hs_code -> (description, sector)."""
    with (REFERENCE_DIR / "hs_codes.csv").open(encoding="utf-8") as fh:
        return {row["hs_code"]: (row["description"], row["sector"]) for row in csv.DictReader(fh)}


# --- countries -----------------------------------------------------------


def to_iso3(value: str | int) -> str:
    """M49 code, ISO-3 code, or country name -> ISO 3166-1 alpha-3.

    Accepts what it already is, so callers can pass a mixed column through
    without branching on which convention this particular source used.
    """
    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate in _by_iso3():
            return candidate
        if candidate.isdigit():
            return to_iso3(int(candidate))
        for country in _countries():
            if country.name.upper() == candidate:
                return country.iso3
        raise CrosswalkError(f"no ISO-3 for {value!r}")

    country = _by_m49().get(int(value))
    if country is None:
        alias = _partner_aliases().get(int(value))
        if alias is not None:
            return alias
        if int(value) in _aggregate_partners():
            raise CrosswalkError(
                f"M49 {value} is {_aggregate_partners()[int(value)]!r}, an aggregate partner, "
                "not a country — exclude it before aggregating (see drop_aggregate_partners)"
            )
        raise CrosswalkError(f"no ISO-3 for M49 {value}")
    return country.iso3


def to_m49(value: str | int) -> int:
    """ISO-3 code, M49 code, or country name -> UN M49 numeric.

    `to_m49("LKA") == 144`.
    """
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        code = int(value)
        if code in _by_m49():
            return code
        raise CrosswalkError(f"M49 {code} is not in the reference table")
    return _by_iso3()[to_iso3(value)].m49


def country_name(value: str | int) -> str:
    return _by_iso3()[to_iso3(value)].name


def is_known_country(value: str | int) -> bool:
    try:
        to_iso3(value)
    except (CrosswalkError, KeyError):
        return False
    return True


# --- Comtrade partner aggregates -----------------------------------------


def is_partner_alias(code: str | int) -> bool:
    """True for a Comtrade variant code that maps onto a real country."""
    try:
        return int(code) in _partner_aliases()
    except (TypeError, ValueError):
        return False


def partner_alias_note(code: str | int) -> str | None:
    """Why this code differs from the ISO numeric, for the data-quality log."""
    return _partner_aliases().get(int(code))


def is_aggregate_partner(m49: str | int) -> bool:
    """True for World, EU-as-reported, and the residual `nes` buckets."""
    try:
        return int(m49) in _aggregate_partners()
    except (TypeError, ValueError):
        return False


def aggregate_partner_label(m49: str | int) -> str | None:
    return _aggregate_partners().get(int(m49))


def drop_aggregate_partners(frame: pd.DataFrame, column: str = "partner_m49") -> pd.DataFrame:
    """Remove bloc and residual partner rows before any aggregation.

    Call this on every Comtrade extract. Keeping `partner = 97` alongside the EU
    member states, or `partner = 0` alongside real partners, inflates every total
    and every share derived from it.
    """
    if column not in frame.columns:
        raise KeyError(f"{column!r} not in frame; columns are {list(frame.columns)}")
    codes = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[~codes.isin(_aggregate_partners().keys())].copy()


# --- HS codes ------------------------------------------------------------


def normalize_hs(code: str | int, digits: int = 6) -> str:
    """Normalize an HS code to `digits` significant digits, zero-padded.

    Comtrade returns codes as integers, which drops the leading zero that tea
    (0902) and cinnamon (0906) both have — `902` and `906` join against nothing.
    Restoring it is the entire reason this function exists.

    Truncates rather than rounds, because HS is hierarchical: the first two
    digits are the chapter, the first four the heading. `normalize_hs("610910", 4)`
    is `"6109"`, and `normalize_hs("6109", 2)` is `"61"`.
    """
    if digits not in (2, 4, 6, 8, 10):
        raise ValueError(f"HS codes are addressed at 2, 4, 6, 8 or 10 digits, not {digits}")

    text = str(code).strip().replace(".", "").replace(" ", "")
    if not text.isdigit():
        raise CrosswalkError(f"{code!r} is not an HS code")

    # An odd length means a leading zero was lost somewhere upstream.
    if len(text) % 2:
        text = "0" + text

    if len(text) < digits:
        raise CrosswalkError(
            f"HS {text!r} has {len(text)} digits; cannot express it at {digits}. "
            "Truncating is safe, inventing precision is not"
        )
    return text[:digits]


def hs_chapter(code: str | int) -> str:
    """The 2-digit chapter: `hs_chapter("610910") == "61"`."""
    return normalize_hs(code, digits=2)


def hs_sector(code: str | int) -> str:
    """`agriculture` or `apparel`, resolved by walking up the HS hierarchy.

    A 6-digit code that is not itself in the reference table resolves through its
    4-digit heading and then its 2-digit chapter, so a new subheading appearing
    in a fresh Comtrade pull classifies correctly without a reference edit.
    """
    table = _hs_codes()
    text = str(code).strip()
    for digits in (6, 4, 2):
        try:
            candidate = normalize_hs(text, digits=digits)
        except CrosswalkError:
            continue
        if candidate in table:
            return table[candidate][1]
    raise CrosswalkError(f"HS {code!r} is outside CeyNex's sector scope (SRS 2.4)")


def hs_description(code: str | int) -> str:
    table = _hs_codes()
    for digits in (6, 4, 2):
        try:
            candidate = normalize_hs(code, digits=digits)
        except CrosswalkError:
            continue
        if candidate in table:
            return table[candidate][0]
    raise CrosswalkError(f"no description for HS {code!r}")


def is_in_scope(code: str | int) -> bool:
    """SRS 2.4 scope: tea, cinnamon, rubber, coconut, and HS 61/62 apparel."""
    try:
        hs_sector(code)
    except (CrosswalkError, ValueError):
        return False
    return True


# --- dimension tables ----------------------------------------------------


def dim_country_rows() -> list[tuple[str, int, str]]:
    """Seed rows for `dim_country`. Current territories only."""
    return [(c.iso3, c.m49, c.name) for c in _countries() if c.status == "current"]


def dim_hs_rows() -> list[tuple[str, str, str]]:
    """Seed rows for `dim_hs`."""
    return [(code, description, sector) for code, (description, sector) in _hs_codes().items()]
