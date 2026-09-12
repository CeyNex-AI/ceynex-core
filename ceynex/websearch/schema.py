"""What a web result is, and deliberately what it is not — deviation D14.

`WebResult` is not `Evidence` and must not become it by convenience. Evidence
names a source from the frozen `SourceId` vocabulary and a query that produced
it; a web snippet has neither. The conversion is explicit and one-way
(`agents/common.py::evidence_from_web`), which is what stops a scraped paragraph
being handed to something expecting a verified figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebResult:
    """One search hit. Snippet only — v1 never fetches the page.

    That bound is a safety decision, not a shortcut: a snippet is a few hundred
    characters of untrusted text, and a full page is an unbounded amount of it.
    It also matches this project's recorded reason for not scraping news
    articles.
    """

    title: str
    url: str
    snippet: str
    published: str | None = None
    domain: str = ""


__all__ = ["WebResult"]
