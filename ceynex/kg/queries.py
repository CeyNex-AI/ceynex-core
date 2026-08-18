"""Implements SRS 3.1.6 and 3.1.9 — the named Cypher library both KG-driven agents use.

Every function returns `(cypher, params)` rather than executing anything. Three
reasons, in order of how much they matter:

1. **The Cypher text is evidence.** SRS 3.1.4 requires each answer to carry what
   produced it, and the Export Analytics agent's architectural claim (SRS 3.1.6)
   is that relational questions are answered *from the graph*, not from a model.
   Returning the query makes that claim checkable rather than asserted.
2. **They are testable without a database.** M3's spot-check suite and our own
   tests assert the shape of these queries directly.
3. **One definition per question.** Two agents asking "what is the market share"
   in two slightly different ways is how two numbers that should agree stop
   agreeing.

Nothing here interpolates a value into a string. Every filter is a `$param`.
"""

from __future__ import annotations

from typing import Any

from ceynex.data.crosswalk import CrosswalkError, normalize_hs, to_iso3

Query = tuple[str, dict[str, Any]]

# Sri Lanka is always the reporter; CeyNex answers questions about its exports.
REPORTER_ISO3 = "LKA"


def top_partners(item: str, year: int, limit: int = 5) -> Query:
    """The largest destination markets for an item in a year, by export value.

    Ordering is by value rather than volume because the sectors are not
    comparable by weight — a tonne of tea and a tonne of T-shirts are not the
    same question.
    """
    cypher = """
    MATCH (i)-[e:EXPORTS_TO]->(c:Country)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND e.year = $year
    RETURN c.iso3 AS partner_iso3,
           c.name AS partner,
           e.value AS export_value_usd,
           e.volume AS export_volume
    ORDER BY e.value DESC
    LIMIT $limit
    """
    return cypher, {"item": item, "year": year, "limit": limit}


def market_share(item: str, year: int) -> Query:
    """Each partner's share of an item's total export value in a year.

    The denominator is computed inside the query, over the same filtered set, so
    the shares sum to 1 by construction. Computing the total separately is how a
    stale denominator produces shares that sum to 103%.
    """
    cypher = """
    MATCH (i)-[e:EXPORTS_TO]->(:Country)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND e.year = $year
    WITH sum(e.value) AS total
    MATCH (i)-[e:EXPORTS_TO]->(c:Country)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND e.year = $year
    RETURN c.iso3 AS partner_iso3,
           c.name AS partner,
           e.value AS export_value_usd,
           total AS total_export_value_usd,
           CASE WHEN total > 0 THEN e.value / total ELSE 0.0 END AS share
    ORDER BY share DESC
    """
    return cypher, {"item": item, "year": year}


def cagr(item: str, partner: str | None, from_year: int, to_year: int) -> Query:
    """Compound annual growth rate of export value between two years.

    `partner=None` means all partners combined. The growth rate itself is
    computed by the caller, not in Cypher: CAGR is undefined when the start value
    is zero or negative, and that case has to be reported as "cannot be computed"
    rather than returned as a number (SAD §4.1). Cypher returning `null` there
    would be indistinguishable from missing data.
    """
    partner_filter = "AND c.iso3 = $partner_iso3" if partner else ""
    cypher = f"""
    MATCH (i)-[e:EXPORTS_TO]->(c:Country)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND e.year IN [$from_year, $to_year]
      {partner_filter}
    RETURN e.year AS year,
           sum(e.value) AS export_value_usd,
           sum(e.volume) AS export_volume
    ORDER BY year
    """
    params: dict[str, Any] = {"item": item, "from_year": from_year, "to_year": to_year}
    if partner:
        params["partner_iso3"] = to_iso3(partner)
    return cypher, params


def district_concentration(commodity: str) -> Query:
    """Which districts produce a commodity, and in what share (SRS 3.1.6).

    Agriculture only — apparel is not modelled by district.
    """
    cypher = """
    MATCH (c:Commodity)-[p:PRODUCED_IN]->(d:District)
    WHERE toLower(c.name) = toLower($commodity)
    RETURN d.name AS district,
           p.share AS share
    ORDER BY p.share DESC
    """
    return cypher, {"commodity": commodity}


def agreement_coverage(hs_code: str) -> Query:
    """Which trade agreements cover an HS code — the GSP+ check from SRS 3.1.9.

    Matches at every level of the HS hierarchy, because coverage is declared at
    whatever granularity the agreement uses: GSP+ may be attached to chapter 61
    while the query asks about 610910. Checking only the exact string is how
    "does GSP+ cover 6109?" wrongly answers no.
    """
    prefixes = _hs_prefixes(hs_code)
    if not prefixes:
        raise CrosswalkError(f"{hs_code!r} is not an HS code")
    normalized = prefixes[0]  # the most specific level the caller actually gave us
    cypher = """
    MATCH (h:HSCode)-[cov:COVERED_BY]->(t:TradeAgreement)
    WHERE h.code IN $hs_prefixes
    RETURN $hs_code AS queried_code,
           t.name AS agreement,
           t.type AS agreement_type,
           t.in_force_from AS in_force_from,
           t.verified AS agreement_verified,
           h.code AS matched_on,
           cov.from_year AS covered_from,
           cov.to_year AS covered_to
    ORDER BY size(h.code) DESC, t.name
    """
    return cypher, {"hs_prefixes": prefixes, "hs_code": normalized}


def competing_exporters(item: str, year: int, limit: int = 5) -> Query:
    """Other countries exporting the same item to Sri Lanka's partners.

    Answers "who are we competing with" (Bangladesh and Vietnam for apparel,
    Kenya and India for tea) without needing a model.
    """
    cypher = """
    MATCH (i)-[:CLASSIFIED_AS]->(h:HSCode)<-[:CLASSIFIED_AS]-(other)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND other <> i
    MATCH (other)-[e:EXPORTS_TO]->(c:Country)
    WHERE e.year = $year
    RETURN other.name AS competitor_item,
           c.iso3 AS partner_iso3,
           sum(e.value) AS export_value_usd
    ORDER BY export_value_usd DESC
    LIMIT $limit
    """
    return cypher, {"item": item, "year": year, "limit": limit}


def items_in_sector(sector: str) -> Query:
    """Every commodity or apparel category the graph knows about in a sector.

    The router uses this to decide whether a query names something in scope
    before it dispatches an agent at it.
    """
    cypher = """
    MATCH (i)
    WHERE (i:Commodity AND $sector = 'agriculture')
       OR (i:ApparelCategory AND $sector = 'apparel')
    RETURN labels(i)[0] AS label, i.name AS name, i.hs_code AS hs_code
    ORDER BY name
    """
    return cypher, {"sector": sector}


def graph_summary() -> Query:
    """Node and relationship counts. Backs /health and the staleness term in confidence."""
    cypher = """
    MATCH (n)
    WITH labels(n)[0] AS label, count(*) AS nodes
    RETURN label, nodes
    ORDER BY label
    """
    return cypher, {}


def latest_observation_year() -> Query:
    """The most recent year on any EXPORTS_TO edge.

    Feeds the staleness penalty in `ceynex/orchestrator/confidence.py`: an answer
    drawn from data three years old should not score as highly as one drawn from
    last quarter's.
    """
    cypher = """
    MATCH ()-[e:EXPORTS_TO]->()
    RETURN max(e.year) AS latest_year
    """
    return cypher, {}


def _hs_prefixes(hs_code: str | int) -> list[str]:
    """Every level of the HS hierarchy for a code, longest first.

    `610910` -> `["610910", "6109", "61"]`. Coverage declared at any of these
    levels applies to the code.
    """
    text = str(hs_code).strip()
    prefixes: list[str] = []
    for digits in (6, 4, 2):
        try:
            prefixes.append(normalize_hs(text, digits=digits))
        except (CrosswalkError, ValueError):
            # A 4-digit code has no 6-digit form; that is expected, not an error.
            continue
    # dict.fromkeys rather than set(), to keep longest-first ordering
    return list(dict.fromkeys(prefixes))
