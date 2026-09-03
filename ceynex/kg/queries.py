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
    prefixes = hs_hierarchy(hs_code)
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


def policy_documents_for(iso3: str | None = None, hs_code: str | None = None) -> Query:
    """Which policy documents could answer a question about this country and code.

    **The graph-anchoring step (deviation D10).** The vector search is never run
    unfiltered: this query decides which documents are eligible, and the caller
    passes the resulting `doc_id` list to Qdrant as a filter. Trade-policy
    documents all read alike — objectives, market access, competitiveness — so an
    unanchored search will happily answer a question about Germany with Canadian
    text that scores marginally higher. Restricting the candidate set first is
    what makes the retrieval about the right country rather than merely about the
    right topic.

    Both filters are optional and are `OR`-ed against the document's own scope,
    not `AND`-ed: a general trade strategy carries no `APPLIES_TO` edge at all,
    and requiring one would exclude exactly the documents that discuss policy
    broadly. A document qualifies if it was issued by the country **or** covers
    the code; with neither argument, every indexed document qualifies.

    Only `indexed` documents come back. A row that was fetched but not embedded —
    the German AWG, which the English-only model cannot represent — is in the
    graph as a record that it was found and skipped, and citing it would point a
    reader at text no search can reach.
    """
    cypher = """
    MATCH (p:PolicyDocument)
    WHERE p.indexed = true
      AND (
        $iso3 IS NULL AND $hs_prefixes IS NULL
        OR ($iso3 IS NOT NULL AND $iso3 IN p.iso3)
        OR ($hs_prefixes IS NOT NULL AND EXISTS {
              MATCH (p)-[:APPLIES_TO]->(h:HSCode) WHERE h.code IN $hs_prefixes
           })
      )
    RETURN p.doc_id      AS doc_id,
           p.title       AS title,
           p.publisher   AS publisher,
           p.url         AS url,
           p.iso3        AS iso3,
           p.verified    AS verified,
           p.chunk_count AS chunk_count
    ORDER BY p.doc_id
    """
    params: dict[str, Any] = {"iso3": None, "hs_prefixes": None}
    if iso3:
        params["iso3"] = to_iso3(iso3)
    if hs_code:
        params["hs_prefixes"] = hs_hierarchy(hs_code)
    return cypher, params


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


def latest_observation_year(item: str | None = None) -> Query:
    """The most recent year on any EXPORTS_TO edge -- or one item's, if given.

    Feeds the staleness penalty in `ceynex/orchestrator/confidence.py`: an answer
    drawn from data three years old should not score as highly as one drawn from
    last quarter's.

    Unscoped (`item=None`) is a graph-wide max, not any one item's -- callers
    that then use the result to query a *specific* item (a "what's the latest
    year, now show me that item's figures for it" pattern, as
    `export_analytics`/`trade_economics` both do) must pass `item`, not rely on
    the graph-wide max being that item's own latest year. Found live
    2026-08-26: a single item with a later (and partly erroneous -- see
    `ceynex/data/connectors/jaaf.py`) latest year silently made every *other*
    item's cross-sector analytics query a year with no data for that item,
    producing a false "no data" answer instead of the real, available figures.
    """
    item_filter = "WHERE toLower(i.name) = toLower($item)" if item is not None else ""
    cypher = f"""
    MATCH (i)-[e:EXPORTS_TO]->()
    {item_filter}
    RETURN max(e.year) AS latest_year
    """
    return cypher, {"item": item} if item is not None else {}


# --- subgraph projections -------------------------------------------------
#
# Every query above returns scalars — `c.iso3 AS partner_iso3` — because a figure
# is what an agent needs. Drawing the graph needs the shape as well, and that has
# to be asked for explicitly: `KnowledgeGraphClient.run()` ends in
# `record.data()`, which flattens a Node to `dict(node)` — its properties, with
# the labels and the element id gone. A `RETURN n` here would arrive as a bare
# property bag that no longer knows it was a Country.
#
# So each of these projects `labels()`, `properties()` and the relationship type
# by hand, and every one returns the same nine columns:
#
#   source_label, source_key, source_name,
#   rel_type, rel_props,
#   target_label, target_key, target_name,
#   weight
#
# One shape means `kg/subgraph.py` has one decoder rather than one per facet, and
# means a new facet is a query here rather than a query plus a branch there.
#
# `*_key` is the label's own uniqueness key from schema.cypher (`Country.iso3`,
# `HSCode.code`, everything else `.name`) — never `elementId()`, which changes
# across a reload and so could not survive a click-to-expand round trip.

#: Each constrained label's uniqueness key. `neighbours()` needs this because
#: Cypher cannot parameterize a label or a property name; keeping the mapping
#: here, next to the queries it shapes, is what makes the substitution an
#: allowlist lookup rather than string handling. PolicyDocument is included
#: even though schema.cypher has no constraint for it yet (deviation D10) —
#: the loader merges on `doc_id`, so it is the key in practice.
NODE_KEYS: dict[str, str] = {
    "Country": "iso3",
    "HSCode": "code",
    "Commodity": "name",
    "ApparelCategory": "name",
    "District": "name",
    "TradeAgreement": "name",
    "PolicyDocument": "doc_id",
}


def export_subgraph(item: str, year: int, limit: int = 8) -> Query:
    """An item and the markets it went to, as drawable triples (SRS 3.1.4).

    The same `EXPORTS_TO` edges `top_partners` counts, carrying `e.value` as the
    weight so the drawing can make a big market look like one. Limited for the
    same reason `top_partners` is: a picture of sixty destinations shows nothing.
    """
    cypher = """
    MATCH (i)-[e:EXPORTS_TO]->(c:Country)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
      AND e.year = $year
    RETURN labels(i)[0]  AS source_label,
           i.name        AS source_key,
           i.name        AS source_name,
           'EXPORTS_TO'  AS rel_type,
           properties(e) AS rel_props,
           'Country'     AS target_label,
           c.iso3        AS target_key,
           c.name        AS target_name,
           e.value       AS weight
    ORDER BY e.value DESC
    LIMIT $limit
    """
    return cypher, {"item": item, "year": year, "limit": limit}


def classification_subgraph(item: str) -> Query:
    """An item's HS codes and the agreements covering them, as triples.

    `UNION` rather than `OPTIONAL MATCH` on the second leg. Optional-matching the
    agreement would return a row per HS code with null agreement columns whenever
    nothing covers it, and the decoder would need a null branch to avoid emitting
    a `TradeAgreement:None` node. A union of two whole-triple branches yields
    only edges that exist, so an uncovered code is simply one row rather than one
    row plus a special case. The classification leg carries no weight — an HS
    code is not more or less classified than another.
    """
    cypher = """
    MATCH (i)-[:CLASSIFIED_AS]->(h:HSCode)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
    RETURN labels(i)[0]     AS source_label,
           i.name           AS source_key,
           i.name           AS source_name,
           'CLASSIFIED_AS'  AS rel_type,
           {}               AS rel_props,
           'HSCode'         AS target_label,
           h.code           AS target_key,
           h.description    AS target_name,
           null             AS weight
    UNION
    MATCH (i)-[:CLASSIFIED_AS]->(h:HSCode)-[cov:COVERED_BY]->(t:TradeAgreement)
    WHERE (i:Commodity OR i:ApparelCategory)
      AND toLower(i.name) = toLower($item)
    RETURN 'HSCode'         AS source_label,
           h.code           AS source_key,
           h.description    AS source_name,
           'COVERED_BY'     AS rel_type,
           properties(cov)  AS rel_props,
           'TradeAgreement' AS target_label,
           t.name           AS target_key,
           t.name           AS target_name,
           null             AS weight
    """
    return cypher, {"item": item}


def production_subgraph(commodity: str) -> Query:
    """Where a commodity is grown, as triples — the drawable half of
    `district_concentration`. Agriculture only; apparel is not modelled by
    district, so an apparel item correctly returns no rows rather than an error.
    """
    cypher = """
    MATCH (c:Commodity)-[p:PRODUCED_IN]->(d:District)
    WHERE toLower(c.name) = toLower($commodity)
    RETURN 'Commodity'    AS source_label,
           c.name         AS source_key,
           c.name         AS source_name,
           'PRODUCED_IN'  AS rel_type,
           properties(p)  AS rel_props,
           'District'     AS target_label,
           d.name         AS target_key,
           d.name         AS target_name,
           p.share        AS weight
    ORDER BY p.share DESC
    """
    return cypher, {"commodity": commodity}


def neighbours(label: str, key: str, limit: int = 12) -> Query:
    """One hop out from a node, in both directions — backs click-to-expand.

    `label` and its key property are substituted into the query text because
    Cypher has no parameter form for either: `MATCH (n:$label)` is a syntax
    error, not a slow query. That substitution is safe here and only here
    because both come from `NODE_KEYS` — an unknown label raises below rather
    than reaching the database, so nothing a caller sends can become query
    structure. The value itself stays a `$param`, as everywhere else in this
    module. `cagr` and `agreement_coverage` build their filters the same way.

    Matched undirected (`-[r]-`) so expanding a Country finds the items that
    export to it, not the nothing that it exports. Direction is then recovered
    from `startNode`/`endNode` — without it every expanded edge would be drawn
    pointing away from whatever the user happened to click.

    The neighbour's own key is a `coalesce` over the same seven properties
    `NODE_KEYS` names: which one applies depends on the neighbour's label, which
    is not known until the row comes back.

    Ordered newest-first because this is deliberately *not* year-scoped — a
    click means "what else is this connected to", and filtering to one year
    would hide a partner that stopped trading. But an `EXPORTS_TO` pair has one
    edge per year, and the assembler keeps one edge per pair, so without this
    ordering the edge that survives is whichever year the planner happened to
    return first: found live, a tea->Iraq edge drawn at $104M next to an answer
    graph showing $150M for the same pair. Newest-first makes the kept edge the
    most recent one, and `kg/subgraph.py` puts its year in the label so the two
    figures are never silently different.
    """
    if label not in NODE_KEYS:
        raise CrosswalkError(
            f"{label!r} is not a node label in this graph; expected one of {sorted(NODE_KEYS)}"
        )
    key_property = NODE_KEYS[label]
    cypher = f"""
    MATCH (n:{label} {{{key_property}: $key}})-[r]-()
    WITH startNode(r) AS s, endNode(r) AS t, r
    RETURN labels(s)[0] AS source_label,
           coalesce(s.iso3, s.code, s.doc_id, s.name) AS source_key,
           coalesce(s.name, s.description, s.title, s.code, s.doc_id) AS source_name,
           type(r)       AS rel_type,
           properties(r) AS rel_props,
           labels(t)[0]  AS target_label,
           coalesce(t.iso3, t.code, t.doc_id, t.name) AS target_key,
           coalesce(t.name, t.description, t.title, t.code, t.doc_id) AS target_name,
           r.value       AS weight
    ORDER BY coalesce(r.year, 0) DESC, weight DESC
    LIMIT $limit
    """
    return cypher, {"key": key, "limit": limit}


def hs_hierarchy(hs_code: str | int) -> list[str]:
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
