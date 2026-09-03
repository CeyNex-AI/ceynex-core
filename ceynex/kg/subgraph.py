"""Supports SRS 3.1.4 and 3.1.6 — the drawable half of a knowledge-graph answer.

`Evidence.detail` carries the Cypher that produced a figure, which makes the
"answered from the graph" claim checkable by anyone who reads Cypher. This module
is the same claim for everyone else: the nodes and edges that query walked, in a
shape a browser can draw.

It is deliberately one layer below the API. `kg/queries.py` owns the Cypher,
`kg/client.py` runs it, and this turns the rows into a node/edge set — no pydantic,
no FastAPI, and nothing imported from `agents/`, which sits above the Knowledge
Layer (SAD §8). The API converts these dataclasses to its own response models, the
way `retrieval/schema.py`'s `PolicyChunk` is converted rather than returned.

**Node ids are semantic, not database ids.** `"Country:USA"` is the label plus that
label's uniqueness key from schema.cypher. `elementId()` would be shorter and
wrong: it changes when the graph is reloaded, so a node id the browser held from
one answer could not be expanded against the next, and `make kg-load` runs often.

**A missing facet is a smaller picture, never no picture.** The facets run
concurrently and independently; one raising leaves the others' nodes on screen.
That is the same partial-result posture as SAD §4.1, applied to a drawing.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ceynex.kg import queries as q
from ceynex.kg.client import KnowledgeGraphUnavailableError

log = logging.getLogger(__name__)

#: How many edges one facet may contribute. Chosen for legibility rather than for
#: the database: a radial layout stops being readable somewhere around a dozen
#: spokes, well before the query cost matters.
DEFAULT_FACET_LIMIT = 8

#: The expand endpoint's cap. Higher than a facet's because the user asked for
#: this one specifically, and a country with fifteen trading partners should not
#: come back looking like it has eight.
DEFAULT_EXPAND_LIMIT = 12

#: The nine columns every subgraph query in `kg/queries.py` returns. Named here
#: so a row that does not carry them is skipped loudly rather than drawn as a
#: node called `None` — the failure mode when a projection is edited and one
#: `AS` alias is dropped.
TRIPLE_COLUMNS = (
    "source_label",
    "source_key",
    "rel_type",
    "target_label",
    "target_key",
)


@dataclass(frozen=True)
class Node:
    """One node, keyed by label and its schema.cypher uniqueness property."""

    id: str
    label: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    #: The node the question was about — the item in "top markets for tea". The
    #: drawing centres on it; there is at most one.
    focus: bool = False


@dataclass(frozen=True)
class Edge:
    """One relationship, already oriented source -> target."""

    id: str
    source: str
    target: str
    type: str
    #: 0..1 within this subgraph, or None for a relationship that has no natural
    #: magnitude (`CLASSIFIED_AS`). Normalised rather than absolute because it
    #: drives stroke width, and a browser cannot scale a stroke by USD.
    weight: float | None = None
    #: What to print on the edge — the raw figure, formatted. `weight` is unit-less
    #: by then, so without this the drawing would show a thick line and no number.
    label: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Subgraph:
    """A drawable answer fragment, with the Cypher that produced it."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    focus_id: str | None = None
    #: The queries that ran, for the panel's own "show the Cypher" footer. The
    #: drawing gets the same provenance the figures already have.
    queries: list[str] = field(default_factory=list)
    #: True when a facet returned exactly its limit, i.e. there is probably more
    #: graph than is shown. Said out loud rather than left for the user to
    #: assume the picture is complete.
    truncated: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.nodes


def node_id(label: str, key: str) -> str:
    """`("Country", "USA") -> "Country:USA"`. The id the browser sends back."""
    return f"{label}:{key}"


def parse_node_id(value: str) -> tuple[str, str]:
    """Inverse of `node_id`, validating the label against `queries.NODE_KEYS`.

    Raises `ValueError` on anything else. Split on the *first* colon only: a
    `PolicyDocument` id may contain one, and `HSCode:6109` and `Country:USA`
    never do, so splitting on the last would corrupt exactly the label whose key
    is least predictable.
    """
    label, separator, key = value.partition(":")
    if not separator or not key.strip():
        raise ValueError(f"{value!r} is not a node id; expected 'Label:key'")
    if label not in q.NODE_KEYS:
        raise ValueError(
            f"{label!r} is not a node label in this graph; "
            f"expected one of {sorted(q.NODE_KEYS)}"
        )
    return label, key


async def build_answer_subgraph(
    kg: Any,
    *,
    item: str | None,
    year: int | None,
    sectors: tuple[str, ...] = (),
    limit: int = DEFAULT_FACET_LIMIT,
) -> Subgraph:
    """The subgraph behind an answer about `item`.

    Three facets — where it went, how it is classified, where it is grown — run
    concurrently. Without an `item` there is no subject to centre on and the
    result is empty: a graph of everything is not an answer to anything.

    `sectors` only suppresses the production facet for apparel, which is not
    modelled by district. Running it anyway would be a harmless empty result; not
    running it saves a round trip inside the SRS 3.4.1 budget.
    """
    if not item:
        return Subgraph()

    facets: list[tuple[str, q.Query]] = []
    if year is not None:
        facets.append(("exports", q.export_subgraph(item, year, limit=limit)))
    facets.append(("classification", q.classification_subgraph(item)))
    if "apparel" not in sectors:
        facets.append(("production", q.production_subgraph(item)))

    results = await asyncio.gather(
        *(kg.run(cypher, params) for _, (cypher, params) in facets),
        return_exceptions=True,
    )

    rows: list[dict[str, Any]] = []
    cyphers: list[str] = []
    truncated = False
    for (name, _), result in zip(facets, results, strict=True):
        if isinstance(result, BaseException):
            # One facet failing is a smaller picture, not a failed answer. Logged
            # at warning because a facet that always fails is a real bug, and the
            # only symptom the user sees is a graph that looks a bit sparse.
            log.warning("subgraph facet %s failed: %s", name, result)
            continue
        facet_rows, cypher = result
        rows.extend(facet_rows)
        cyphers.append(" ".join(cypher.split()))
        if name == "exports" and len(facet_rows) >= limit:
            truncated = True

    return _assemble(
        rows, focus_id=_focus_of(rows, item), queries=cyphers, truncated=truncated
    )


async def expand(
    kg: Any,
    *,
    label: str,
    key: str,
    limit: int = DEFAULT_EXPAND_LIMIT,
) -> Subgraph:
    """One hop out from a node the user clicked.

    Raises nothing on an unreachable graph — the caller gets an empty fragment
    and the canvas keeps what it already has. A click that quietly does nothing
    is a better failure than one that empties the drawing.
    """
    cypher, params = q.neighbours(label, key, limit=limit)
    try:
        rows, _ = await kg.run(cypher, params)
    except KnowledgeGraphUnavailableError as exc:
        log.warning("expand %s:%s failed: %s", label, key, exc)
        return Subgraph(queries=[" ".join(cypher.split())])

    return _assemble(
        rows,
        focus_id=node_id(label, key),
        queries=[" ".join(cypher.split())],
        truncated=len(rows) >= limit,
    )


def _assemble(
    rows: list[dict[str, Any]],
    *,
    focus_id: str | None,
    queries: list[str],
    truncated: bool,
) -> Subgraph:
    """Triples -> deduplicated nodes and edges, with weights normalised.

    Node identity is the semantic id, so the same country reached by two facets
    is one node with two edges rather than two overlapping circles. Later rows do
    not overwrite earlier ones: the first row to name a node wins, which keeps
    the assembled result independent of the order `asyncio.gather` happened to
    return the facets in.
    """
    nodes: dict[str, Node] = {}
    edges: dict[str, Edge] = {}
    weights: dict[str, float] = {}

    for row in rows:
        if any(row.get(column) is None for column in TRIPLE_COLUMNS):
            # A projection that lost an `AS` alias, or a node with no key. Drawing
            # it would produce a node labelled "None" joined to the real graph.
            log.debug("skipping malformed subgraph row: %s", row)
            continue

        source = _node_from(row, "source")
        target = _node_from(row, "target")
        nodes.setdefault(source.id, source)
        nodes.setdefault(target.id, target)

        rel_type = str(row["rel_type"])
        edge_id = f"{source.id}|{rel_type}|{target.id}"
        if edge_id in edges:
            continue
        raw_weight = _as_float(row.get("weight"))
        if raw_weight is not None:
            weights[edge_id] = raw_weight
        properties = dict(row.get("rel_props") or {})
        edges[edge_id] = Edge(
            id=edge_id,
            source=source.id,
            target=target.id,
            type=rel_type,
            label=_edge_label(rel_type, raw_weight, properties.get("year")),
            properties=properties,
        )

    if focus_id in nodes:
        focus = nodes[focus_id]
        nodes[focus_id] = Node(
            id=focus.id,
            label=focus.label,
            name=focus.name,
            properties=focus.properties,
            focus=True,
        )
    else:
        # The item was named in the question but has no edges in the graph. Better
        # to report no focus than to point the drawing at a node it does not hold.
        focus_id = None

    return Subgraph(
        nodes=list(nodes.values()),
        edges=_normalize_weights(list(edges.values()), weights),
        focus_id=focus_id,
        queries=queries,
        truncated=truncated,
    )


def _node_from(row: dict[str, Any], end: str) -> Node:
    label = str(row[f"{end}_label"])
    key = str(row[f"{end}_key"])
    # `target_name` is null for an HSCode with no description; the code itself is
    # a better label than an empty circle.
    name = row.get(f"{end}_name") or key
    return Node(id=node_id(label, key), label=label, name=str(name), properties={})


def _normalize_weights(edges: list[Edge], weights: dict[str, float]) -> list[Edge]:
    """Scale each weight to 0..1 *within its own relationship type*.

    Across types would be meaningless — a `PRODUCED_IN` share of 0.4 and an
    `EXPORTS_TO` value of 400 million are not on one scale, and normalising them
    together would draw every district edge as hairline. Types with no weight at
    all (`CLASSIFIED_AS`) keep `None` and get the stylesheet's default width.
    """
    maxima: dict[str, float] = {}
    for edge in edges:
        weight = weights.get(edge.id)
        if weight is not None and weight > maxima.get(edge.type, 0.0):
            maxima[edge.type] = weight

    scaled: list[Edge] = []
    for edge in edges:
        weight = weights.get(edge.id)
        maximum = maxima.get(edge.type, 0.0)
        scaled.append(
            Edge(
                id=edge.id,
                source=edge.source,
                target=edge.target,
                type=edge.type,
                weight=(weight / maximum) if weight is not None and maximum > 0 else None,
                label=edge.label,
                properties=edge.properties,
            )
        )
    return scaled


def _edge_label(rel_type: str, weight: float | None, year: Any = None) -> str | None:
    """The number to print on an edge, in the unit that relationship is in.

    A trade flow carries its year. The answer graph is scoped to one year and
    could state it once in the panel, but `expand()` deliberately is not — so an
    unlabelled "$104M" there sits next to a "$150M" for the same pair in the
    answer graph and reads as a contradiction rather than as two different
    years. Labelling every export edge is one rule instead of two, and the
    redundancy in the scoped case is the cheaper mistake.
    """
    if weight is None:
        return None
    if rel_type == "EXPORTS_TO":
        return f"{_usd(weight)} · {year}" if year is not None else _usd(weight)
    if rel_type == "PRODUCED_IN":
        return f"{weight * 100:.0f}%"
    return None


def _usd(value: float) -> str:
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.0f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def _focus_of(rows: list[dict[str, Any]], item: str) -> str | None:
    """The node id of the item the question was about, read off the rows.

    Both halves come from the data rather than from the caller. The label because
    `parse_intent` yields a bare name and `Commodity` and `ApparelCategory` share
    one namespace of names; the key because the queries match case-insensitively
    (`toLower(i.name) = toLower($item)`), so the graph may hold "Tea" where the
    question said "tea" — composing the id from the parsed string would then
    produce a focus that matches no node in the very subgraph it centres.
    """
    lowered = item.lower()
    for row in rows:
        for end in ("source", "target"):
            label = row.get(f"{end}_label")
            key = row.get(f"{end}_key")
            if label in ("Commodity", "ApparelCategory") and str(key).lower() == lowered:
                return node_id(str(label), str(key))
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
