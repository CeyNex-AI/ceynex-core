"""Assertions for the drawable subgraph (SRS 3.1.4, 3.1.6).

No database. The builder's whole job is turning triple rows into deduplicated
nodes and edges, so the rows are fixtures and the assertions are about what
comes out — the same posture as `test_queries.py`, one layer up.

The rows below are shaped exactly as `kg/queries.py` projects them and as
`record.data()` would deliver them: flat dicts of scalars and property maps,
with no Neo4j objects anywhere, because by that point there are none.
"""

import pytest

from ceynex.kg import subgraph as s
from ceynex.kg.client import KnowledgeGraphUnavailableError


def triple(
    source_label="Commodity",
    source_key="tea",
    rel_type="EXPORTS_TO",
    target_label="Country",
    target_key="USA",
    target_name="United States",
    weight=100.0,
    rel_props=None,
):
    return {
        "source_label": source_label,
        "source_key": source_key,
        "source_name": source_key,
        "rel_type": rel_type,
        "rel_props": rel_props if rel_props is not None else {"year": 2024},
        "target_label": target_label,
        "target_key": target_key,
        "target_name": target_name,
        "weight": weight,
    }


class FakeKG:
    """Returns canned rows per query, keyed by a substring of the Cypher."""

    def __init__(self, by_marker: dict[str, object]):
        self._by_marker = by_marker
        self.calls: list[str] = []

    async def run(self, cypher: str, params=None):
        self.calls.append(cypher)
        for marker, result in self._by_marker.items():
            if marker in cypher:
                if isinstance(result, Exception):
                    raise result
                return result, cypher
        return [], cypher


EXPORT_ROWS = [
    triple(target_key="USA", target_name="United States", weight=400.0),
    triple(target_key="IRQ", target_name="Iraq", weight=200.0),
    triple(target_key="TUR", target_name="Turkey", weight=100.0),
]

CLASSIFICATION_ROWS = [
    triple(
        rel_type="CLASSIFIED_AS",
        target_label="HSCode",
        target_key="0902",
        target_name="Tea",
        weight=None,
        rel_props={},
    )
]


def kg(exports=None, classification=None, production=None):
    return FakeKG(
        {
            "EXPORTS_TO": EXPORT_ROWS if exports is None else exports,
            "CLASSIFIED_AS": CLASSIFICATION_ROWS if classification is None else classification,
            "PRODUCED_IN": [] if production is None else production,
        }
    )


# --- node ids -------------------------------------------------------------


def test_node_id_is_the_label_and_its_uniqueness_key():
    """Not elementId: it changes on every `make kg-load`, so a node id the
    browser held from one answer could not be expanded against the next."""
    assert s.node_id("Country", "USA") == "Country:USA"
    assert s.parse_node_id("Country:USA") == ("Country", "USA")


def test_parse_node_id_splits_on_the_first_colon_only():
    """A PolicyDocument id may contain one; splitting on the last would corrupt
    exactly the label whose key is least predictable."""
    assert s.parse_node_id("PolicyDocument:edb:2024:strategy") == (
        "PolicyDocument",
        "edb:2024:strategy",
    )


@pytest.mark.parametrize("bad", ["Country", "Country:", ":USA", "", "Bogus:x", "User:admin"])
def test_parse_node_id_rejects_anything_else(bad):
    with pytest.raises(ValueError):
        s.parse_node_id(bad)


# --- assembly -------------------------------------------------------------


@pytest.mark.asyncio
async def test_builds_nodes_and_edges_from_triples():
    built = await s.build_answer_subgraph(kg(), item="tea", year=2024)

    assert built.focus_id == "Commodity:tea"
    ids = {node.id for node in built.nodes}
    assert "Commodity:tea" in ids
    assert "Country:USA" in ids
    assert "HSCode:0902" in ids
    assert len(built.edges) == 4  # three destinations plus the HS code


@pytest.mark.asyncio
async def test_a_node_reached_by_two_facets_is_one_node():
    """Otherwise the same country is drawn twice, overlapping, with the edges
    split between the copies."""
    built = await s.build_answer_subgraph(kg(), item="tea", year=2024)
    tea = [node for node in built.nodes if node.id == "Commodity:tea"]
    assert len(tea) == 1


@pytest.mark.asyncio
async def test_the_focus_node_is_marked_and_is_the_only_one():
    built = await s.build_answer_subgraph(kg(), item="tea", year=2024)
    focused = [node for node in built.nodes if node.focus]
    assert [node.id for node in focused] == ["Commodity:tea"]


@pytest.mark.asyncio
async def test_focus_uses_the_graphs_spelling_not_the_questions():
    """The queries match with `toLower`, so the graph may hold "Tea" where the
    question said "tea". Composing the focus id from the parsed string would
    then point the drawing at a node absent from its own subgraph."""
    rows = [triple(source_key="Tea", target_key="USA", weight=1.0)]
    built = await s.build_answer_subgraph(
        FakeKG({"EXPORTS_TO": rows}), item="tea", year=2024
    )
    assert built.focus_id == "Commodity:Tea"
    assert any(node.focus for node in built.nodes)


@pytest.mark.asyncio
async def test_focus_is_none_when_the_item_has_no_edges():
    """Better no focus than a focus pointing at a node the graph does not hold."""
    built = await s.build_answer_subgraph(FakeKG({}), item="tea", year=2024)
    assert built.focus_id is None
    assert built.is_empty


@pytest.mark.asyncio
async def test_an_apparel_item_identifies_as_an_apparel_category():
    """Commodity and ApparelCategory share one namespace of names, so the label
    is read off the rows rather than guessed."""
    rows = [triple(source_label="ApparelCategory", source_key="apparel_knit", weight=5.0)]
    built = await s.build_answer_subgraph(
        FakeKG({"EXPORTS_TO": rows}), item="apparel_knit", year=2024
    )
    assert built.focus_id == "ApparelCategory:apparel_knit"


# --- weights --------------------------------------------------------------


@pytest.mark.asyncio
async def test_weights_are_normalised_within_the_subgraph():
    built = await s.build_answer_subgraph(kg(), item="tea", year=2024)
    by_target = {edge.target: edge.weight for edge in built.edges}
    assert by_target["Country:USA"] == 1.0
    assert by_target["Country:IRQ"] == 0.5
    assert by_target["Country:TUR"] == 0.25


@pytest.mark.asyncio
async def test_weights_are_normalised_per_relationship_type():
    """A PRODUCED_IN share of 0.4 and an EXPORTS_TO value of 400 million are not
    on one scale; normalising them together draws every district edge hairline."""
    production = [
        triple(
            rel_type="PRODUCED_IN",
            target_label="District",
            target_key="Kandy",
            target_name="Kandy",
            weight=0.4,
            rel_props={"share": 0.4},
        )
    ]
    built = await s.build_answer_subgraph(
        kg(production=production), item="tea", year=2024
    )
    # Each type's own largest edge is 1.0. The district edge's raw 0.4 would be
    # 0.000000001 against the export scale, which is a line nobody can see.
    for rel_type in ("PRODUCED_IN", "EXPORTS_TO"):
        weights = [e.weight for e in built.edges if e.type == rel_type]
        assert max(weights) == 1.0, rel_type


@pytest.mark.asyncio
async def test_a_relationship_with_no_magnitude_has_no_weight():
    """CLASSIFIED_AS gets the stylesheet's default width, not a zero-width line."""
    built = await s.build_answer_subgraph(kg(), item="tea", year=2024)
    classified = next(edge for edge in built.edges if edge.type == "CLASSIFIED_AS")
    assert classified.weight is None
    assert classified.label is None


@pytest.mark.asyncio
async def test_export_edges_carry_a_formatted_figure_and_its_year():
    """`weight` is unit-less by the time it arrives, so without a label the
    drawing shows a thick line and no number.

    The year is on the label because `expand()` is not year-scoped: found live
    against the dev graph, a tea->Iraq edge drawn at $104M beside an answer
    graph showing $150M for the same pair, with nothing on either to say they
    were different years.
    """
    rows = [triple(target_key="USA", weight=412_000_000.0, rel_props={"year": 2024})]
    built = await s.build_answer_subgraph(
        FakeKG({"EXPORTS_TO": rows}), item="tea", year=2024
    )
    assert built.edges[0].label == "$412M · 2024"


@pytest.mark.asyncio
async def test_an_export_edge_without_a_year_still_shows_its_figure():
    rows = [triple(target_key="USA", weight=412_000_000.0, rel_props={})]
    built = await s.build_answer_subgraph(
        FakeKG({"EXPORTS_TO": rows}), item="tea", year=2024
    )
    assert built.edges[0].label == "$412M"


# --- degradation ----------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failing_facet_leaves_the_others_standing():
    """SAD §4.1's partial-result guarantee, applied to a drawing."""
    failing = FakeKG(
        {
            "EXPORTS_TO": KnowledgeGraphUnavailableError("neo4j down"),
            "CLASSIFIED_AS": CLASSIFICATION_ROWS,
        }
    )
    built = await s.build_answer_subgraph(failing, item="tea", year=2024)
    assert {node.id for node in built.nodes} == {"Commodity:tea", "HSCode:0902"}


@pytest.mark.asyncio
async def test_no_item_means_no_subgraph():
    """A graph of everything is not an answer to anything."""
    built = await s.build_answer_subgraph(FakeKG({}), item=None, year=2024)
    assert built.is_empty
    assert built.queries == []


@pytest.mark.asyncio
async def test_without_a_year_the_export_facet_does_not_run():
    """An unfiltered EXPORTS_TO match returns every year at once, which draws
    one country several times over with no way to tell the edges apart."""
    fake = kg()
    await s.build_answer_subgraph(fake, item="tea", year=None)
    assert not any("EXPORTS_TO" in call for call in fake.calls)


@pytest.mark.asyncio
async def test_apparel_skips_the_district_facet():
    """Apparel is not modelled by district. Harmless to run, one round trip
    saved inside the SRS 3.4.1 budget not to."""
    fake = kg()
    await s.build_answer_subgraph(fake, item="apparel_knit", year=2024, sectors=("apparel",))
    assert not any("PRODUCED_IN" in call for call in fake.calls)


@pytest.mark.asyncio
async def test_a_malformed_row_is_skipped_not_drawn():
    """A projection that lost an `AS` alias would otherwise put a node called
    "None" in the middle of the real graph."""
    rows = [triple(), {**triple(target_key="IRQ"), "target_label": None}]
    built = await s.build_answer_subgraph(
        FakeKG({"EXPORTS_TO": rows}), item="tea", year=2024
    )
    assert {node.id for node in built.nodes} == {"Commodity:tea", "Country:USA"}


@pytest.mark.asyncio
async def test_truncation_is_reported():
    """A partial view that does not say so reads as a complete one."""
    rows = [triple(target_key=f"C{i}", weight=float(i + 1)) for i in range(3)]
    built = await s.build_answer_subgraph(
        FakeKG({"EXPORTS_TO": rows}), item="tea", year=2024, limit=3
    )
    assert built.truncated is True


@pytest.mark.asyncio
async def test_the_cypher_travels_with_the_drawing():
    """The picture gets the same provenance the figures already have."""
    built = await s.build_answer_subgraph(kg(), item="tea", year=2024)
    assert built.queries
    assert any("EXPORTS_TO" in text for text in built.queries)


# --- expand ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_expand_returns_a_neighbourhood():
    rows = [triple(source_label="Commodity", source_key="cinnamon", target_key="USA")]
    fragment = await s.expand(FakeKG({"startNode": rows}), label="Country", key="USA")
    assert {node.id for node in fragment.nodes} == {"Commodity:cinnamon", "Country:USA"}


@pytest.mark.asyncio
async def test_expand_keeps_one_edge_per_pair_and_labels_its_year():
    """An EXPORTS_TO pair has one edge per year and expand is not year-scoped,
    so several arrive for the same pair and only one is drawn. Which one must
    not depend on the planner: `neighbours` orders newest-first, and the label
    carries the year so it cannot silently disagree with the answer graph."""
    rows = [
        triple(source_key="tea", target_key="IRQ", weight=150.0, rel_props={"year": 2024}),
        triple(source_key="tea", target_key="IRQ", weight=104.0, rel_props={"year": 2022}),
    ]
    fragment = await s.expand(FakeKG({"startNode": rows}), label="Country", key="IRQ")
    assert len(fragment.edges) == 1
    assert fragment.edges[0].label == "$150 · 2024"


@pytest.mark.asyncio
async def test_expand_degrades_to_an_empty_fragment():
    """A click that quietly does nothing beats one that empties the canvas."""
    fake = FakeKG({"startNode": KnowledgeGraphUnavailableError("neo4j down")})
    fragment = await s.expand(fake, label="Country", key="USA")
    assert fragment.is_empty
    assert fragment.edges == []
