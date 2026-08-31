"""SRS 3.1.8 — one real-world product category is one `item` key.

Every label asserted here was read out of the production `fact_trade` on
2026-08-30, not invented: `SELECT DISTINCT item FROM fact_trade WHERE
source_id='EDB'` returned exactly these 22 strings, including the en-dashes and
the two typos. Nothing in the suite guarded item normalization before this file,
which is why the split went unnoticed.
"""

import pytest

from ceynex.data.crosswalk import (
    ItemVocabularyError,
    canonical_item,
    item_granularity,
    known_items,
)

# The three families that actually collide in production.
_APPAREL = ["APPAREL", "APPREL"]
_APPAREL_TEXTILES = [
    "APPAREL & TEXTILES (Made - Up Textile Article, Apparel, Woven Fabrics & Other Textile Articles)",
    "APPAREL AND TEXTILES (Made – Up Textile Articles, Apparel, Woven Fabrics, Other Textile Articles)",
]
_MADE_UP = [
    "MADE - UP TEXTILE ARTICLES (Blankets, Rugs, Linen & Curtains etc.)",
    "MADE-UP TEXTILE ARTICLES (Blankets, Rugs, Linen, Curtains etc)",
    "MADE – UP TEXTILE ARTICLES",
]

# Every distinct EDB label in production, verbatim.
_ALL_PRODUCTION_LABELS = [
    *_APPAREL,
    *_APPAREL_TEXTILES,
    *_MADE_UP,
    "ACTIVEWEAR/ SPORTSWERA",
    "BABIES' GARMENTS",
    "GLOVES, MITTS & MITTENS OF TEXTILE",
    "HOSIERY",
    "KNITTED FABRICS",
    "MADE - UP CLOTHING ACCESSORIES (Handkerchief, Shawls, Scarves, Ties etc)",
    "MEN'S & WOMEN'S UNDER GARMENTS",
    "MEN'S OUTERWEAR",
    "T-SHIRTS",
    "TEXTILE FLOOR COVERING (Carpets, Mats, Floor Covering etc)",
    "TEXTILES ( Knitted Fabrics, Woven Fabrics, Yarn, Made - Up Textile Articles, Textile Floor Covering etc.)",
    "WARM CLOTHS (Jerseys, Pullovers etc.)",
    "WOMEN'S OUTERWEAR",
    "WOVEN FABRICS",
    "YARN",
]


@pytest.mark.parametrize("label", _ALL_PRODUCTION_LABELS)
def test_every_production_label_resolves(label: str) -> None:
    """A label EDB has actually published must never raise in ingest."""
    assert canonical_item(label)


def test_the_apparel_typo_merges_rather_than_splitting_the_series() -> None:
    """`APPAREL` held 2014-2018 and `APPREL` 2019-2024 — one series, two names.

    `item` is part of `fact_trade_upsert_key`, so keeping them distinct is what
    fitted the registered apparel model on 5 rows.
    """
    assert len({canonical_item(x) for x in _APPAREL}) == 1
    assert canonical_item("APPREL") == "apparel_edb"


def test_ampersand_and_en_dash_variants_are_one_item() -> None:
    assert len({canonical_item(x) for x in _APPAREL_TEXTILES}) == 1
    assert len({canonical_item(x) for x in _MADE_UP}) == 1


def test_the_parenthetical_gloss_is_not_part_of_the_identity() -> None:
    """Two of the three MADE-UP variants gloss their contents; one does not."""
    assert canonical_item("MADE – UP TEXTILE ARTICLES") == canonical_item(
        "MADE-UP TEXTILE ARTICLES (Blankets, Rugs, Linen, Curtains etc)"
    )


def test_the_22_production_labels_collapse_to_18_items() -> None:
    resolved = {canonical_item(x) for x in _ALL_PRODUCTION_LABELS}
    assert len(_ALL_PRODUCTION_LABELS) == 22
    assert len(resolved) == 18


def test_an_unknown_label_raises_rather_than_passing_through() -> None:
    """The whole point. Passing an unknown label through is what created the split."""
    with pytest.raises(ItemVocabularyError, match="item_vocabulary.csv"):
        canonical_item("SMART TEXTILES (a category EDB has not published)")


def test_the_edb_rollups_are_not_marked_as_components() -> None:
    """`kg/loaders/apparel.py` excludes these precisely because they self-sum."""
    assert item_granularity(_APPAREL_TEXTILES[0]) == "total"
    assert item_granularity("APPAREL") == "group"
    assert item_granularity("T-SHIRTS") == "component"


def test_the_graph_item_key_the_loaders_pin_to_still_exists() -> None:
    """`kg/loaders/apparel._EDB_GRAPH_ITEM` and the apparel agent both pin this."""
    assert "apparel_edb" in known_items()
