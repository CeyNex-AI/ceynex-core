"""Assertions for the :PolicyDocument loader (deviation D10).

No docker: a `_RecordingKG` captures the statements, same approach as
`test_agriculture_loader.py`. What is worth asserting is not that the loader
writes — it is *what* it refuses to write. A loader that quietly creates a bare
`Country` node breaks the crosswalk's `country_m49` uniqueness constraint on the
next real load, in a different module, hours later.
"""

from __future__ import annotations

from typing import Any

import pytest

from ceynex.kg.loaders import policy_documents


class _RecordingKG:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def write(self, cypher: str, params: dict[str, Any]) -> int:
        self.calls.append((cypher, params))
        return 0


@pytest.fixture
def loaded(monkeypatch):
    # The real one reaches for Qdrant; chunk counts are not what these tests are about.
    async def no_counts(collection=None):  # noqa: ANN001, ARG001
        return {}

    monkeypatch.setattr(policy_documents, "chunk_counts", no_counts)
    return _RecordingKG()


# --- the manifest --------------------------------------------------------


def test_the_manifest_parses_with_its_provenance_header_skipped():
    rows = policy_documents.document_rows()

    assert rows, "the committed manifest must not read as empty"
    assert all(row["doc_id"] for row in rows)
    assert not any(row["doc_id"].startswith("#") for row in rows), "header leaked into the rows"


def test_one_eu_document_serves_three_member_states_as_a_single_row():
    """Three rows would triple-count the same document in every EU retrieval."""
    eu = next(r for r in policy_documents.document_rows() if r["doc_id"] == "EU-DGTRADE-POLICY")

    assert set(eu["iso3"]) == {"DEU", "ITA", "FRA"}


def test_a_non_english_document_is_recorded_but_not_marked_indexed():
    """The row is the evidence the source was found and consciously skipped.

    Deleting it would leave no trace of the decision; marking it `indexed` would
    let an agent cite text the English-only embedding model cannot retrieve.
    """
    awg = next(r for r in policy_documents.document_rows() if r["doc_id"] == "DEU-AWG-2013")

    assert awg["language"] == "de"
    assert awg["indexed"] is False


def test_agreement_names_match_the_trade_agreement_csv_exactly():
    """A typo here produces an orphan node instead of a DESCRIBES edge, silently."""
    from ceynex.kg.loaders.trade_agreements import agreement_rows

    known = {row["name"] for row in agreement_rows()}
    named = {name for row in policy_documents.document_rows() for name in row["agreements"]}

    assert named <= known, f"unknown agreement names in the manifest: {named - known}"


# --- what the loader writes ----------------------------------------------


async def test_the_loader_only_ever_merges(loaded):
    counts = await policy_documents.load(loaded)  # type: ignore[arg-type]

    assert counts["documents"] == len(policy_documents.document_rows())
    for cypher, _ in loaded.calls:
        assert "CREATE " not in cypher.upper(), "three members load into one graph; MERGE only"
        assert "DELETE" not in cypher.upper()


async def test_issued_by_matches_a_country_and_never_merges_one(loaded):
    """The check `docs/CONTRACT_PROPOSAL_POLICY_DOCUMENT.md` asks M1 and M3 for.

    `dim_country` and the trade-flow loaders own country creation. A Country
    merged here would have no `m49` and would collide with the `country_m49`
    uniqueness constraint the next time a real load ran.
    """
    await policy_documents.load(loaded)  # type: ignore[arg-type]

    issued_by = next(c for c, _ in loaded.calls if "ISSUED_BY" in c)
    assert "MATCH (c:Country" in issued_by
    assert "MERGE (c:Country" not in issued_by


def test_every_write_is_parameterized():
    """The KG client's f-string tripwire, asserted rather than logged."""
    for statement in (
        policy_documents.MERGE_DOCUMENTS,
        policy_documents.MERGE_ISSUED_BY,
        policy_documents.MERGE_APPLIES_TO,
        policy_documents.MERGE_DESCRIBES,
        policy_documents.SET_CHUNK_COUNTS,
    ):
        assert "$" in statement, f"unparameterized statement: {statement[:60]}"


async def test_running_twice_writes_the_same_statements(loaded):
    """Idempotence, at the level this test can see it.

    The real guarantee is Neo4j's — `MERGE` on a unique key — but a loader that
    accumulated state between runs would show up here as differing parameters.
    """
    await policy_documents.load(loaded)  # type: ignore[arg-type]
    first = list(loaded.calls)
    loaded.calls.clear()
    await policy_documents.load(loaded)  # type: ignore[arg-type]

    assert [c for c, _ in first] == [c for c, _ in loaded.calls]
    assert [p for _, p in first] == [p for _, p in loaded.calls]
