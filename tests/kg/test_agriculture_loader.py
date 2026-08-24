from __future__ import annotations

from typing import Any

from ceynex.kg.loaders import agriculture


class _RecordingKG:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def write(self, cypher: str, params: dict[str, Any]) -> int:
        self.calls.append((cypher, params))
        return 0


async def test_agriculture_loader_merges_nodes_edges_and_supported_coverage() -> None:
    kg = _RecordingKG()

    counts = await agriculture.load(kg)  # type: ignore[arg-type]

    assert counts == {"commodities": 4, "classifications": 5, "districts": 6, "coverage_edges": 12}
    assert len(kg.calls) == 3

    commodity_rows = kg.calls[0][1]["rows"]
    assert {row["name"] for row in commodity_rows} == {"tea", "cinnamon", "rubber", "coconut"}
    assert next(row for row in commodity_rows if row["name"] == "coconut")["hs_codes"] == ["0801", "1513"]

    district_rows = kg.calls[1][1]["rows"]
    assert {row["district"] for row in district_rows} == {
        "Nuwara Eliya", "Badulla", "Kandy", "Matara", "Galle", "Ratnapura",
    }
    assert kg.calls[1][1]["share_note"] == agriculture.DISTRICT_SHARE_NOTE

    coverage_rows = kg.calls[2][1]["rows"]
    assert {row["hs_code"] for row in coverage_rows} == {"0902", "0906", "4001", "1513"}
    assert all(row["verified"] == "unverified" for row in coverage_rows)


async def test_agriculture_loader_is_idempotent_by_construction() -> None:
    kg = _RecordingKG()

    first = await agriculture.load(kg)  # type: ignore[arg-type]
    first_calls = list(kg.calls)
    second = await agriculture.load(kg)  # type: ignore[arg-type]
    second_calls = kg.calls[len(first_calls) :]

    assert first == second
    assert [(cypher, params) for cypher, params in first_calls] == second_calls
    for cypher, _params in kg.calls:
        assert "MERGE" in cypher
        assert "CREATE (" not in cypher
