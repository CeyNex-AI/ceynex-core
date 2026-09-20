"""The API's request/response contract does not drift unnoticed.

ceynex-web integrates against this backend's OpenAPI surface. Nothing verified
that a change to a route's request or response shape was intentional, or told
the frontend it happened. This test snapshots a stable projection of the spec
(endpoints, body shapes, component schemas -- see `_api_contract.project`) and
fails when the live app diverges from `openapi_snapshot.json`.

When a change IS intended, regenerate the snapshot deliberately:

    python -m tests.contracts.refresh_openapi_snapshot

and commit it in the same PR as the API change, so the diff is the record of
what the frontend now has to handle.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._api_contract import current_projection

SNAPSHOT = Path(__file__).parent / "openapi_snapshot.json"


def _diff_keys(old: dict, new: dict) -> tuple[list[str], list[str]]:
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    return added, removed


def test_openapi_contract_matches_the_committed_snapshot() -> None:
    committed = json.loads(SNAPSHOT.read_text())
    live = current_projection()

    if live == committed:
        return

    lines: list[str] = ["The API contract drifted from openapi_snapshot.json."]

    added, removed = _diff_keys(committed["paths"], live["paths"])
    for p in added:
        lines.append(f"  + new endpoint: {p}")
    for p in removed:
        lines.append(f"  - removed endpoint: {p}")
    for p in sorted(set(committed["paths"]) & set(live["paths"])):
        if committed["paths"][p] != live["paths"][p]:
            lines.append(f"  ~ changed request/response shape: {p}")

    s_added, s_removed = _diff_keys(committed["schemas"], live["schemas"])
    for s in s_added:
        lines.append(f"  + new schema: {s}")
    for s in s_removed:
        lines.append(f"  - removed schema: {s}")
    for s in sorted(set(committed["schemas"]) & set(live["schemas"])):
        if committed["schemas"][s] != live["schemas"][s]:
            lines.append(f"  ~ changed schema: {s}")

    lines.append(
        "\nIf this change is intended, regenerate the snapshot with"
        "\n  python -m tests.contracts.refresh_openapi_snapshot"
        "\nand commit it alongside the API change."
    )
    raise AssertionError("\n".join(lines))
