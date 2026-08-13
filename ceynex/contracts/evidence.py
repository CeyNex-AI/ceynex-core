"""Implements SRS 3.1.4 — supporting evidence attached to every answer.

FROZEN CONTRACT. Changes require 3-way approval (M1, M2, M3).
"""

from typing import TypedDict

from typing_extensions import NotRequired

SourceId = str
"""One of: UN_COMTRADE | WITS | FAOSTAT | CBSL | JAAF | EDB | KG | MODEL.

Kept as a plain str rather than a Literal so a member can add a source without
a contract change; the allowed values are asserted in tests, not the type system.
"""


class Evidence(TypedDict):
    """One traceable justification for a claim in an answer.

    Every figure a user sees must be reachable from at least one Evidence entry.
    `detail` is the machine-level provenance — the literal Cypher query for KG
    evidence, `table + filter` for dataset evidence, or the registry model id for
    MODEL evidence. `claim` is the human-readable half: one sentence a
    policymaker can read without knowing what Cypher is.
    """

    source_id: SourceId
    claim: str
    detail: str
    period: NotRequired[str]
    url: NotRequired[str]
