"""Implements SRS 3.1.4 — how the confidence score attached to every answer is computed.

"How is the confidence score calculated?" is the most predictable question this
project will be asked. The answer is the formula below, and nothing else in the
codebase is allowed to invent its own.

Formula
-------
    weighted  = sum(w_i * c_i) / sum(w_i)
                  w_i = the router's relevance weight for agent i (defaults to 1.0)
                  c_i = that agent's self-reported AgentOutput.confidence

    staleness = min(0.20, 0.02 * months_since_latest_observation)
    dq        = min(0.25, 0.05 * n_material_flags + 0.10 * n_severe_flags)
    coverage  = 0.15 if any routed agent failed or returned degraded, else 0.0

    final     = clamp(0.05, 0.95, weighted - staleness - dq - coverage)

Why each term
-------------
weighted   A cross-sector answer should not inherit the confidence of the agent
           that contributed least to it, so agents are weighted by the relevance
           the router assigned them rather than averaged flat.
staleness  Trade data is published with a lag. A figure whose latest observation
           is two years old deserves less confidence than the same figure
           computed last month. 0.02/month reaches the 0.20 cap at ten months,
           which is roughly when an annual series stops being current.
dq         A `dq_flag` means two sources disagree about a number the answer may
           rest on (SRS 3.1.8). Severe disagreements are penalised twice as hard
           as material ones; minor (<5%) ones are ignored as ordinary noise.
coverage   SRS 3.4.3 requires saying which part of a question could not be
           answered. A partial answer is still useful, but it is not as
           trustworthy as a complete one, and the score should show that.
clamp      Never 0.0 (the system did answer) and never 1.0 (nothing forecast
           from historical trade data is certain). Both bounds are a deliberate
           refusal to overclaim.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ceynex.contracts.state import AgentName, AgentOutput

STALENESS_PER_MONTH = 0.02
STALENESS_CAP = 0.20
DQ_MATERIAL_PENALTY = 0.05
DQ_SEVERE_PENALTY = 0.10
DQ_CAP = 0.25
COVERAGE_PENALTY = 0.15
FLOOR = 0.05
CEILING = 0.95


def clamp(value: float, low: float = FLOOR, high: float = CEILING) -> float:
    return max(low, min(high, value))


def weighted_agent_confidence(
    outputs: Mapping[AgentName, AgentOutput],
    relevance: Mapping[AgentName, float] | None = None,
) -> float:
    """Relevance-weighted mean of the agents' self-reported confidences.

    Agents that failed still participate with their (zero) confidence — dropping
    them would let a two-agent answer where one agent died score as highly as
    one where both succeeded.
    """
    if not outputs:
        return 0.0
    relevance = relevance or {}
    total_weight = 0.0
    total = 0.0
    for name, output in outputs.items():
        weight = float(relevance.get(name, 1.0))
        if weight <= 0.0:
            continue
        total += weight * float(output["confidence"])
        total_weight += weight
    return total / total_weight if total_weight else 0.0


def staleness_penalty(months_since_latest_observation: float | None) -> float:
    if months_since_latest_observation is None or months_since_latest_observation <= 0:
        return 0.0
    return min(STALENESS_CAP, STALENESS_PER_MONTH * months_since_latest_observation)


def dq_penalty(severities: Iterable[str]) -> float:
    """`severities` is the severity column of the dq_flag rows touching this answer."""
    material = sum(1 for s in severities if s == "material")
    severe = sum(1 for s in severities if s == "severe")
    return min(DQ_CAP, DQ_MATERIAL_PENALTY * material + DQ_SEVERE_PENALTY * severe)


def coverage_penalty(
    route: Iterable[AgentName],
    outputs: Mapping[AgentName, AgentOutput],
) -> float:
    """Penalise when a routed agent failed, degraded, or never reported at all."""
    for agent in route:
        output = outputs.get(agent)
        if output is None or output.get("error") or output["degraded"]:
            return COVERAGE_PENALTY
    return 0.0


def aggregate_confidence(
    outputs: Mapping[AgentName, AgentOutput],
    route: Iterable[AgentName] = (),
    relevance: Mapping[AgentName, float] | None = None,
    months_since_latest_observation: float | None = None,
    dq_severities: Iterable[str] = (),
) -> float:
    """The single entrypoint. See the module docstring for the formula and its rationale."""
    return aggregate_confidence_breakdown(
        outputs, route, relevance, months_since_latest_observation, dq_severities
    ).final


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Every term in the formula, kept rather than discarded.

    The four penalty functions above have always been separate and named; only
    the final number survived, which meant "how is this calculated?" — the most
    predictable question this project will be asked (SRS 3.1.4) — could be
    answered from the docstring but never from the answer in front of the
    reader. This changes nothing about the arithmetic; it stops throwing the
    working away.
    """

    weighted: float
    staleness: float
    dq: float
    coverage: float
    final: float

    def as_dict(self) -> dict[str, float]:
        return {
            "weighted": round(self.weighted, 4),
            "staleness": round(self.staleness, 4),
            "dq": round(self.dq, 4),
            "coverage": round(self.coverage, 4),
            "final": round(self.final, 4),
        }


def aggregate_confidence_breakdown(
    outputs: Mapping[AgentName, AgentOutput],
    route: Iterable[AgentName] = (),
    relevance: Mapping[AgentName, float] | None = None,
    months_since_latest_observation: float | None = None,
    dq_severities: Iterable[str] = (),
) -> ConfidenceBreakdown:
    """The same computation as `aggregate_confidence`, showing its working.

    `aggregate_confidence` delegates here, so the two cannot disagree — which
    matters more than the small duplication it avoids: a breakdown that did not
    add up to the score beside it would be worse than no breakdown at all.
    """
    route = list(route) or list(outputs.keys())
    weighted = weighted_agent_confidence(outputs, relevance)
    staleness = staleness_penalty(months_since_latest_observation)
    dq = dq_penalty(dq_severities)
    coverage = coverage_penalty(route, outputs)
    return ConfidenceBreakdown(
        weighted=weighted,
        staleness=staleness,
        dq=dq,
        coverage=coverage,
        final=clamp(weighted - staleness - dq - coverage),
    )


def confidence_band(score: float) -> str:
    """Qualitative label shown beside the percentage in the UI (SRS 3.2.2)."""
    if score >= 0.75:
        return "High"
    if score >= 0.50:
        return "Moderate"
    if score >= 0.30:
        return "Low"
    return "Very low"
