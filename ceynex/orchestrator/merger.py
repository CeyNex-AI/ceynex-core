"""Implements SRS 3.1.2 and 3.1.4 — one coherent answer, not a list of agent replies.

The SRS forbids presenting agent outputs "as separate, disconnected responses",
and that is the line between an orchestrator and a concatenator. If the answer
reads "The agriculture agent says X. The apparel agent says Y." then this module
has failed, whatever else it does.

So the merge is structured around what the agents *found*, not around who they
are:

- **Conflicts are surfaced, never averaged.** Two agents disagreeing on the
  direction of an effect is information. Splitting the difference destroys it and
  produces a number neither agent would defend.
- **Partial results still answer.** One agent failing means the answer covers
  what succeeded and states plainly what it could not cover (SAD §4.1). It does
  not mean an error page.
- **Evidence is deduplicated but attribution is preserved.** The same Cypher run
  by two agents is one piece of evidence; the same claim from two different
  sources is two, and worth more than either alone.

The LLM writes the prose. When it is unavailable the deterministic merge still
produces a readable answer from the figures (SRS 3.4.3) — that path is exercised
on every keyless run, not just in a test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ceynex.contracts import AgentName, AgentOutput, AgentState, Evidence
from ceynex.orchestrator.confidence import aggregate_confidence, confidence_band

log = logging.getLogger(__name__)

# Two agents reporting the same figure within this relative distance are
# agreeing. Wider than measurement noise, narrower than a real disagreement.
AGREEMENT_TOLERANCE = 0.10

# Figures whose *direction* is the claim. A sign disagreement on one of these is
# a substantive conflict, not a rounding difference.
DIRECTIONAL_SUFFIXES = ("_impact_pct", "_impact_usd", "cagr", "_change", "_growth")

MERGE_SYSTEM = """You write the final answer for a Sri Lankan export intelligence platform.

You are given findings from several specialist analyses of one question. Write ONE answer.

Absolute rules:
1. Never write "the X agent found" or otherwise name the internal analyses. The reader
   asked a question, not for a committee's minutes. Organise by finding, not by source.
2. Every number you state must appear in the findings given to you. Never estimate,
   never extrapolate, never add a figure from your own knowledge.
3. If the findings disagree, say so explicitly and give both figures. Do not average
   them, do not pick one silently.
4. If something could not be answered, say which part and why, in one clause.
5. Four to eight sentences. Plain English for a policymaker who is not an economist.
6. No preamble, no bullet lists, no headings. Start with the answer.
7. You describe data. You do not give financial, legal or investment advice."""


@dataclass
class Conflict:
    """Two agents reporting materially different values for the same quantity."""

    figure: str
    values: dict[AgentName, float]
    kind: str  # "direction" | "magnitude"

    def describe(self) -> str:
        pairs = ", ".join(
            f"{agent.replace('_', ' ')} {value:,.4g}" for agent, value in self.values.items()
        )
        if self.kind == "direction":
            return (
                f"The analyses disagree on the direction of {_humanize(self.figure)}: {pairs}. "
                "Both figures are shown rather than reconciled."
            )
        return (
            f"The analyses give different magnitudes for {_humanize(self.figure)}: {pairs}. "
            "Both are shown rather than averaged."
        )


@dataclass
class MergeResult:
    answer: str
    confidence: float
    band: str
    evidence: list[Evidence]
    agents_used: list[AgentName]
    forecast: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    conflicts: list[Conflict] = field(default_factory=list)
    unanswered: list[str] = field(default_factory=list)

    def as_state_patch(self) -> dict[str, Any]:
        return {
            "final_answer": self.answer,
            "final_confidence": self.confidence,
            "merged_evidence": self.evidence,
        }


async def merge(state: AgentState, llm, *, dq_severities=()) -> MergeResult:  # noqa: ANN001
    """Combine every agent output in `state` into one answer."""
    outputs = state.get("agent_outputs", {})
    route = list(state.get("route", []) or outputs.keys())
    relevance = state.get("relevance", {})

    succeeded = {name: out for name, out in outputs.items() if not out.get("error")}
    failed = {name: out for name, out in outputs.items() if out.get("error")}
    never_reported = [name for name in route if name not in outputs]

    unanswered = _describe_gaps(failed, never_reported) + _out_of_scope_gaps(state)
    conflicts = detect_conflicts(succeeded)
    evidence = dedupe_evidence(succeeded)
    forecast = _first_forecast(succeeded)

    confidence = aggregate_confidence(
        outputs,
        route=route,
        relevance=relevance,
        months_since_latest_observation=_staleness_months(state),
        dq_severities=dq_severities,
    )

    if not succeeded:
        answer = _nothing_succeeded(unanswered)
        return MergeResult(
            answer=answer,
            confidence=confidence,
            band=confidence_band(confidence),
            evidence=evidence,
            agents_used=[],
            degraded=True,
            conflicts=conflicts,
            unanswered=unanswered,
        )

    deterministic = compose_deterministic(state["query"], succeeded, conflicts, unanswered)

    prose = await llm.generate(
        "merge",
        MERGE_SYSTEM,
        _merge_prompt(state["query"], succeeded, conflicts, unanswered),
    )
    degraded = not prose

    answer = prose or deterministic
    if prose:
        # The LLM is told to surface conflicts, but the requirement is not
        # optional, so the deterministic sentence is appended if it did not.
        answer = _ensure_conflicts_stated(answer, conflicts)
        answer = _ensure_gaps_stated(answer, unanswered)

    return MergeResult(
        answer=answer.strip(),
        confidence=confidence,
        band=confidence_band(confidence),
        evidence=evidence,
        agents_used=sorted(succeeded),
        forecast=forecast,
        degraded=degraded or any(o.get("degraded") for o in succeeded.values()),
        conflicts=conflicts,
        unanswered=unanswered,
    )


# --- conflicts -----------------------------------------------------------


def detect_conflicts(outputs: dict[AgentName, AgentOutput]) -> list[Conflict]:
    """Figures reported by more than one agent that materially disagree."""
    by_figure: dict[str, dict[AgentName, float]] = {}
    for agent, output in outputs.items():
        for figure, value in (output.get("figures") or {}).items():
            if isinstance(value, int | float):
                by_figure.setdefault(figure, {})[agent] = float(value)

    conflicts: list[Conflict] = []
    for figure, values in by_figure.items():
        if len(values) < 2:
            continue
        numbers = list(values.values())

        directional = any(figure.endswith(suffix) or suffix in figure for suffix in DIRECTIONAL_SUFFIXES)
        signs = {_sign(n) for n in numbers if n != 0}
        if directional and len(signs) > 1:
            conflicts.append(Conflict(figure=figure, values=values, kind="direction"))
            continue

        low, high = min(numbers), max(numbers)
        scale = max(abs(low), abs(high))
        if scale > 0 and (high - low) / scale > AGREEMENT_TOLERANCE:
            conflicts.append(Conflict(figure=figure, values=values, kind="magnitude"))

    return conflicts


def _sign(value: float) -> int:
    return 1 if value > 0 else -1


# --- evidence ------------------------------------------------------------


def dedupe_evidence(outputs: dict[AgentName, AgentOutput]) -> list[Evidence]:
    """One entry per distinct (source, claim, detail), attribution preserved.

    Identical evidence from two agents is one fact, not two. Distinct claims from
    two sources are corroboration and both are kept — that is what makes an
    answer more trustworthy, and collapsing them would throw it away.
    """
    seen: dict[tuple[str, str, str], Evidence] = {}
    for agent in sorted(outputs):
        for item in outputs[agent].get("evidence") or []:
            key = (item.get("source_id", ""), item.get("claim", ""), item.get("detail", ""))
            if key not in seen:
                seen[key] = dict(item)  # type: ignore[assignment]
    return list(seen.values())


def _first_forecast(outputs: dict[AgentName, AgentOutput]) -> list[dict[str, Any]]:
    """The forecast agent's points if it ran, otherwise any agent's."""
    if "forecast" in outputs and outputs["forecast"].get("forecast"):
        return list(outputs["forecast"]["forecast"])
    for output in outputs.values():
        if output.get("forecast"):
            return list(output["forecast"])
    return []


# --- gaps ----------------------------------------------------------------


def _describe_gaps(
    failed: dict[AgentName, AgentOutput], never_reported: list[AgentName]
) -> list[str]:
    """What could not be answered, phrased for a reader, not a log file."""
    gaps: list[str] = []
    for agent, output in sorted(failed.items()):
        gaps.append(f"{_topic_of(agent)} could not be covered ({output.get('error', 'unknown error')})")
    for agent in sorted(never_reported):
        gaps.append(f"{_topic_of(agent)} did not return in time")
    return gaps


OUT_OF_SCOPE_PREFIX = "out_of_scope: "


def _out_of_scope_gaps(state: AgentState) -> list[str]:
    """Surface the router's out-of-scope finding in the answer, not just in state.

    The router already detects a query naming a sector CeyNex does not cover and
    records it in `errors`. Nothing read it back, so "how does tea compare with
    fisheries" returned a confident tea answer that never mentioned fisheries —
    the reader had no way to tell half their question was silently dropped.
    Omitting the limit is the same failure as inventing the figure.
    """
    gaps = []
    for error in state.get("errors", []) or []:
        text = str(error)
        if text.startswith(OUT_OF_SCOPE_PREFIX):
            note = text[len(OUT_OF_SCOPE_PREFIX) :].strip()
            gaps.append(note or "part of the question names a sector CeyNex does not cover")
    return gaps


TOPICS: dict[str, str] = {
    "export_analytics": "trade trends and market concentration",
    "agriculture_commodity": "the agriculture side (tea, cinnamon, rubber, coconut)",
    "apparel_manufacturing": "the apparel side (knit and woven garments)",
    "trade_economics": "the policy and exchange-rate simulation",
    "forecast": "the forward-looking projection",
}


def _topic_of(agent: str) -> str:
    return TOPICS.get(agent, agent.replace("_", " "))


# --- composing -----------------------------------------------------------


def compose_deterministic(
    query: str,
    outputs: dict[AgentName, AgentOutput],
    conflicts: list[Conflict],
    unanswered: list[str],
) -> str:
    """The answer when the LLM is unavailable (SRS 3.4.3).

    Still organised by finding rather than by agent: the degraded answer is
    plainer, not structurally different. Naming the agents here would violate
    SRS 3.1.2 just as much as naming them in the prose would.
    """
    sentences: list[str] = []

    # Ordered by confidence, so the best-supported finding leads.
    for _, output in sorted(
        outputs.items(), key=lambda kv: kv[1].get("confidence", 0.0), reverse=True
    ):
        summary = (output.get("summary") or "").strip()
        if summary:
            sentences.append(summary)

    for conflict in conflicts:
        sentences.append(conflict.describe())

    if unanswered:
        sentences.append("Not covered: " + "; ".join(unanswered) + ".")

    assumptions = _collect_assumptions(outputs)
    if assumptions:
        sentences.append("This rests on stated assumptions: " + " ".join(assumptions))

    return " ".join(sentences) if sentences else _nothing_succeeded(unanswered)


def _collect_assumptions(outputs: dict[AgentName, AgentOutput]) -> list[str]:
    seen: list[str] = []
    for agent in sorted(outputs):
        for assumption in outputs[agent].get("assumptions") or []:
            text = assumption.strip()
            if text and text not in seen:
                seen.append(text)
    return seen[:4]


def _merge_prompt(
    query: str,
    outputs: dict[AgentName, AgentOutput],
    conflicts: list[Conflict],
    unanswered: list[str],
) -> str:
    """Findings as structured context, deliberately not labelled by agent name.

    The model is told not to name the analyses; not giving it the names in the
    first place is the stronger guarantee.
    """
    blocks = [f"QUESTION: {query}", ""]

    for index, (_, output) in enumerate(
        sorted(outputs.items(), key=lambda kv: kv[1].get("confidence", 0.0), reverse=True), start=1
    ):
        blocks.append(f"FINDING {index} (confidence {output.get('confidence', 0):.2f}):")
        blocks.append(f"  {output.get('summary', '').strip()}")
        figures = output.get("figures") or {}
        if figures:
            blocks.append("  figures: " + ", ".join(f"{k}={v:,.4g}" for k, v in figures.items()))
        for assumption in (output.get("assumptions") or [])[:3]:
            blocks.append(f"  assumption: {assumption}")
        blocks.append("")

    if conflicts:
        blocks.append("DISAGREEMENTS you must state explicitly, with both figures:")
        blocks += [f"  {c.describe()}" for c in conflicts]
        blocks.append("")

    if unanswered:
        blocks.append("COULD NOT BE ANSWERED (say so in one clause):")
        blocks += [f"  {gap}" for gap in unanswered]

    return "\n".join(blocks)


def _ensure_conflicts_stated(answer: str, conflicts: list[Conflict]) -> str:
    """SRS 3.1.2 requires disagreements to reach the user. Not the LLM's choice."""
    missing = [c for c in conflicts if not _mentions(answer, c.figure)]
    if not missing:
        return answer
    return answer.rstrip() + " " + " ".join(c.describe() for c in missing)


def _ensure_gaps_stated(answer: str, unanswered: list[str]) -> str:
    """SAD §4.1 requires the user to be told what could not be answered."""
    if not unanswered:
        return answer
    lowered = answer.lower()
    if any(marker in lowered for marker in ("could not", "unable", "not covered", "no data")):
        return answer
    return answer.rstrip() + " Not covered: " + "; ".join(unanswered) + "."


def _mentions(answer: str, figure: str) -> bool:
    words = [w for w in _humanize(figure).split() if len(w) > 3]
    lowered = answer.lower()
    return bool(words) and all(word in lowered for word in words)


def _humanize(figure: str) -> str:
    return figure.replace("_usd", "").replace("_pct", "").replace("_", " ").strip()


def _nothing_succeeded(unanswered: list[str]) -> str:
    if unanswered:
        return (
            "This question could not be answered from the data currently loaded. "
            + "; ".join(unanswered).capitalize()
            + "."
        )
    return "This question could not be answered from the data currently loaded."


# How far behind "now" a source is expected to run before anything is wrong.
# UN Comtrade publishes an annual series through the following year and revises
# it after that, so a 2024 series being the newest available in mid-2026 is
# current data, not stale data.
EXPECTED_ANNUAL_PUBLICATION_LAG_MONTHS = 18.0


def _staleness_months(state: AgentState) -> float | None:
    """Months the newest observation lags *beyond* its source's normal cadence.

    Not absolute age. Penalising annual trade statistics for being a year old
    would mark every answer down for a property of the source rather than a
    problem with the answer — and since the penalty caps at 0.20, it would sit at
    the cap permanently and stop discriminating between anything.

    What the confidence formula is trying to capture is "this answer rests on
    data older than it should be", so the measure is the excess over the expected
    publication lag. Zero means the system is as current as the source allows.
    """
    from datetime import UTC, datetime

    latest_year: int | None = None
    for output in (state.get("agent_outputs") or {}).values():
        for item in output.get("evidence") or []:
            period = str(item.get("period", ""))
            for token in period.replace("-", " ").split():
                if token.isdigit() and len(token) == 4:
                    year = int(token)
                    latest_year = year if latest_year is None else max(latest_year, year)
    if latest_year is None:
        return None

    now = datetime.now(UTC)
    # Months from the end of the latest observed year to now.
    months_since = (now.year - latest_year) * 12 + (now.month - 12)
    return max(0.0, months_since - EXPECTED_ANNUAL_PUBLICATION_LAG_MONTHS)


__all__ = ["Conflict", "MergeResult", "compose_deterministic", "detect_conflicts", "dedupe_evidence", "merge"]
