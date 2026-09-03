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
import re
from dataclasses import dataclass, field
from typing import Any

from ceynex.contracts import AgentName, AgentOutput, AgentState, Evidence
from ceynex.orchestrator.confidence import aggregate_confidence, confidence_band
from ceynex.orchestrator.grounding import ungrounded_figures

log = logging.getLogger(__name__)

# Two agents reporting the same figure within this relative distance are
# agreeing. Wider than measurement noise, narrower than a real disagreement.
AGREEMENT_TOLERANCE = 0.10

# An agent can "succeed" (no error key) while genuinely having nothing to
# contribute -- an honest SAD Section 4.1 refusal (e.g.
# agriculture_commodity._unsupported_target), not a competing finding.
# Several agents' explicit refusal paths hardcode confidence=0.20; the
# weakest genuinely-informative finding traced in this codebase
# (agriculture_commodity._series_confidence under the maximum staleness
# penalty) sits at roughly 0.32, and ceynex.agents.common._confidence's
# generic empty-figures floor is 0.3. 0.25 sits cleanly between the two
# without needing to inspect figures directly (empty figures alone is not
# a safe signal -- plenty of genuine findings, especially in tests, never
# populate a figures dict at all).
DECLINE_CONFIDENCE_CEILING = 0.25

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
3a. A disagreement means two findings measuring THE SAME THING and getting different
   answers. Two findings that state different scopes, sources or reference periods are
   measuring DIFFERENT things and are not in conflict. When an assumption tells you two
   figures come from different source boundaries or different years, say in one clause
   which figure is which -- "USD X in 2025 on the HS-code basis, USD Y in 2024 on the
   national reporting basis" -- and move on. Never call that a discrepancy between
   sources, never present it as something the data cannot resolve, and never lead with it.
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
    # Figures the composed prose stated that no finding or evidence entry
    # supports (SRS 3.1.3). Non-empty means the prose was discarded and
    # `answer` is the deterministic composition instead — see
    # `_reject_ungrounded_prose`. Empty on every normal answer, including
    # every degraded one, since the deterministic path cannot invent a figure.
    ungrounded: list[str] = field(default_factory=list)

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
    no_topic = no_topic_recognized(state)

    # A routed agent still runs and can still answer confidently even when the
    # question named nothing CeyNex covers -- export_analytics.py:62's own
    # `item = intent.item or "tea"` default is exactly this: no item named, so
    # it substitutes one and reports on it as if asked. That is real, valid
    # output for a question that *was* about tea; it is noise for one that
    # was never about trade at all ("who is Euler", found live 2026-08-27).
    # Treating the whole route as if nothing succeeded is what the "mixed
    # question" case (an excluded sector named *alongside* tea or apparel)
    # must not get -- that one still deserves the real in-scope answer, which
    # is why this only fires on `no_topic`, not on `out_of_scope` generally.
    succeeded = (
        {}
        if no_topic
        else {name: out for name, out in outputs.items() if not out.get("error")}
    )
    failed = {name: out for name, out in outputs.items() if out.get("error")}
    never_reported = [name for name in route if name not in outputs]

    # Split out honest refusals (see DECLINE_CONFIDENCE_CEILING) so they read
    # as a stated gap rather than a "finding" that appears to conflict with
    # a genuinely successful agent. Found live 2026-08-26: a cinnamon
    # forecast question was narrated as "uncertain due to conflicting
    # findings" because agriculture_commodity's confidence=0.20 refusal
    # ("no model target was requested") was presented as a peer FINDING
    # alongside the forecast agent's real, correct answer -- even though
    # detect_conflicts never found an actual numeric disagreement between
    # them, since a refusal has no figures to disagree with in the first
    # place. Evidence, agents_used and the forecast are still drawn from the
    # full `succeeded` set below -- only the merge prose's FINDING list and
    # conflict detection are narrowed.
    contributing, declined = _split_succeeded(succeeded)

    unanswered = (
        _describe_gaps(failed, never_reported)
        + _describe_declines(state["query"], declined, contributing)
        + _out_of_scope_gaps(state)
    )
    conflicts = detect_conflicts(contributing)
    evidence = dedupe_evidence(succeeded)
    forecast = _first_forecast(succeeded)

    # aggregate_confidence(outputs=...) reads the *unfiltered* agent outputs,
    # so it would otherwise score this on export_analytics's real (and often
    # high) confidence in its own irrelevant-to-this-question answer -- a 90%
    # -confidence "who is Euler" reply is worse than a wrong number, since it
    # tells the reader to trust it.
    confidence = (
        NO_TOPIC_CONFIDENCE
        if no_topic
        else aggregate_confidence(
            outputs,
            route=route,
            relevance=relevance,
            months_since_latest_observation=_staleness_months(state),
            dq_severities=list(dq_severities) + _dq_severities_from_evidence(evidence),
        )
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

    deterministic = compose_deterministic(state["query"], contributing, conflicts, unanswered)

    prose = await llm.generate(
        "merge",
        MERGE_SYSTEM,
        _merge_prompt(state["query"], contributing, conflicts, unanswered),
    )
    degraded = not prose

    # SRS 3.1.3. Checked against the raw model output, before the guards below
    # append any deterministic sentence of their own — those restate figures
    # the agents reported, so including them would let the guard's own text
    # vouch for the prose it is meant to be policing.
    ungrounded = _reject_ungrounded_prose(prose, state["query"], contributing, evidence)
    if ungrounded:
        prose = ""

    answer = prose or deterministic
    if prose:
        # The LLM is told to surface conflicts, but the requirement is not
        # optional, so the deterministic sentence is appended if it did not.
        answer = _ensure_conflicts_stated(answer, conflicts)
        answer = _ensure_gaps_stated(answer, unanswered)
        answer = _ensure_lists_referenced(answer, evidence)

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
        ungrounded=ungrounded,
    )


# --- grounding (SRS 3.1.3) -----------------------------------------------


def _reject_ungrounded_prose(
    prose: str,
    query: str,
    outputs: dict[AgentName, AgentOutput],
    evidence: list[Evidence],
) -> list[str]:
    """Figures the composed prose asserts that nothing it was given supports.

    Returning a non-empty list means `merge` discards the prose and serves
    `compose_deterministic` instead. That is a real cost — the deterministic
    answer is plainer — and it is the right trade: a fabricated figure in a
    fluent sentence is the one failure mode a reader cannot detect, and this
    system's whole claim is that its numbers are traceable. Serving the
    invented figure with a caveat attached would keep the fluency and give up
    the claim.

    Discarding wholesale rather than excising the offending number is
    deliberate too. A sentence built around a figure does not survive having
    that figure cut out of it, and the surrounding clauses are exactly as
    unsupported as the number was.
    """
    if not prose:
        return []
    missing = ungrounded_figures(prose, _grounding_corpus(query, outputs, evidence))
    if missing:
        log.warning(
            "merge prose discarded: %d figure(s) not supported by any finding or "
            "evidence entry (%s); serving the deterministic composition instead",
            len(missing),
            ", ".join(missing),
        )
    return missing


def _grounding_corpus(
    query: str, outputs: dict[AgentName, AgentOutput], evidence: list[Evidence]
) -> list[str]:
    """Everything the merge LLM was shown, as text to draw figures from.

    Mirrors `_merge_prompt` plus the evidence. The figures are rendered in
    several forms on purpose: the prompt shows `f"{value:,.4g}"`, so a large
    number reaches the model as `1.235e+09` and can honestly come back as
    "1.235 billion", while the same value's plain `str()` is what a summary
    sentence would contain. Only offering one spelling would reject correct
    prose for restating a figure in the form it was given.
    """
    texts: list[str] = [query]
    for output in outputs.values():
        texts.append(output.get("summary") or "")
        for value in (output.get("figures") or {}).values():
            texts.append(str(value))
            if isinstance(value, int | float):
                texts.append(f"{value:,.4g}")
                texts.append(f"{value:,.2f}")
        texts.extend(output.get("assumptions") or [])
    for item in evidence:
        texts.append(str(item.get("claim", "")))
        texts.append(str(item.get("detail", "")))
    return texts


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


def _dq_severities_from_evidence(evidence: list[Evidence]) -> list[str]:
    """Real cross-source discrepancy severities, read from the merged evidence.

    Closes a real gap found live 2026-08-26: `aggregate_confidence`'s `dq`
    term (SRS 3.1.8) had a correct, unit-tested formula, but `merge()` never
    called it with anything but the default `dq_severities=()` -- nothing in
    the actual orchestrator graph ever passed real severities in, so the
    term always contributed 0 regardless of real `dq_flag` data.

    Reading it back out of the evidence (`agriculture_commodity.py`'s
    `_with_dq_flags` already emits a `source_id="DQ_FLAG"` entry per flag,
    with `severity=...` in `detail`) rather than requiring a new typed
    channel is deliberate: `AgentState` is the frozen contracts package (a
    change needs 3-way approval), and this way any current or future agent
    that surfaces a DQ_FLAG entry in the same shape is picked up here with
    no orchestrator-side knowledge of which agent or which item it came
    from -- consistent with SRS 3.1.2's "organised by finding, not by
    source".

    This is also why `agriculture_commodity._with_dq_flags` no longer
    subtracts `dq_penalty` from its own agent-level confidence: doing so
    there *and* here would double-penalize the same flag. The penalty now
    applies exactly once, centrally, matching confidence.py's own
    docstring ("nothing else in the codebase is allowed to invent its
    own").
    """
    severities: list[str] = []
    for item in evidence:
        if item.get("source_id") != "DQ_FLAG":
            continue
        detail = item.get("detail") or ""
        for token in detail.split(";"):
            key, _, value = token.strip().partition("=")
            if key == "severity" and value:
                severities.append(value)
    return severities


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


def _split_succeeded(
    succeeded: dict[AgentName, AgentOutput],
) -> tuple[dict[AgentName, AgentOutput], dict[AgentName, AgentOutput]]:
    """(contributing, declined) -- see DECLINE_CONFIDENCE_CEILING."""
    contributing = {
        name: out for name, out in succeeded.items()
        if out.get("confidence", 0.0) >= DECLINE_CONFIDENCE_CEILING
    }
    declined = {name: out for name, out in succeeded.items() if name not in contributing}
    return contributing, declined


def unanswered_from_outputs(state: AgentState) -> list[str]:
    """The same "could not be answered" list `merge()` uses internally, public.

    `unanswered` is not part of `AgentState` (the frozen contracts package --
    a change there needs 3-way approval) or `MergeResult.as_state_patch()`
    (only `final_answer`/`final_confidence`/`merged_evidence` are written
    back to state), so a caller that only has the graph's final `state` --
    the API route, notably -- had no way to reconstruct it and fell back to
    "agents that hard-failed" only, silently dropping honest declines and
    out-of-scope gaps from `QueryResponse.unanswered`. This is the shared
    source of truth both `merge()` and that caller should use instead of
    each recomputing (or under-computing) their own version.
    """
    outputs = state.get("agent_outputs", {})
    route = list(state.get("route", []) or outputs.keys())
    failed = {name: out for name, out in outputs.items() if out.get("error")}
    never_reported = [name for name in route if name not in outputs]
    # Mirrors merge()'s own no_topic suppression -- see no_topic_recognized()'s
    # docstring. A routed agent's real output is noise, not a gap, when the
    # question named nothing CeyNex covers; only the out-of-scope note itself
    # belongs in "could not be answered".
    if no_topic_recognized(state):
        return _out_of_scope_gaps(state)
    succeeded = {name: out for name, out in outputs.items() if not out.get("error")}
    contributing, declined = _split_succeeded(succeeded)
    return (
        _describe_gaps(failed, never_reported)
        + _describe_declines(state["query"], declined, contributing)
        + _out_of_scope_gaps(state)
    )


def agents_used_from_outputs(state: AgentState) -> list[str]:
    """Same suppression `merge()` applies internally, public -- the API route
    needs this rather than recomputing "succeeded" itself, same reasoning as
    `unanswered_from_outputs()` above (see `no_topic_recognized()`'s docstring
    for the bug this closes)."""
    if no_topic_recognized(state):
        return []
    outputs = state.get("agent_outputs", {})
    return sorted(name for name, out in outputs.items() if not out.get("error"))


def _describe_declines(
    query: str,
    declined: dict[AgentName, AgentOutput],
    contributing: dict[AgentName, AgentOutput],
) -> list[str]:
    """An honest SAD Section 4.1 refusal, phrased for a reader.

    Reuses the agent's own summary (it already states the specific reason,
    e.g. "No registered national tea export volume model is available")
    rather than a generic topic label -- the same principle _describe_gaps
    applies to a genuine failure's error text.

    A decline that another finding already answered is dropped. Found live
    2026-09-03: every non-tea agriculture question came back *correct* from
    export_analytics and still carried "No sourced export volume series is
    available for rubber…" in `unanswered`, because agriculture_commodity was
    also routed and holds only tea volume and cinnamon price. S03, CO1, X02 and
    M01 all reported a full, grounded answer that read as half-failed. Telling
    the reader a question went unanswered when it was answered is the same class
    of error as omitting a limit that was real -- both misdescribe the answer.
    """
    gaps: list[str] = []
    for agent, output in sorted(declined.items()):
        summary = (output.get("summary") or "").strip()
        if summary and _covered_by_another_finding(query, summary, contributing):
            log.info("%s declined, but another finding covered it; not surfacing as a gap", agent)
            continue
        gaps.append(summary or f"{_topic_of(agent)} had nothing to add")
    return gaps


def _covered_by_another_finding(
    query: str, decline: str, contributing: dict[AgentName, AgentOutput]
) -> bool:
    """Did a contributing finding already answer what this decline declines?

    The decline's *subject* is what it shares with the question: "rubber" and
    "export" for "no sourced export volume series is held for rubber" asked
    against "how concentrated are Sri Lanka's rubber export destinations?". When
    every one of those words appears in some contributing finding, that finding
    covered the same ground, and the decline describes a gap in one agent's own
    sources rather than a gap in the answer.

    Deliberately not the word-overlap ratio `_gap_already_stated` uses. That one
    asks "has the answer already said this?" of two texts that are both phrased
    as declines, so a loose threshold is right. This asks whether a *finding*
    covers a *decline* -- two texts with almost nothing in common except the
    subject -- so it matches on the subject alone, and requires all of it. A
    partial match is how "no district share is recorded for cinnamon" would get
    swallowed by a cinnamon market-share answer that never mentions districts.
    """
    subject = _significant_words(query) & _significant_words(decline)
    if not subject:
        return False
    for output in contributing.values():
        covered = " ".join(
            [
                output.get("summary") or "",
                *(str(item.get("claim", "")) for item in output.get("evidence") or []),
            ]
        ).lower()
        if all(word in covered for word in subject):
            return True
    return False


def _significant_words(text: str) -> set[str]:
    """Words long enough to carry a subject. Same filter as `_gap_already_stated`."""
    return {word for word in re.findall(r"[a-z]+", text.lower()) if len(word) > 3}


OUT_OF_SCOPE_PREFIX = "out_of_scope: "
NO_TOPIC_MARKER = "out_of_scope_no_topic: true"

# Not merely "no data for this item" (DECLINE_CONFIDENCE_CEILING) -- the
# question named nothing CeyNex covers at all, so there's no evidence quality
# to score in the first place.
NO_TOPIC_CONFIDENCE = 0.15


def no_topic_recognized(state: AgentState) -> bool:
    """Public: `merge()` uses this internally, and so must every caller that
    independently recomputes something from raw `state` instead of from a
    `MergeResult` -- `agents_used_from_outputs()` and `unanswered_from_outputs()`
    below, and the API route's own forecast lookup. Found live 2026-08-27:
    `merge()`'s own no-topic suppression (added in the PR that introduced this
    flag) landed correctly in `final_answer`/`final_confidence`/`merged_evidence`
    (the fields `MergeResult.as_state_patch()` writes back), but the API
    route's separately-computed `agents_used` and `unanswered` fields still
    came from the raw, unsuppressed `agent_outputs` -- so a "whats 4+4"
    response correctly said "could not be answered" with 0 evidence, while
    still listing `agents_used: ["export_analytics"]` and an `unanswered`
    entry containing a full, irrelevant tea report.
    """
    return NO_TOPIC_MARKER in (state.get("errors", []) or [])


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
            # Belt to the router's braces. Both routers now populate `notes`, so
            # this should never fire -- and when it does, it must not assert a
            # *sector* was named. It used to, and with `llm_route` supplying no
            # note it fired on every LLM-router out-of-scope verdict: "who was
            # Leonhard Euler?" was told it had named a sector CeyNex does not
            # cover (live 2026-09-03).
            gaps.append(note or "part of the question is outside what CeyNex covers")
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


_GAP_MARKERS = (
    "could not", "cannot", "can't", "unable", "not covered",
    "no data", "not available", "no specific",
)


def _gap_already_stated(answer: str, item: str) -> bool:
    """Is this specific gap already conveyed in the answer, in any wording?

    Two independent checks, either is enough:

    - a handful of common decline-phrasing markers anywhere in the answer
      (cheap, and still catches the common case);
    - most of *this item's own* significant words already appearing in the
      answer, regardless of which words those are.

    The marker list alone is not enough on its own: found live 2026-08-26
    ("cannot be provided"/"is not available" matched none of the original
    four markers), and recurred live 2026-08-27 with yet another phrasing
    ("no available...", "no compatible... that can be substituted") that
    also matched none of them -- proof a fixed vocabulary can't keep up with
    open-ended LLM paraphrasing. Checking the gap's own words instead doesn't
    depend on guessing every way a decline might be phrased.
    """
    lowered = answer.lower()
    if any(marker in lowered for marker in _GAP_MARKERS):
        return True
    words = [w for w in re.findall(r"[a-z]+", item.lower()) if len(w) > 3]
    if not words:
        return False
    hits = sum(1 for word in words if word in lowered)
    return hits / len(words) >= 0.6


def _ensure_gaps_stated(answer: str, unanswered: list[str]) -> str:
    """SAD §4.1 requires the user to be told what could not be answered.

    Per item, not all-or-nothing: only the gaps `_gap_already_stated` cannot
    find in the answer get appended, rather than either appending the whole
    list or none of it based on the answer as a whole.
    """
    missing = [item for item in unanswered if not _gap_already_stated(answer, item)]
    if not missing:
        return answer
    return answer.rstrip() + " Not covered: " + "; ".join(missing) + "."


_LIST_SHAPED_MIN_ITEMS = 10


def _is_list_shaped(claim: str) -> bool:
    """A claim that itself enumerates many items (a full country list, say)
    rather than stating one figure -- comma-separated segments with no
    digits in them, so a claim dense with thousand-separated dollar figures
    (also comma-heavy) doesn't false-positive: "USD 1,149,365,364" splits
    into digit-only segments, a country list splits into name-only ones.
    """
    segments = [s.strip() for s in claim.split(",")]
    name_like = sum(1 for s in segments if s and not any(ch.isdigit() for ch in s))
    return name_like >= _LIST_SHAPED_MIN_ITEMS


def _ensure_lists_referenced(answer: str, evidence: list[Evidence]) -> str:
    """`_merge_prompt` never forwards evidence to the merge LLM, only each
    finding's summary/figures/assumptions (kept lean and cheap) -- so when the
    answer *is* a list (every partner's name, say), the merge LLM only ever
    sees the count, not the names, and has no way to know the evidence panel
    already carries them. Found live 2026-08-27: "give the names of all
    countries..." answered "the specific names ... were not provided" --
    false, they were sitting in the evidence panel the whole time, just
    outside what this LLM call was ever shown.
    """
    if not any(_is_list_shaped(e["claim"]) for e in evidence):
        return answer
    lowered = answer.lower()
    if "evidence" in lowered:
        return answer
    return answer.rstrip() + " The full list is in the evidence panel below."


def _mentions(answer: str, figure: str) -> bool:
    words = [w for w in _humanize(figure).split() if len(w) > 3]
    lowered = answer.lower()
    return bool(words) and all(word in lowered for word in words)


def _humanize(figure: str) -> str:
    return figure.replace("_usd", "").replace("_pct", "").replace("_", " ").strip()


def _nothing_succeeded(unanswered: list[str]) -> str:
    if unanswered:
        joined = "; ".join(unanswered)
        # Not `.capitalize()` -- it lowercases everything after the first
        # character, which mangles "CeyNex"/"SRS" the moment a gap sentence
        # (e.g. the out-of-scope note below) contains one.
        sentence = joined[:1].upper() + joined[1:] if joined else joined
        return f"This question could not be answered from the data currently loaded. {sentence}."
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


__all__ = [
    "Conflict",
    "MergeResult",
    "agents_used_from_outputs",
    "compose_deterministic",
    "detect_conflicts",
    "dedupe_evidence",
    "merge",
    "no_topic_recognized",
    "unanswered_from_outputs",
]
