"""Implements the M2 plan's Day 9 evaluation — the orchestrator's measured behaviour.

    python -m eval.harness                     # the whole set
    python -m eval.harness --category simulation
    python -m eval.harness --degraded          # with the LLM forced unavailable
    python -m eval.harness --json results.json

Runs every question in `eval/questions.yaml` through the same entry point the API
uses, and reports five things:

**Routing accuracy** — the fraction of questions whose agent set matches the one
written down before the run. Reported two ways, because they answer different
questions: *exact* set equality, and *recall* (did the router at least include
every agent that was needed). A router that fans out to all five agents every
time scores 1.00 on recall and near 0 on exact — reporting only recall would
hide that, and reporting only exact would punish a router that adds one
defensible extra agent as harshly as one that misses the only relevant one.

**Evidence grounding** — every figure in the merged answer must appear in some
`Evidence` entry attached to it. A number in the prose with no evidence behind it
is the failure mode that matters most in a system whose whole claim is
traceability, so ungrounded figures are counted individually, not just flagged.

**Latency** — p50 and p95 by category, against SRS 3.4.1 (10 s single-sector,
20 s cross-sector).

**Refusal correctness** — for the three questions with no answer, did the system
say so rather than inventing one.

**Degraded-mode correctness** — the same set with the LLM unavailable must still
return figures and evidence (SRS 3.4.3), not exceptions.

**Citation discipline** — when `CEYNEX_CITATIONS=on` puts `[n]` markers in the
prose, every marker must index a real evidence entry and every sentence that
states a figure should carry one. Reported whether or not markers were present,
so a run with the flag off and a run with it on have the same shape.

**Repeated runs** — `--repeat N --cold` runs the set N times with the prompt
cache cleared before each, and `eval/repeat.py` reports medians and spread.
§8 of `docs/EVALUATION.md` measured why: one run's difference of one question
is inside the noise floor and is not a result.

This measures the machine. Merge coherence needs human raters and lives in
`eval/coherence.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ceynex.orchestrator import grounding

log = logging.getLogger(__name__)

QUESTIONS = Path(__file__).parent / "questions.yaml"

# Phrases that count as the system declining to answer. Matched case-insensitively
# against the merged answer and the stated assumptions.
REFUSAL_MARKERS = (
    "cannot be simulated",
    "cannot be completed",
    "could not be",
    "not be answered",
    "no data",
    "not in the",
    "outside the",
    "out of scope",
    "does not cover",
    "no records",
    "no export records",
    "holds no",
    "not recorded",
    "unable to",
    "does not cover",
    "not one of the sectors",
)

# A bare number in prose. Used to check every claimed figure traces to evidence.
# Defined in ceynex.orchestrator.grounding, which the runtime guard also uses:
# the measurement and the guarantee have to agree on what counts as a figure.
# Re-exported here because this name was part of the harness first.
NUMBER = grounding.NUMBER

# An inline citation marker, exactly as `CitedAnswer.tsx` recognises one, so the
# harness counts the same markers a reader would see linked.
CITATION = re.compile(r"\[(\d{1,2})\]")

# A sentence boundary for the citation metric: terminal punctuation, optional
# closing quote or bracket, then whitespace. Crude, and the same in both
# directions — a mis-split sentence costs at most one count either way.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


@dataclass
class Result:
    """One question's outcome."""

    id: str
    category: str
    question: str
    expected_route: list[str]
    actual_route: list[str]
    agents_used: list[str]
    answerable: bool
    partial: bool
    elapsed_ms: float
    confidence: float
    degraded: bool
    evidence_count: int
    answer: str
    ungrounded_figures: list[str] = field(default_factory=list)
    refused: bool = False
    error: str | None = None
    #: Inline citations (`CEYNEX_CITATIONS`). All zero when the flag is off.
    citations_total: int = 0
    citations_valid: int = 0
    figure_sentences: int = 0
    figure_sentences_cited: int = 0

    @property
    def route_exact(self) -> bool:
        return set(self.actual_route) == set(self.expected_route)

    @property
    def route_recall(self) -> float:
        """Fraction of the expected agents the router actually selected."""
        if not self.expected_route:
            return 1.0
        hit = len(set(self.expected_route) & set(self.actual_route))
        return hit / len(set(self.expected_route))


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text())["questions"]


async def run_question(question: dict[str, Any], *, use_llm: bool) -> Result:
    """Run one question through the orchestrator. Never raises."""
    from ceynex.orchestrator.demo import answer

    common = {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "expected_route": list(question.get("expected_route", [])),
        "answerable": bool(question.get("answerable", True)),
        "partial": bool(question.get("partial", False)),
    }

    try:
        result = await answer(question["question"], use_llm=use_llm, user_id="eval")
    except Exception as exc:  # noqa: BLE001 - a crash is a result, and the run continues
        log.warning("%s raised: %s", question["id"], exc)
        return Result(
            **common,
            actual_route=[],
            agents_used=[],
            elapsed_ms=0.0,
            confidence=0.0,
            degraded=True,
            evidence_count=0,
            answer="",
            error=str(exc),
        )

    text = result.get("answer", "") or ""
    evidence = result.get("evidence", []) or []

    return Result(
        **common,
        actual_route=list(result.get("route", [])),
        agents_used=list(result.get("agents_used", [])),
        elapsed_ms=float(result.get("elapsed_ms", 0.0)),
        confidence=float(result.get("confidence", 0.0)),
        degraded=bool(result.get("degraded", False)),
        evidence_count=len(evidence),
        answer=text,
        ungrounded_figures=ungrounded(text, evidence),
        refused=is_refusal(text, result),
        **citation_counts(text, len(evidence)),
    )


def ungrounded(answer: str, evidence: list[dict[str, Any]]) -> list[str]:
    """Figures in the prose that appear in no evidence entry.

    Deliberately crude and deliberately generous: it compares digit strings, so a
    number rounded differently in the prose than in the evidence is reported as
    ungrounded. That over-reports rather than under-reports, which is the right
    direction for a metric whose purpose is to catch hallucinated figures — a
    false alarm costs a manual check, a miss costs the claim.

    **Evidence only**, deliberately, and this is where the metric is stricter
    than the runtime guard in `ceynex.orchestrator.merger`: that one also
    accepts a figure quoted from the question or stated in a finding's summary,
    because rejecting an answer for restating its own question would be wrong.
    Here the claim being measured is the stronger one — every figure traces to
    a cited source — so a shock magnitude the user supplied genuinely does not
    count as grounded, and the report says so rather than quietly widening the
    denominator. The comparison itself is shared, so both agree on what a
    figure is.
    """
    return grounding.ungrounded_figures(
        answer,
        [str(e.get("claim", "")) + " " + str(e.get("detail", "")) for e in evidence],
    )


def strip_citations(text: str) -> str:
    """The prose without its `[n]` markers.

    `grounding.STRUCTURAL_DIGIT_LIMIT` already keeps a two-digit marker out of
    the ungrounded check, so this is not needed for grounding; it is needed so a
    marker's digits are never mistaken for a figure by the sentence count below.
    """
    return CITATION.sub("", text or "")


def citation_counts(answer: str, evidence_count: int) -> dict[str, int]:
    """How disciplined the prose's citations are.

    - `citations_total` / `citations_valid`: every `[n]` and how many of them
      index an evidence entry that exists (1 ≤ n ≤ evidence_count). A marker
      pointing past the list is the citation equivalent of an invented figure.
    - `figure_sentences` / `figure_sentences_cited`: sentences stating a figure
      — a digit string longer than `STRUCTURAL_DIGIT_LIMIT` — and how many of
      them carry at least one marker.
    """
    markers = [int(m) for m in CITATION.findall(answer or "")]
    valid = sum(1 for n in markers if 1 <= n <= evidence_count)

    figure_sentences = cited = 0
    for sentence in SENTENCE_BOUNDARY.split(answer or ""):
        bare = strip_citations(sentence)
        has_figure = any(
            len(raw.replace(",", "").lstrip("-").replace(".", "")) > grounding.STRUCTURAL_DIGIT_LIMIT
            for raw in NUMBER.findall(bare)
        )
        if not has_figure:
            continue
        figure_sentences += 1
        if CITATION.search(sentence):
            cited += 1
    return {
        "citations_total": len(markers),
        "citations_valid": valid,
        "figure_sentences": figure_sentences,
        "figure_sentences_cited": cited,
    }


# The "Not covered:" clause carries two different kinds of gap, and they must be
# scored differently. A teammate's agent not being built yet says nothing about
# whether the question was answerable, so those fragments are dropped. A gap
# because the question named an uncovered sector is exactly the behaviour being
# measured, so those stay.
UNIMPLEMENTED = re.compile(r"[^;.]*not implemented yet[^;.]*[;.]?", re.IGNORECASE)


def is_refusal(answer: str, result: dict[str, Any]) -> bool:
    """Did the system state a limit on answering the question it was asked?

    **Structural first.** The orchestrator already knows what it could not
    cover and records it in `unanswered` (`merger.unanswered_from_outputs`), so
    the honest test is whether that list is non-empty — not whether the prose
    happens to contain a phrase this module guessed in advance.

    That distinction is not academic. Measured 2026-08-28 on the same 30
    questions: the marker list scored refusals at **100%** in degraded mode and
    **33%** with the LLM writing the prose. The system behaved identically; the
    deterministic composer simply uses the vocabulary the list was built from,
    and the LLM paraphrases. X11 ("data for the fisheries sector is not
    available for comparison, so a direct comparison ... cannot be made") is a
    textbook correct refusal that matched no marker. `merger._gap_already_stated`
    documents the same lesson from the other side — a fixed vocabulary cannot
    keep up with open-ended paraphrasing.

    The marker list is kept as a fallback for results that predate `unanswered`
    (older `--json` dumps replayed through `report`), and an empty-answer check
    stays first because producing nothing at all is a refusal whatever the
    fields say.
    """
    substantive = UNIMPLEMENTED.sub("", answer).strip()
    if not substantive:
        return True
    if "unanswered" in result:
        return bool(result["unanswered"])
    return any(marker in substantive.lower() for marker in REFUSAL_MARKERS)


# --- reporting -----------------------------------------------------------


def report(results: list[Result]) -> dict[str, Any]:
    """Aggregate metrics. Every rate is reported with its denominator."""
    total = len(results)
    crashed = [r for r in results if r.error]
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    summary: dict[str, Any] = {
        "questions": total,
        "crashed": len(crashed),
        "routing": {
            "exact_match": _rate([r.route_exact for r in results]),
            "recall": round(statistics.fmean([r.route_recall for r in results]), 4) if results else 0.0,
            "never_empty": _rate([bool(r.actual_route) for r in results]),
        },
        "evidence": {
            "answers_fully_grounded": _rate([not r.ungrounded_figures for r in answerable]),
            "ungrounded_figures_total": sum(len(r.ungrounded_figures) for r in answerable),
            "mean_evidence_per_answer": round(
                statistics.fmean([r.evidence_count for r in answerable]), 2
            )
            if answerable
            else 0.0,
            "answers_with_no_evidence": sum(1 for r in answerable if r.evidence_count == 0),
        },
        "refusal": {
            # The three questions with no honest answer. This is the number that
            # says whether the system hallucinates when it has nothing.
            "unanswerable_correctly_refused": _rate([r.refused for r in unanswerable]),
            # An answerable question that produced no content at all: a real
            # failure, distinct from one that answered and noted a limit.
            "answerable_with_no_content": _rate(
                [not r.answer.strip() for r in answerable]
            ),
            # Informational, not an error rate: stating "CAGR could not be
            # computed, the base year is zero" while still answering is the
            # behaviour SAD §4.1 asks for, and counting it as a failure would
            # push the system toward hiding its gaps.
            "answerable_that_stated_some_limit": _rate([r.refused for r in answerable]),
        },
        "latency_ms": _latency(results),
        "degraded_answers": sum(1 for r in results if r.degraded),
        "citations": {
            # Present in every run, marker or not, so an "off" run and an "on"
            # run summarise to the same shape and can be diffed field for field.
            "answers_with_markers": sum(1 for r in answerable if r.citations_total),
            "marker_valid_rate": _ratio(
                sum(r.citations_valid for r in answerable),
                sum(r.citations_total for r in answerable),
            ),
            "figure_sentences_cited_rate": _ratio(
                sum(r.figure_sentences_cited for r in answerable),
                sum(r.figure_sentences for r in answerable),
            ),
        },
    }
    return summary


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """Like `_rate`, for counts that are already summed."""
    if not denominator:
        return {"rate": None, "of": 0}
    return {"rate": round(numerator / denominator, 4), "of": denominator}


def _rate(flags: list[bool]) -> dict[str, Any]:
    """A rate always carries its denominator; '100%' of two is not a result."""
    if not flags:
        return {"rate": None, "of": 0}
    return {"rate": round(sum(flags) / len(flags), 4), "of": len(flags)}


def _latency(results: list[Result]) -> dict[str, Any]:
    budgets = {"single_sector": 10_000, "cross_sector": 20_000, "simulation": 20_000}
    out: dict[str, Any] = {}
    for category in sorted({r.category for r in results}):
        times = sorted(r.elapsed_ms for r in results if r.category == category and not r.error)
        if not times:
            continue
        budget = budgets.get(category)
        out[category] = {
            "p50": round(_percentile(times, 50), 1),
            "p95": round(_percentile(times, 95), 1),
            "max": round(times[-1], 1),
            "budget_ms": budget,
            "within_budget": all(t <= budget for t in times) if budget else None,
        }
    return out


def _percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank percentile. On 12 samples an interpolated p95 is false precision."""
    if not ordered:
        return 0.0
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered) + 0.5)))
    return ordered[rank - 1]


def render(results: list[Result], summary: dict[str, Any]) -> str:
    lines = ["", "=" * 78, "CeyNex orchestrator evaluation", "=" * 78, ""]

    lines.append(f"{'id':5s} {'category':14s} {'route':6s} {'ev':>3s} {'conf':>5s} {'ms':>8s}  question")
    lines.append("-" * 78)
    for r in results:
        mark = "ok" if r.route_exact else ("part" if r.route_recall > 0 else "MISS")
        lines.append(
            f"{r.id:5s} {r.category:14s} {mark:6s} {r.evidence_count:3d} "
            f"{r.confidence:5.2f} {r.elapsed_ms:8.1f}  {r.question[:40]}"
        )

    lines += ["", "-" * 78, json.dumps(summary, indent=2), ""]

    misrouted = [r for r in results if not r.route_exact]
    if misrouted:
        lines.append("Routing differences (expected -> actual):")
        for r in misrouted:
            lines.append(f"  {r.id}: {sorted(r.expected_route)} -> {sorted(r.actual_route)}")
        lines.append("")

    ungrounded_any = [r for r in results if r.ungrounded_figures]
    if ungrounded_any:
        lines.append("Figures with no supporting evidence (check each by hand):")
        for r in ungrounded_any:
            lines.append(f"  {r.id}: {r.ungrounded_figures}")
        lines.append("")

    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------


async def run_all(questions: list[dict[str, Any]], *, use_llm: bool) -> list[Result]:
    """Sequential on purpose: concurrent runs would make the latency numbers meaningless."""
    results = []
    for question in questions:
        result = await run_question(question, use_llm=use_llm)
        log.info("%s %s (%.0f ms)", result.id, "ok" if not result.error else "ERROR", result.elapsed_ms)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CeyNex evaluation set.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--category", choices=["single_sector", "cross_sector", "simulation"])
    parser.add_argument("--id", help="run a single question by id")
    parser.add_argument("--degraded", action="store_true", help="force the LLM unavailable (SRS 3.4.3)")
    parser.add_argument("--json", type=Path, help="write full results here")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run the set this many times and report medians (eval/repeat.py)")
    parser.add_argument("--cold", action="store_true",
                        help="clear the prompt cache before each run, so every call is paid for")
    parser.add_argument("--json-dir", type=Path,
                        help="with --repeat: write run-N.json per run and summary.json here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    questions = load_questions(args.questions)
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.id:
        questions = [q for q in questions if q["id"] == args.id]
    if not questions:
        print("no questions matched", file=sys.stderr)
        return 2
    if args.repeat < 1:
        print("--repeat must be at least 1", file=sys.stderr)
        return 2

    from eval import repeat as repeat_mod

    summaries: list[dict[str, Any]] = []
    runs: list[list[dict[str, Any]]] = []
    for index in range(1, args.repeat + 1):
        if args.cold:
            cleared = repeat_mod.clear_prompt_cache()
            print(f"run {index}/{args.repeat}: cleared {cleared} cached completions")
        results = asyncio.run(run_all(questions, use_llm=not args.degraded))
        summary = report(results)
        print(render(results, summary))
        payload = {
            "degraded_run": args.degraded,
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        summaries.append(summary)
        runs.append(payload["results"])

        target = args.json
        if args.json_dir:
            args.json_dir.mkdir(parents=True, exist_ok=True)
            target = args.json_dir / f"run-{index}.json"
        if target:
            target.write_text(json.dumps(payload, indent=2))
            print(f"wrote {target}")

    if args.repeat > 1 or args.json_dir:
        out_dir = args.json_dir or (args.json.parent if args.json else Path("."))
        out_dir.mkdir(parents=True, exist_ok=True)
        written = repeat_mod.write_repeat_summary(out_dir, summaries, runs, degraded=args.degraded)
        print(f"wrote {written}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
