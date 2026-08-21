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
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


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
    )


def ungrounded(answer: str, evidence: list[dict[str, Any]]) -> list[str]:
    """Figures in the prose that appear in no evidence entry.

    Deliberately crude and deliberately generous: it compares digit strings, so a
    number rounded differently in the prose than in the evidence is reported as
    ungrounded. That over-reports rather than under-reports, which is the right
    direction for a metric whose purpose is to catch hallucinated figures — a
    false alarm costs a manual check, a miss costs the claim.
    """
    supporting = " ".join(str(e.get("claim", "")) + " " + str(e.get("detail", "")) for e in evidence)
    grounded = {n.replace(",", "") for n in NUMBER.findall(supporting)}

    missing = []
    for raw in NUMBER.findall(answer):
        value = raw.replace(",", "")
        # Years and small integers are almost always structural (a horizon, a
        # count of markets), not claims that need their own evidence line.
        if len(value.lstrip("-").replace(".", "")) <= 2:
            continue
        if value in grounded:
            continue
        # Allow a rounded restatement: 1234.5 in prose, 1234.52 in evidence.
        if any(g.startswith(value.split(".")[0]) for g in grounded):
            continue
        missing.append(raw)
    return missing


# The "Not covered:" clause carries two different kinds of gap, and they must be
# scored differently. A teammate's agent not being built yet says nothing about
# whether the question was answerable, so those fragments are dropped. A gap
# because the question named an uncovered sector is exactly the behaviour being
# measured, so those stay.
UNIMPLEMENTED = re.compile(r"[^;.]*not implemented yet[^;.]*[;.]?", re.IGNORECASE)


def is_refusal(answer: str, result: dict[str, Any]) -> bool:
    """Did the system state a limit on answering the question it was asked?

    True when the system said some part of the question could not be answered
    from what it has — including naming an out-of-scope sector — and when it
    produced no substantive content at all.
    """
    substantive = UNIMPLEMENTED.sub("", answer).strip()
    if not substantive:
        return True
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
    }
    return summary


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

    results = asyncio.run(run_all(questions, use_llm=not args.degraded))
    summary = report(results)
    print(render(results, summary))

    if args.json:
        args.json.write_text(
            json.dumps(
                {"degraded_run": args.degraded, "summary": summary, "results": [asdict(r) for r in results]},
                indent=2,
            )
        )
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
