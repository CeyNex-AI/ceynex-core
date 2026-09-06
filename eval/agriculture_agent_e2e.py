"""Run the five M1 agriculture questions against real local services.

This is deliberately separate from the orchestrator's 30-question harness. It
tests the Agriculture & Commodity agent directly, using the live PostgreSQL
``fact_trade`` records, Neo4j graph, and registered local models. The LLM is
forced unavailable so the result is repeatable and exercises the required
degraded path without an API key.

    python -m eval.agriculture_agent_e2e
    python -m eval.agriculture_agent_e2e --out eval/results/m1_agriculture.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ceynex.agents.agriculture_commodity import AGENT, agriculture_commodity_node
from ceynex.agents.common import AgentDeps
from ceynex.contracts import new_state
from ceynex.kg.client import KnowledgeGraphClient
from ceynex.llm import FakeLLMClient


@dataclass(frozen=True)
class Check:
    """One question and the minimum behaviour that makes its result acceptable."""

    id: str
    question: str
    expected: str
    figure_key: str | None = None
    summary_fragment: str | None = None
    requires_forecast: bool = False


CHECKS = (
    Check(
        id="A01",
        question="What is the current price trend for cinnamon?",
        expected="A sourced cinnamon price trend with figures and evidence.",
        figure_key="latest_price",
    ),
    Check(
        id="A02",
        question="Will cinnamon prices rise or fall over the next two quarters?",
        expected="A target-matched cinnamon price forecast with an interval and annual-frequency caveat.",
        requires_forecast=True,
    ),
    Check(
        id="A03",
        question="Which district contributes the largest share of cinnamon exports?",
        expected="An honest refusal to name a largest district when no sourced numerical share exists.",
        summary_fragment="no sourced numerical district share",
    ),
    Check(
        id="A04",
        question="How have tea export volumes changed over the last five years?",
        expected="A sourced Tea Board export-volume trend with figures and evidence.",
        figure_key="latest_export_volume",
    ),
    Check(
        id="A05",
        question="If tea prices rise, what happens to demand for rubber?",
        expected="An honest refusal to infer an unsupported substitution effect.",
        summary_fragment="cannot be estimated responsibly",
    ),
)


def assess(check: Check, output: dict[str, Any]) -> list[str]:
    """Return failed acceptance checks; an empty list means the result passed."""
    failures: list[str] = []
    summary = str(output.get("summary", ""))
    evidence = list(output.get("evidence", []))

    if output.get("error"):
        failures.append(f"agent error: {output['error']}")
    if not summary:
        failures.append("missing summary")
    if len(evidence) < 2:
        failures.append(f"expected at least two evidence records, found {len(evidence)}")
    if check.figure_key and check.figure_key not in output.get("figures", {}):
        failures.append(f"missing figure: {check.figure_key}")
    if check.requires_forecast and not output.get("forecast"):
        failures.append("missing forecast")
    if check.summary_fragment and check.summary_fragment not in summary.lower():
        failures.append(f"missing required limitation: {check.summary_fragment!r}")
    return failures


def serialise(check: Check, output: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    """Keep exactly the review fields needed for a reproducible evaluation record."""
    return {
        "id": check.id,
        "question": check.question,
        "expected": check.expected,
        "passed": not failures,
        "failures": failures,
        "summary": output.get("summary", ""),
        "figures": output.get("figures", {}),
        "forecast": output.get("forecast", []),
        "evidence_count": len(output.get("evidence", [])),
        "confidence": output.get("confidence", 0.0),
        "degraded": bool(output.get("degraded", False)),
        "assumptions": output.get("assumptions", []),
        "error": output.get("error"),
    }


async def run_checks() -> list[dict[str, Any]]:
    """Run all checks against live PostgreSQL, Neo4j, and registered models."""
    records: list[dict[str, Any]] = []
    async with KnowledgeGraphClient() as kg:
        deps = AgentDeps(kg=kg, llm=FakeLLMClient(available=False))
        for check in CHECKS:
            patch = await agriculture_commodity_node(new_state(check.question, "agriculture-e2e"), deps)
            output = dict(patch.get("agent_outputs", {}).get(AGENT, {}))
            failures = assess(check, output)
            records.append(serialise(check, output, failures))
    return records


def report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a small, denominator-preserving summary for the report."""
    return {
        "questions": len(records),
        "passed": sum(bool(record["passed"]) for record in records),
        "failed": sum(not bool(record["passed"]) for record in records),
        "degraded": sum(bool(record["degraded"]) for record in records),
        "mean_evidence_count": round(
            sum(float(record["evidence_count"]) for record in records) / len(records), 2
        )
        if records
        else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M1 agriculture agent checks against local services.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/results/agriculture_agent_e2e.json"),
        help="Local JSON record path; eval/results is gitignored.",
    )
    args = parser.parse_args(argv)

    records = asyncio.run(run_checks())
    payload = {"summary": report(records), "results": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    print(json.dumps(payload["summary"], indent=2))
    for record in records:
        status = "PASS" if record["passed"] else "FAIL"
        print(f"{status} {record['id']}: {record['question']}")
        for failure in record["failures"]:
            print(f"  - {failure}")
    print(f"Results written to {args.out}")
    return 0 if payload["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
