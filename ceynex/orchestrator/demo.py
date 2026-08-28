"""CLI over the orchestration graph — routing, merged answer, confidence, evidence.

    python -m ceynex.orchestrator.demo "which district exports the most cinnamon?"
    python -m ceynex.orchestrator.demo --no-llm "..."     # force degraded mode
    python -m ceynex.orchestrator.demo --json "..."

Exists so the orchestrator can be exercised without the API and without a
browser. The demo segment for the mid-evaluation runs from here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from typing import Any

from ceynex.agents.common import AgentDeps
from ceynex.contracts import new_state
from ceynex.kg.client import KnowledgeGraphClient
from ceynex.llm import FakeLLMClient, LLMReasoningClient
from ceynex.orchestrator.graph import build_graph
from ceynex.orchestrator.merger import agents_used_from_outputs, unanswered_from_outputs

log = logging.getLogger(__name__)

RULE = "─" * 78


async def answer(query: str, *, use_llm: bool = True, user_id: str = "cli") -> dict[str, Any]:
    """Run one query through the graph and return the merged result."""
    llm = LLMReasoningClient() if use_llm else FakeLLMClient(available=False)

    async with KnowledgeGraphClient() as kg:
        deps = AgentDeps(kg=kg, llm=llm)
        graph = build_graph(deps, use_llm_router=use_llm and getattr(llm, "available", False))

        started = time.perf_counter()
        final = await graph.ainvoke(new_state(query, user_id))
        elapsed_ms = (time.perf_counter() - started) * 1000

    outputs = final.get("agent_outputs", {})
    return {
        "query": query,
        "answer": final.get("final_answer", ""),
        "confidence": final.get("final_confidence", 0.0),
        # The same helpers the API route uses, rather than recomputing here.
        # Recomputing is what let the API drift once already (see
        # merger.no_topic_recognized's docstring), and this path feeds
        # eval/harness.py — a metric computed from a second, slightly different
        # definition measures a system nobody ships.
        "agents_used": agents_used_from_outputs(final),
        "agents_failed": sorted(name for name, out in outputs.items() if out.get("error")),
        # What the system said it could not answer, structurally. `is_refusal`
        # reads this instead of grepping the prose for decline phrasing.
        "unanswered": unanswered_from_outputs(final),
        "route": final.get("route", []),
        "sectors": final.get("sectors", []),
        "evidence": final.get("merged_evidence", []),
        "forecast": _forecast_of(outputs),
        "degraded": final.get("degraded", False),
        "errors": final.get("errors", []),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def _forecast_of(outputs: dict[str, Any]) -> list[dict[str, Any]]:
    for name in ("forecast", *outputs):
        if outputs.get(name, {}).get("forecast"):
            return list(outputs[name]["forecast"])
    return []


def render(result: dict[str, Any]) -> str:
    lines = [
        RULE,
        f"  {result['query']}",
        RULE,
        "",
        f"  ROUTED TO   {', '.join(result['route']) or '(none)'}",
        f"  SECTORS     {', '.join(result['sectors']) or '(none)'}",
        "",
        "  ANSWER",
    ]
    lines += [f"    {line}" for line in _wrap(result["answer"] or "(no answer)", 72)]

    lines += [
        "",
        f"  CONFIDENCE  {result['confidence']:.2f}"
        f"{'   (degraded — no LLM prose, figures only)' if result['degraded'] else ''}",
        f"  AGENTS      {', '.join(result['agents_used']) or '(none succeeded)'}",
    ]
    if result["agents_failed"]:
        lines.append(f"  UNANSWERED  {', '.join(result['agents_failed'])}")

    if result["forecast"]:
        lines += ["", "  FORECAST (80% interval)"]
        for point in result["forecast"]:
            lines.append(
                f"    {point['period']}  {point['point']:>16,.0f}"
                f"  [{point['lower']:>15,.0f} .. {point['upper']:>15,.0f}] {point['unit']}"
            )

    lines += ["", f"  EVIDENCE ({len(result['evidence'])})"]
    for item in result["evidence"]:
        lines.append(f"    [{item.get('source_id', '?')}] {item.get('claim', '')}")
        detail = item.get("detail", "")
        if detail:
            lines.append(f"        {detail[:100]}{'...' if len(detail) > 100 else ''}")

    lines += ["", f"  {result['elapsed_ms']:.0f} ms", RULE]
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask CeyNex a question from the terminal.")
    parser.add_argument("query", nargs="+", help="the question, in plain English")
    parser.add_argument("--no-llm", action="store_true", help="force degraded mode (SRS 3.4.3)")
    parser.add_argument("--json", action="store_true", help="emit the raw result")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = asyncio.run(answer(" ".join(args.query), use_llm=not args.no_llm))
    print(json.dumps(result, indent=2, default=str) if args.json else render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
