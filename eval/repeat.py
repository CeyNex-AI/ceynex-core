"""Repeated runs of the evaluation set, and what they agree on.

`docs/EVALUATION.md` §8 measured the 30-question set's noise floor by running
identical code twice: routing exact match moved by one question, the
ungrounded-figure count by one, and single-sector p95 by 35%. A difference
between two single runs that sits inside that floor is not a result. This module
is the protocol that follows from it — run the set N times cold, report the
median and the spread, and list the questions that disagreed with themselves —
so that a comparison (a prompt change, a branch, a flag) is made on medians of
repeated runs rather than on one draw each.

    python -m eval.harness --repeat 3 --cold --json-dir eval_runs/off

`summarize_runs` takes the per-run summaries `harness.report()` produces and is
pure, so it is unit-tested without a model or a database.
"""

from __future__ import annotations

import json
import logging
import shutil
import statistics
from pathlib import Path
from typing import Any

from ceynex.settings import load_config

log = logging.getLogger(__name__)

#: Dotted paths into `harness.report()`'s summary. Rates are read from their
#: `{"rate": x, "of": n}` wrapper; everything else as the number it is.
HEADLINE_METRICS = (
    "routing.exact_match",
    "routing.recall",
    "routing.never_empty",
    "evidence.answers_fully_grounded",
    "evidence.ungrounded_figures_total",
    "evidence.mean_evidence_per_answer",
    "evidence.answers_with_no_evidence",
    "refusal.unanswerable_correctly_refused",
    "citations.answers_with_markers",
    "citations.marker_valid_rate",
    "citations.figure_sentences_cited_rate",
    "crashed",
    "degraded_answers",
)

LATENCY_CATEGORIES = ("single_sector", "cross_sector", "simulation")


def prompt_cache_path() -> Path:
    """Where `LLMReasoningClient` keeps its prompt cache, from `config/llm.yaml`."""
    cache_cfg = load_config("llm").get("cache", {})
    return Path(cache_cfg.get("path", ".cache/llm"))


def clear_prompt_cache(path: Path | None = None) -> int:
    """Remove every cached completion so the next run pays for each call.

    A warm cache gives the false 65 ms p50 EVALUATION.md §1 warns about, and it
    also hides run-to-run variance in the model's own output — which is the
    thing a repeated run exists to measure. Returns how many entries went.
    """
    target = path or prompt_cache_path()
    if not target.exists():
        return 0
    entries = [p for p in target.iterdir() if p.is_file()]
    for entry in entries:
        entry.unlink()
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    return len(entries)


def _lookup(summary: dict[str, Any], dotted: str) -> float | None:
    node: Any = summary
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, dict) and "rate" in node:
        node = node["rate"]
    if node is None or isinstance(node, bool):
        return None if node is None else float(node)
    if isinstance(node, int | float):
        return float(node)
    return None


def _spread(values: list[float]) -> dict[str, Any]:
    return {
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "values": [round(v, 4) for v in values],
        "runs": len(values),
    }


def summarize_runs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Median, min, max and every value, per headline metric, across runs.

    A metric absent from every run is omitted rather than reported as zero — a
    zero is a claim, and "not measured" is the true statement.
    """
    if not summaries:
        return {"runs": 0, "metrics": {}, "latency_ms": {}}

    metrics: dict[str, Any] = {}
    for dotted in HEADLINE_METRICS:
        values = [v for v in (_lookup(s, dotted) for s in summaries) if v is not None]
        if values:
            metrics[dotted] = _spread(values)

    latency: dict[str, Any] = {}
    for category in LATENCY_CATEGORIES:
        for pct in ("p50", "p95"):
            values = [
                v for v in (_lookup(s, f"latency_ms.{category}.{pct}") for s in summaries)
                if v is not None
            ]
            if values:
                latency.setdefault(category, {})[pct] = _spread(values)

    return {"runs": len(summaries), "metrics": metrics, "latency_ms": latency}


def disagreements(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Questions whose route or grounding differed between runs of the same code.

    Each entry of `runs` is one run's `results` list (dicts from `asdict(Result)`).
    A question is listed when its actual route was not the same set every time,
    or its ungrounded-figure count moved. These are the S07 class: real
    weaknesses that a single run either shows or hides by luck.
    """
    by_id: dict[str, list[dict[str, Any]]] = {}
    for results in runs:
        for result in results:
            by_id.setdefault(result["id"], []).append(result)

    out: list[dict[str, Any]] = []
    for qid, seen in by_id.items():
        routes = [sorted(r.get("actual_route", [])) for r in seen]
        ungrounded = [len(r.get("ungrounded_figures", [])) for r in seen]
        route_moved = any(route != routes[0] for route in routes)
        grounding_moved = any(count != ungrounded[0] for count in ungrounded)
        if route_moved or grounding_moved:
            out.append({
                "id": qid,
                "question": seen[0].get("question", ""),
                "routes": routes,
                "ungrounded_counts": ungrounded,
                "route_moved": route_moved,
                "grounding_moved": grounding_moved,
            })
    return out


def render(summary: dict[str, Any], moved: list[dict[str, Any]]) -> str:
    lines = ["", "=" * 78, f"CeyNex evaluation — {summary['runs']} repeated cold runs", "=" * 78, ""]
    lines.append(f"{'metric':44s} {'median':>8s} {'min':>8s} {'max':>8s}")
    lines.append("-" * 78)
    for name, spread in summary["metrics"].items():
        lines.append(
            f"{name:44s} {spread['median']:8.4f} {spread['min']:8.4f} {spread['max']:8.4f}"
        )
    for category, pcts in summary.get("latency_ms", {}).items():
        for pct, spread in pcts.items():
            name = f"latency_ms.{category}.{pct}"
            lines.append(
                f"{name:44s} {spread['median']:8.1f} {spread['min']:8.1f} {spread['max']:8.1f}"
            )
    lines.append("")
    if moved:
        lines.append("Questions that disagreed with themselves across runs:")
        for entry in moved:
            what = []
            if entry["route_moved"]:
                what.append(f"route {entry['routes']}")
            if entry["grounding_moved"]:
                what.append(f"ungrounded {entry['ungrounded_counts']}")
            lines.append(f"  {entry['id']}: {'; '.join(what)}")
        lines.append("")
    else:
        lines.append("Every question routed and grounded the same way in every run.")
        lines.append("")
    return "\n".join(lines)


def write_repeat_summary(json_dir: Path, summaries: list[dict[str, Any]],
                         runs: list[list[dict[str, Any]]], *, degraded: bool) -> Path:
    """Write `summary.json` beside the per-run files and return its path."""
    summary = summarize_runs(summaries)
    moved = disagreements(runs)
    out = json_dir / "summary.json"
    out.write_text(json.dumps(
        {"degraded_run": degraded, "repeat": summary, "disagreements": moved}, indent=2
    ))
    print(render(summary, moved))
    return out


__all__ = [
    "HEADLINE_METRICS",
    "clear_prompt_cache",
    "disagreements",
    "prompt_cache_path",
    "render",
    "summarize_runs",
    "write_repeat_summary",
]
