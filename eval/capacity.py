"""SRS PR-01 / SAD C6: does moving the merge role to gpt-4o-mini bring the
single-sector p95 inside 10 s without costing answer quality?

Pre-registered in docs/EVALUATION.md §15 and committed before any run. Two
steps:

    python -m eval.capacity arm-b-config /tmp/capacity-b-config
    # arm A, today's config:                make eval-repeat (3 cold runs) -> eval_runs/capacity-a
    # arm B, merge on gpt-4o-mini:          CEYNEX_CONFIG_DIR=/tmp/capacity-b-config ... -> eval_runs/capacity-b
    python -m eval.capacity verdict eval_runs/capacity-a eval_runs/capacity-b

The rule, as registered, adopts arm B only if all four hold:

1. pooled single-sector p95 of B <= 10,000 ms, over every single-sector answer
   in its runs (36 samples from 3 runs), not a per-run p95 (§8: a one-run p95
   moves 35% with no code change);
2. B's mean fully grounded answers per run >= A's minus 1 (§8's noise floor);
3. no answerable question in any B run comes back with zero evidence;
4. B's mean ungrounded figures per run <= A's plus 1.

Routing is a control here, not an outcome: the router is its own role and does
not change between arms, so a routing difference means something else moved.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from ceynex import settings

BUDGET_MS = 10_000
NOISE_FLOOR = 1  # questions, from EVALUATION.md §8

MINI = {"model": "gpt-4o-mini", "cost_per_1k_input_tokens": "0.00015", "cost_per_1k_output_tokens": "0.0006"}


def write_arm_b_config(out: Path) -> Path:
    """Today's config with the merge role, and only the merge role, on gpt-4o-mini.

    Edited as text so every comment and every other role stays byte-identical.
    The cost rates move with the model, or spend accounting would count a mini
    call at gpt-4o's price, about 16x too high.
    """
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(settings.config_dir(), out)
    llm = out / "llm.yaml"
    lines = llm.read_text(encoding="utf-8").splitlines(keepends=True)
    in_merge = False
    changed = 0
    for i, line in enumerate(lines):
        if line.startswith("  merge:"):
            in_merge = True
            continue
        if in_merge and line.startswith("  ") and not line.startswith("    "):
            break  # the next role
        if in_merge:
            for key, value in MINI.items():
                if line.strip().startswith(f"{key}:"):
                    lines[i] = f"    {key}: {value}\n"
                    changed += 1
    if changed != len(MINI):
        raise RuntimeError(f"expected to change {len(MINI)} merge fields, changed {changed}")
    llm.write_text("".join(lines), encoding="utf-8")
    return out


def _runs(directory: Path) -> list[list[dict[str, Any]]]:
    files = sorted(Path(directory).glob("run-*.json"))
    if not files:
        raise FileNotFoundError(f"no run-*.json in {directory}")
    return [json.loads(f.read_text(encoding="utf-8"))["results"] for f in files]


def _p95(values: list[float]) -> float:
    """Nearest rank, as eval/harness.py and eval/load_test.py compute it."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(0.95 * len(ordered) + 0.5)))
    return ordered[rank - 1]


def measure(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    single = [r["elapsed_ms"] for run in runs for r in run if r["category"] == "single_sector"]
    answerable = [[r for r in run if r["answerable"]] for run in runs]
    return {
        "runs": len(runs),
        "single_sector_samples": len(single),
        "single_sector_p95_ms": round(_p95(single), 1),
        "fully_grounded_per_run": mean(sum(not r["ungrounded_figures"] for r in run) for run in answerable),
        "ungrounded_figures_per_run": mean(
            sum(len(r["ungrounded_figures"]) for r in run) for run in answerable
        ),
        "without_evidence_total": sum(r["evidence_count"] == 0 for run in answerable for r in run),
        "routing_exact_per_run": mean(
            sum(sorted(r["actual_route"]) == sorted(r["expected_route"]) for r in run) for run in runs
        ),
    }


def verdict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "p95_within_budget": b["single_sector_p95_ms"] <= BUDGET_MS,
        "grounding_held": b["fully_grounded_per_run"] >= a["fully_grounded_per_run"] - NOISE_FLOOR,
        "no_answer_without_evidence": b["without_evidence_total"] == 0,
        "ungrounded_held": b["ungrounded_figures_per_run"] <= a["ungrounded_figures_per_run"] + NOISE_FLOOR,
    }
    return {
        "adopt_b": all(checks.values()),
        "checks": checks,
        "a_already_within_budget": a["single_sector_p95_ms"] <= BUDGET_MS,
        "arm_a": a,
        "arm_b": b,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    config = sub.add_parser("arm-b-config", help="write arm B's config directory")
    config.add_argument("out", type=Path)
    judge = sub.add_parser("verdict", help="apply the registered rule to two arms' runs")
    judge.add_argument("arm_a", type=Path)
    judge.add_argument("arm_b", type=Path)
    args = parser.parse_args(argv)

    if args.command == "arm-b-config":
        print(write_arm_b_config(args.out))
        return 0
    outcome = verdict(measure(_runs(args.arm_a)), measure(_runs(args.arm_b)))
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["adopt_b"] else 1


if __name__ == "__main__":
    sys.exit(main())
