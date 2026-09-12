"""Implements SRS 3.4.2 — 50 concurrent users against a running API server.

    make up                                          # postgres/neo4j/qdrant
    .venv/Scripts/python -m uvicorn ceynex.api.main:app --host 127.0.0.1 --port 8000 &
    python -m eval.load_test --json load_results.json

Unlike `eval/harness.py`, which drives the orchestrator's own entry point
in-process to measure the machine (SRS 3.4.1's single-user latency, routing,
grounding), this drives real HTTP requests against a *running* server, because
SRS 3.4.2 is a claim about serving capacity under concurrency: the ASGI event
loop, the connection pools to Postgres/Neo4j, and the rate limiter, none of
which an in-process call exercises. `docs/DEFERRED.md` has flagged this
untested since the rate limiter shipped; this is that measurement.

**Each virtual user is a distinct rate-limit identity.** `POST /api/query`
rate-limits per client address when there is no signed-in user
(`ceynex/api/rate_limit.py::identity_of`), keyed off `X-Real-IP` — the header
nginx sets in front of the deployed backend (`client_ip()`'s own docstring).
Sending 50 concurrent requests from one real IP with no header would collapse
them into one caller's 30/minute allowance and measure the rate limiter
instead of the server; this assigns each virtual user its own `X-Real-IP` so
the 50 are actually 50 distinct callers, as SRS 3.4.2 intends.

**Questions repeat.** N users cycle through the real 30-question set
(`eval/questions.yaml`, the same one `eval/harness.py` uses), so several fire
the identical question concurrently — a realistic mix of cold and (if this
runs a second time inside the LLM cache's 168h TTL) warm prompt-cache hits,
not an artificially distinct workload.

**Cost.** Every question the LLM has not cached is a real OpenAI call, bounded
by `config/llm.yaml`'s `daily_spend_cap_usd` (5.0) — once tripped mid-run,
remaining calls degrade (SRS 3.4.3) rather than erroring, and this script
reports `degraded` per response rather than counting it as a failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

QUESTIONS = Path(__file__).parent / "questions.yaml"

# Same budgets eval/harness.py checks single-user latency against (SRS 3.4.1).
# SRS 3.4.2 doesn't set a separate per-request number under load — the
# requirement is that the system stays *available* at 50 concurrent users —
# but reporting against the same budgets shows whether concurrency itself
# degrades individual response times, not just whether requests eventually
# come back.
BUDGETS_MS = {"single_sector": 10_000, "cross_sector": 20_000, "simulation": 20_000}


@dataclass
class Result:
    user: int
    id: str
    category: str
    question: str
    status: int | None
    elapsed_ms: float
    degraded: bool
    error: str | None


def _load_questions() -> list[dict[str, Any]]:
    return yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"]


def _percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank percentile — same choice as eval/harness.py's, and for the
    same reason: an interpolated p95 on a few dozen samples is false precision.
    """
    if not ordered:
        return 0.0
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered) + 0.5)))
    return ordered[rank - 1]


def _real_ip_for(user: int) -> str:
    """A distinct, private-range address per virtual user — never a real one."""
    return f"10.50.{(user // 250) % 256}.{user % 250 + 1}"


async def _one_user(client: httpx.AsyncClient, user: int, q: dict[str, Any]) -> Result:
    headers = {"X-Real-IP": _real_ip_for(user)}
    start = time.perf_counter()
    try:
        resp = await client.post("/api/query", json={"query": q["question"]}, headers=headers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if resp.status_code != 200:
            return Result(
                user, q["id"], q["category"], q["question"], resp.status_code,
                elapsed_ms, False, f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        body = resp.json()
        return Result(
            user, q["id"], q["category"], q["question"], 200,
            elapsed_ms, bool(body.get("degraded")), None,
        )
    except Exception as exc:  # noqa: BLE001 - a load test records every failure mode, it doesn't triage one
        elapsed_ms = (time.perf_counter() - start) * 1000
        return Result(
            user, q["id"], q["category"], q["question"], None,
            elapsed_ms, False, f"{type(exc).__name__}: {exc}",
        )


async def run(base_url: str, users: int, timeout_s: float) -> tuple[list[Result], float]:
    questions = _load_questions()
    assignments = [questions[i % len(questions)] for i in range(users)]
    # One pool sized for every virtual user at once -- the point is that all
    # `users` requests are in flight together, not queued behind the client's
    # own connection limit before the server ever sees them.
    limits = httpx.Limits(max_connections=users + 10, max_keepalive_connections=users + 10)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits) as client:
        wall_start = time.perf_counter()
        results = list(
            await asyncio.gather(*(_one_user(client, i, q) for i, q in enumerate(assignments)))
        )
        wall_s = time.perf_counter() - wall_start
    return results, wall_s


def summarize(results: list[Result], wall_s: float) -> dict[str, Any]:
    succeeded = [r for r in results if r.status == 200]
    rate_limited = [r for r in results if r.status == 429]
    failed = [r for r in results if r.status != 200]

    by_category: dict[str, Any] = {}
    for category in sorted({r.category for r in results}):
        times = sorted(r.elapsed_ms for r in succeeded if r.category == category)
        if not times:
            continue
        budget = BUDGETS_MS.get(category)
        by_category[category] = {
            "n": len(times),
            "p50_ms": round(_percentile(times, 50), 1),
            "p95_ms": round(_percentile(times, 95), 1),
            "max_ms": round(times[-1], 1),
            "budget_ms": budget,
            "within_budget": all(t <= budget for t in times) if budget else None,
        }

    return {
        "users": len(results),
        "wall_clock_s": round(wall_s, 2),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "rate_limited": len(rate_limited),
        "availability": round(len(succeeded) / len(results), 4) if results else None,
        "degraded_answers": sum(1 for r in succeeded if r.degraded),
        "sum_of_individual_latency_s": round(sum(r.elapsed_ms for r in succeeded) / 1000, 2),
        "latency_by_category": by_category,
        "errors": [
            {"user": r.user, "id": r.id, "status": r.status, "error": r.error}
            for r in failed
        ],
    }


def render(results: list[Result], summary: dict[str, Any]) -> str:
    lines = ["", "=" * 78, "CeyNex concurrency load test (SRS 3.4.2)", "=" * 78, ""]
    lines.append(f"{'user':>5s} {'id':5s} {'category':14s} {'status':>7s} {'ms':>8s}  question")
    lines.append("-" * 78)
    for r in sorted(results, key=lambda r: r.user):
        status = str(r.status) if r.status else "ERR"
        lines.append(
            f"{r.user:5d} {r.id:5s} {r.category:14s} {status:>7s} {r.elapsed_ms:8.1f}  {r.question[:40]}"
        )
    lines += ["", "-" * 78, json.dumps(summary, indent=2), ""]

    if summary["errors"]:
        lines.append(f"{len(summary['errors'])} request(s) did not return 200 — see errors[] above.")
    if summary["rate_limited"]:
        lines.append(
            f"{summary['rate_limited']} request(s) hit 429 despite distinct X-Real-IP identities — "
            "check _real_ip_for()'s range doesn't collide before trusting this run."
        )
    availability_line = (
        f"availability: {summary['availability']:.1%} of {summary['users']} concurrent users"
        if summary["availability"] is not None
        else "availability: no requests completed"
    )
    lines.append(availability_line)
    lines.append(
        f"wall clock {summary['wall_clock_s']}s for {summary['users']} concurrent requests "
        f"whose individual latencies sum to {summary['sum_of_individual_latency_s']}s "
        "-- the gap is what concurrency bought."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=50, help="SRS 3.4.2's own number")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request client timeout, seconds")
    parser.add_argument("--json", type=Path, default=None, help="write the summary dict here too")
    args = parser.parse_args(argv)

    results, wall_s = asyncio.run(run(args.base_url, args.users, args.timeout))
    summary = summarize(results, wall_s)
    print(render(results, summary))

    if args.json:
        args.json.write_text(
            json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
            encoding="utf-8",
        )

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
