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

**Three shapes of load, and two endpoints** (added 2026-09-12; EVALUATION.md §11):

    --mode burst       every user sends one request at once (the default above)
    --mode sequential  one user, each question once, in order: the baseline
    --mode sustained   every user asks, waits for the answer, and asks again,
                       no sooner than --pace seconds after its last question,
                       for --duration seconds or --per-user questions
    --endpoint chat    POST /api/chat/stream instead of /api/query, timing the
                       first frame as well as the `done` frame
    --signed-in        SRS 3.4.2 says *authenticated* users. Each virtual user
                       gets a real account (`load-NNN@ceynex.dev`, created
                       through the user store, not the IP-limited signup route)
                       and a real token, and the accounts are deleted after.
    python -m eval.load_test --verdict load.json baseline.json
                       the pre-registered rule (EVALUATION.md §11) over two runs

The pace is what keeps a sustained run honest. A signed-in user's allowance is
30 queries or 45 chat turns a minute. At one question per `PACE_S` a user asks
at most 24 a minute, so a 429 in a sustained run means the limiter, or whose
allowance a request was counted against, is wrong. It does not mean the load
generator was careless.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
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

#: One question per user per 2.5 s: 24 a minute, under the 30 (query) and 45
#: (chat) a signed-in user is allowed (`config/api.yaml`).
PACE_S = 2.5

#: Above this ratio of p95 under load to the single-user p95, the rule reads the
#: increase as material, even inside budget. Written down before any run.
MATERIAL_RATIO = 1.5

#: The accounts a signed-in run creates, and deletes when it is done.
LOAD_ACCOUNT = "load-{:03d}@ceynex.dev"


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
    endpoint: str = "query"
    #: Chat only: when the first SSE frame arrived, which is what a reader sees.
    first_frame_ms: float | None = None
    #: Seconds from the start of the run to when this request was sent.
    sent_at_s: float = 0.0


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


def _headers(user: int, token: str | None) -> dict[str, str]:
    """A signed-in user is limited as `user:{email}`, so its address is moot,
    but it is sent anyway: an anonymous run and a signed-in one differ in the
    token and nothing else."""
    headers = {"X-Real-IP": _real_ip_for(user)}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# --- accounts, for a signed-in run -------------------------------------------


def create_accounts(count: int) -> list[tuple[int, str]]:
    """`count` real accounts and a real token for each, as `(user id, token)`.

    Through the user store, in-process, rather than `POST /api/auth/signup`:
    signup is limited per address, and 50 accounts from one address would test
    that limiter instead of the system. The token is the one `login` issues.
    """
    from ceynex.api import users
    from ceynex.api.auth import issue_token

    users.ensure_table()
    accounts = []
    for index in range(count):
        email = LOAD_ACCOUNT.format(index)
        user = users.get_by_email(email) or users.create_user(
            email, secrets.token_urlsafe(18), "researcher"
        )
        accounts.append((user.id, issue_token(user.email, user.role, user.token_epoch)))
    return accounts


def delete_accounts(accounts: list[tuple[int, str]]) -> None:
    """The accounts, and every row they own (history, conversations, usage)."""
    from ceynex.api import users

    for user_id, _ in accounts:
        users.delete_user(user_id)


# --- one request --------------------------------------------------------------


async def _ask_query(client: httpx.AsyncClient, user: int, q: dict[str, Any],
                     headers: dict[str, str]) -> Result:
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


async def _ask_chat(client: httpx.AsyncClient, user: int, q: dict[str, Any],
                    headers: dict[str, str]) -> Result:
    """One stateless chat turn: POST /api/chat/stream, read to the `done` frame.

    A turn that ends without `done`, or whose `done` says it failed, is a failure
    even under a 200. SSE commits the status with its first byte, so a failure
    after that arrives in-band (D12).
    """
    start = time.perf_counter()
    first_frame_ms: float | None = None
    event, data_lines, done = None, [], None
    try:
        async with client.stream("POST", "/api/chat/stream", json={"query": q["question"]},
                                 headers=headers) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return Result(
                    user, q["id"], q["category"], q["question"], resp.status_code,
                    (time.perf_counter() - start) * 1000, False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}", endpoint="chat",
                )
            async for line in resp.aiter_lines():
                if line.startswith(":"):
                    continue  # a heartbeat
                if first_frame_ms is None and line:
                    first_frame_ms = (time.perf_counter() - start) * 1000
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
                elif not line and event is not None:
                    if event == "done":
                        done = json.loads("\n".join(data_lines) or "{}")
                        break
                    event, data_lines = None, []
        elapsed_ms = (time.perf_counter() - start) * 1000
        if done is None:
            return Result(user, q["id"], q["category"], q["question"], 200, elapsed_ms, False,
                          "stream ended without a done frame", "chat", first_frame_ms)
        if done.get("failed") or done.get("cancelled"):
            return Result(user, q["id"], q["category"], q["question"], 200, elapsed_ms, False,
                          f"turn ended: {json.dumps(done)[:200]}", "chat", first_frame_ms)
        answer = done.get("answer") or {}
        return Result(user, q["id"], q["category"], q["question"], 200, elapsed_ms,
                      bool(answer.get("degraded")), None, "chat", first_frame_ms)
    except Exception as exc:  # noqa: BLE001 - a load test records every failure mode, it doesn't triage one
        return Result(user, q["id"], q["category"], q["question"], None,
                      (time.perf_counter() - start) * 1000, False,
                      f"{type(exc).__name__}: {exc}", "chat", first_frame_ms)


async def _ask(client: httpx.AsyncClient, user: int, q: dict[str, Any], *, endpoint: str,
               token: str | None, run_start: float) -> Result:
    sent_at = time.perf_counter() - run_start
    headers = _headers(user, token)
    ask = _ask_chat if endpoint == "chat" else _ask_query
    result = await ask(client, user, q, headers)
    result.sent_at_s = round(sent_at, 3)
    return result


# --- the three shapes of load ---------------------------------------------------


def _client(base_url: str, users: int, timeout_s: float) -> httpx.AsyncClient:
    # One pool sized for every virtual user at once -- the point is that all
    # `users` requests are in flight together, not queued behind the client's
    # own connection limit before the server ever sees them.
    limits = httpx.Limits(max_connections=users + 10, max_keepalive_connections=users + 10)
    return httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits)


async def run(base_url: str, users: int, timeout_s: float, *, endpoint: str = "query",
              tokens: list[str | None] | None = None) -> tuple[list[Result], float]:
    """Burst: every user sends one request at once."""
    questions = _load_questions()
    assignments = [questions[i % len(questions)] for i in range(users)]
    tokens = tokens or [None] * users
    async with _client(base_url, users, timeout_s) as client:
        wall_start = time.perf_counter()
        results = list(
            await asyncio.gather(*(
                _ask(client, i, q, endpoint=endpoint, token=tokens[i], run_start=wall_start)
                for i, q in enumerate(assignments)
            ))
        )
        wall_s = time.perf_counter() - wall_start
    return results, wall_s


async def run_sequential(base_url: str, timeout_s: float, *, endpoint: str = "query",
                         token: str | None = None) -> tuple[list[Result], float]:
    """The baseline: one user, each question once, in order, in the same session."""
    questions = _load_questions()
    async with _client(base_url, 1, timeout_s) as client:
        wall_start = time.perf_counter()
        results = [
            await _ask(client, 0, q, endpoint=endpoint, token=token, run_start=wall_start)
            for q in questions
        ]
        wall_s = time.perf_counter() - wall_start
    return results, wall_s


async def run_sustained(base_url: str, users: int, timeout_s: float, *, endpoint: str = "query",
                        tokens: list[str | None] | None = None, pace_s: float = PACE_S,
                        per_user: int | None = None, duration_s: float | None = None,
                        questions: list[dict[str, Any]] | None = None,
                        ) -> tuple[list[Result], float]:
    """Sustained: `users` readers, each asking, waiting, and asking again.

    A user never has two questions in flight, and never sends one sooner than
    `pace_s` after its last. Starts are spread across one pace interval, so the
    run ramps up rather than opening with 50 identical requests in one tick.
    Ends after `per_user` questions each or `duration_s` seconds, whichever the
    caller gave.
    """
    if per_user is None and duration_s is None:
        raise ValueError("a sustained run needs --per-user or --duration")
    questions = questions or _load_questions()
    tokens = tokens or [None] * users

    async with _client(base_url, users, timeout_s) as client:
        wall_start = time.perf_counter()

        async def reader(user: int) -> list[Result]:
            out: list[Result] = []
            offset = user * pace_s / users
            k = 0
            while per_user is None or k < per_user:
                due = wall_start + offset + k * pace_s
                if duration_s is not None and due - wall_start >= duration_s:
                    break
                wait = due - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                q = questions[(user + k) % len(questions)]
                out.append(await _ask(client, user, q, endpoint=endpoint, token=tokens[user],
                                      run_start=wall_start))
                k += 1
            return out

        per_reader = await asyncio.gather(*(reader(u) for u in range(users)))
        wall_s = time.perf_counter() - wall_start
    return [r for rs in per_reader for r in rs], wall_s


# --- reading the results --------------------------------------------------------


def summarize(results: list[Result], wall_s: float, *, mode: str = "burst",
              endpoint: str = "query", signed_in: bool = False,
              pace_s: float | None = None) -> dict[str, Any]:
    succeeded = [r for r in results if r.status == 200 and r.error is None]
    rate_limited = [r for r in results if r.status == 429]
    failed = [r for r in results if not (r.status == 200 and r.error is None)]

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

    users = len({r.user for r in results})
    summary: dict[str, Any] = {
        "mode": mode,
        "endpoint": endpoint,
        "signed_in": signed_in,
        "users": users,
        "requests": len(results),
        "wall_clock_s": round(wall_s, 2),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "rate_limited": len(rate_limited),
        "availability": round(len(succeeded) / len(results), 4) if results else None,
        "degraded_answers": sum(1 for r in succeeded if r.degraded),
        "sum_of_individual_latency_s": round(sum(r.elapsed_ms for r in succeeded) / 1000, 2),
        "throughput_per_min": round(len(results) / wall_s * 60, 1) if wall_s else None,
        "latency_by_category": by_category,
        "errors": [
            {"user": r.user, "id": r.id, "status": r.status, "error": r.error}
            for r in failed
        ],
    }
    if pace_s is not None and mode == "sustained":
        # The most a run could send if every answer came back instantly.
        summary["offered_max_per_min"] = round(users * 60 / pace_s, 1)
    first_frames = sorted(r.first_frame_ms for r in succeeded if r.first_frame_ms is not None)
    if first_frames:
        summary["first_frame_ms"] = {
            "p50": round(_percentile(first_frames, 50), 1),
            "p95": round(_percentile(first_frames, 95), 1),
            "max": round(first_frames[-1], 1),
        }
    return summary


def verdict(load: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The rule written before the sustained run (EVALUATION.md §11).

    It passes when the run under load had no failures and no 429s, and each
    category's p95 kept its SRS 3.4.1 budget. The ratio to the same session's
    single-user p95 is reported, not scored. Above `MATERIAL_RATIO` it reads as a
    material increase. If the baseline itself breaks a budget, that category
    says so, because concurrency cannot be blamed for a breach already there at
    one user.
    """
    checks: dict[str, bool] = {
        "no_failures": load["failed"] == 0,
        "no_rate_limits": load["rate_limited"] == 0,
    }
    categories: dict[str, Any] = {}
    for category, stats in sorted(load["latency_by_category"].items()):
        budget = BUDGETS_MS.get(category)
        if budget is None:
            continue
        entry: dict[str, Any] = {
            "p95_ms": stats["p95_ms"],
            "budget_ms": budget,
            "within_budget": stats["p95_ms"] <= budget,
        }
        base = baseline.get("latency_by_category", {}).get(category)
        if base:
            entry["baseline_p95_ms"] = base["p95_ms"]
            entry["breached_at_one_user"] = base["p95_ms"] > budget
            if base["p95_ms"]:
                entry["ratio"] = round(stats["p95_ms"] / base["p95_ms"], 2)
                entry["material_increase"] = entry["ratio"] > MATERIAL_RATIO
        categories[category] = entry
        checks[f"{category}_p95_within_budget"] = entry["within_budget"]
    return {"passed": all(checks.values()), "checks": checks, "categories": categories}


def render(results: list[Result], summary: dict[str, Any]) -> str:
    lines = ["", "=" * 78, "CeyNex concurrency load test (SRS 3.4.2)", "=" * 78, ""]
    lines.append(f"{'user':>5s} {'id':5s} {'category':14s} {'status':>7s} {'ms':>8s}  question")
    lines.append("-" * 78)
    for r in sorted(results, key=lambda r: (r.user, r.sent_at_s)):
        status = str(r.status) if r.status else "ERR"
        lines.append(
            f"{r.user:5d} {r.id:5s} {r.category:14s} {status:>7s} {r.elapsed_ms:8.1f}  {r.question[:40]}"
        )
    lines += ["", "-" * 78, json.dumps(summary, indent=2), ""]

    if summary["errors"]:
        lines.append(f"{len(summary['errors'])} request(s) did not succeed — see errors[] above.")
    if summary["rate_limited"]:
        lines.append(
            f"{summary['rate_limited']} request(s) hit 429 despite distinct identities — "
            "check _real_ip_for()'s range, or the pacing, before trusting this run."
        )
    availability_line = (
        f"availability: {summary['availability']:.1%} of {summary['requests']} requests "
        f"from {summary['users']} users"
        if summary["availability"] is not None
        else "availability: no requests completed"
    )
    lines.append(availability_line)
    lines.append(
        f"wall clock {summary['wall_clock_s']}s for {summary['requests']} requests "
        f"whose individual latencies sum to {summary['sum_of_individual_latency_s']}s "
        "-- the gap is what concurrency bought."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=50, help="SRS 3.4.2's own number")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request client timeout, seconds")
    parser.add_argument("--json", type=Path, default=None, help="write the summary dict here too")
    parser.add_argument("--mode", choices=("burst", "sequential", "sustained"), default="burst")
    parser.add_argument("--endpoint", choices=("query", "chat"), default="query")
    parser.add_argument("--signed-in", action="store_true",
                        help="real accounts and tokens, deleted after the run")
    parser.add_argument("--pace", type=float, default=PACE_S,
                        help="sustained: seconds between one user's questions")
    parser.add_argument("--per-user", type=int, default=None, help="sustained: questions per user")
    parser.add_argument("--duration", type=float, default=None, help="sustained: seconds to run")
    parser.add_argument("--verdict", nargs=2, type=Path, metavar=("LOAD_JSON", "BASELINE_JSON"),
                        help="apply the pre-registered rule to two earlier runs and exit")
    args = parser.parse_args(argv)

    if args.verdict:
        load, baseline = (json.loads(p.read_text(encoding="utf-8"))["summary"] for p in args.verdict)
        outcome = verdict(load, baseline)
        print(json.dumps(outcome, indent=2))
        return 0 if outcome["passed"] else 1

    users = 1 if args.mode == "sequential" else args.users
    accounts = create_accounts(users) if args.signed_in else []
    tokens: list[str | None] = [token for _, token in accounts] or [None] * users
    try:
        if args.mode == "sequential":
            results, wall_s = asyncio.run(
                run_sequential(args.base_url, args.timeout, endpoint=args.endpoint, token=tokens[0])
            )
        elif args.mode == "sustained":
            results, wall_s = asyncio.run(run_sustained(
                args.base_url, users, args.timeout, endpoint=args.endpoint, tokens=tokens,
                pace_s=args.pace, per_user=args.per_user, duration_s=args.duration,
            ))
        else:
            results, wall_s = asyncio.run(
                run(args.base_url, users, args.timeout, endpoint=args.endpoint, tokens=tokens)
            )
    finally:
        if accounts:
            delete_accounts(accounts)

    summary = summarize(results, wall_s, mode=args.mode, endpoint=args.endpoint,
                        signed_in=args.signed_in, pace_s=args.pace)
    print(render(results, summary))

    if args.json:
        args.json.write_text(
            json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
            encoding="utf-8",
        )

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
