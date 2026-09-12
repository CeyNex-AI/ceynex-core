"""The multi-turn evaluation — what the conversational layer does, measured.

    python -m eval.chat_harness                    # every conversation, LLM on
    python -m eval.chat_harness --degraded         # keyword classifier, template gate
    python -m eval.chat_harness --id C02 --json out.json

`eval/harness.py` measures one question answered once, and `docs/EVALUATION.md`
§8 shows the conversational layer left that path unchanged. This measures the
layer itself: every conversation in `eval/conversations.yaml` is driven through
`api/turn_runner.py` exactly as `POST /api/chat/stream` drives it — the same
classification, the same clarification gate, the same discuss and analyse
paths, the same persistence — in-process rather than over HTTP, reading the
frames each turn wrote to its log. What is scored:

**Classifier accuracy** — did a follow-up take the path written down for it,
`discuss` (no fan-out) or `analyse` (a rewritten standalone query and a full
graph run)? Both directions matter: a discussion mistaken for a new analysis
costs a five-agent fan-out, a new analysis mistaken for a discussion answers
from stale evidence.

**Rewrite fidelity** — an `analyse` follow-up's standalone query must carry the
things the reader named ("rubber", "the United Kingdom", "10%").

**Discuss grounding** — a discussion's prose must not be withdrawn for stating
a figure the analysis never produced.

**Gate precision** — the clarification gate fires where the set says it should
and nowhere else.

**Cost and shape** — frames, elapsed time and model spend per turn, by mode.
The "3 frames against 22" figure quoted in D13 was measured before discuss
turns had a trace; this is where the current number comes from.

Needs the docker stack: the store is real (a conversation is created for the
run and deleted after it), and so is the graph. Tests never hit the network,
so the scoring is pure and tested over scripted frame logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

CONVERSATIONS = Path(__file__).parent / "conversations.yaml"
EVAL_USER = "eval@ceynex.dev"

MODES = ("discuss", "analyse")
EXPECT_KEYS = {"mode", "standalone_contains", "route", "clarify", "grounded", "refused_ok", "answer"}


@dataclass
class TurnResult:
    """One turn's outcome against what was expected of it."""

    conversation: str
    index: int
    query: str
    expect: dict[str, Any]
    #: What the runner decided: "discuss", "analyse", "clarify", or "first" for
    #: a first turn (no classification) — plus "failed"/"cancelled".
    mode: str
    method: str | None
    standalone_query: str | None
    route: list[str]
    clarified: bool
    grounded: bool | None
    degraded: bool
    #: A decline: no prose, or no evidence at all — the shape of the
    #: out-of-scope and no-data refusals. A partial answer that states a limit
    #: (SAD §4.1) is *not* a refusal and is scored as answered. Confidence is
    #: recorded but deliberately not used here: on a stack missing a series, a
    #: co-routed agent's decline drags a good answer's score under 0.10.
    refused: bool
    stated_limit: bool
    confidence: float | None
    evidence_count: int
    frames: int
    answer_deltas: int
    elapsed_ms: float
    cost_usd: float
    calls: int
    answer: str
    error: str | None = None
    #: Which expectations held. Empty when nothing was expected.
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def load_conversations(path: Path = CONVERSATIONS) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text())["conversations"]
    for conversation in data:
        for turn in conversation["turns"]:
            unknown = set(turn.get("expect", {})) - EXPECT_KEYS
            if unknown:
                raise ValueError(f"{conversation['id']}: unknown expectation keys {sorted(unknown)}")
            mode = turn.get("expect", {}).get("mode")
            if mode is not None and mode not in MODES:
                raise ValueError(f"{conversation['id']}: mode must be one of {MODES}, not {mode!r}")
    return data


# --- scoring (pure) ------------------------------------------------------------


def score_turn(
    conversation: str,
    index: int,
    turn: dict[str, Any],
    frames: list[dict[str, Any]],
    elapsed_ms: float,
) -> TurnResult:
    """Read a turn's frame log — `[{"event": ..., "data": {...}}, ...]` — and
    score it against `turn["expect"]`. Pure, so it is tested without a graph."""
    expect = dict(turn.get("expect", {}))
    by_event: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_event.setdefault(frame["event"], []).append(frame["data"])

    done = (by_event.get("done") or [{}])[-1]
    turn_frame = (by_event.get("turn") or [None])[-1]
    answer = done.get("answer") or {}
    clarified = bool(done.get("clarify"))
    usage = done.get("usage") or {}

    if done.get("cancelled"):
        mode = "cancelled"
    elif done.get("failed"):
        mode = "failed"
    elif clarified:
        mode = "clarify"
    elif turn_frame is not None:
        mode = str(turn_frame.get("mode"))
    else:
        mode = "first"

    route = list(answer.get("route") or [])
    unanswered = list(answer.get("unanswered") or [])
    text = str(answer.get("answer") or "")
    confidence = answer.get("confidence")
    evidence_count = len(answer.get("evidence") or [])
    refused = mode not in ("clarify", "failed", "cancelled") and (
        not text.strip() or evidence_count == 0
    )

    checks: dict[str, bool] = {}
    expected_mode = expect.get("mode")
    if expected_mode is not None:
        # A first turn is always an analysis; the runner emits no `turn` frame.
        actual = "analyse" if mode == "first" else mode
        checks["mode"] = actual == expected_mode
    if "clarify" in expect:
        checks["clarify"] = clarified == bool(expect["clarify"])
    elif index > 0 or expect.get("mode") is not None:
        # Unless the set says a question should be asked, it should not be.
        checks["no_clarify"] = not clarified
    if expect.get("standalone_contains"):
        rewritten = (turn_frame or {}).get("standalone_query") or ""
        lowered = rewritten.lower()
        checks["standalone_contains"] = all(
            str(needle).lower() in lowered for needle in expect["standalone_contains"]
        )
    if expect.get("route"):
        checks["route"] = set(expect["route"]) <= set(route)
    if expect.get("grounded"):
        checks["grounded"] = answer.get("grounded") is not False and not bool(
            by_event.get("answer_reset")
        )
    if mode in ("analyse", "first") and not expect.get("refused_ok"):
        checks["answered"] = not refused

    return TurnResult(
        conversation=conversation,
        index=index,
        query=turn["query"],
        expect=expect,
        mode=mode,
        method=(turn_frame or {}).get("method"),
        standalone_query=(turn_frame or {}).get("standalone_query"),
        route=route,
        clarified=clarified,
        grounded=answer.get("grounded") if "grounded" in answer else None,
        degraded=bool(answer.get("degraded", False)),
        refused=refused,
        stated_limit=bool(unanswered),
        confidence=float(confidence) if confidence is not None else None,
        evidence_count=evidence_count,
        frames=len(frames),
        answer_deltas=len(by_event.get("answer_delta", [])),
        elapsed_ms=round(elapsed_ms, 1),
        cost_usd=float(usage.get("cost_usd", 0.0) or 0.0),
        calls=int(usage.get("calls", 0) or 0),
        answer=text,
        error=(by_event.get("error") or [{}])[-1].get("message") if by_event.get("error") else None,
        checks=checks,
    )


def report(results: list[TurnResult]) -> dict[str, Any]:
    """Aggregate. Every rate carries its denominator."""
    classified = [r for r in results if "mode" in r.checks and r.index > 0]
    rewrites = [r for r in results if "standalone_contains" in r.checks]
    discussions = [r for r in results if "grounded" in r.checks]
    gate_expected = [r for r in results if r.expect.get("clarify")]
    gate_silent = [r for r in results if "no_clarify" in r.checks]

    def rate(flags: list[bool]) -> dict[str, Any]:
        return {"rate": round(sum(flags) / len(flags), 4) if flags else None, "of": len(flags)}

    by_mode: dict[str, Any] = {}
    for mode in ("first", "analyse", "discuss", "clarify"):
        turns = [r for r in results if r.mode == mode]
        if not turns:
            continue
        elapsed = sorted(r.elapsed_ms for r in turns)
        by_mode[mode] = {
            "turns": len(turns),
            "frames_median": statistics.median(r.frames for r in turns),
            "answer_deltas_median": statistics.median(r.answer_deltas for r in turns),
            "elapsed_ms_p50": round(statistics.median(elapsed), 1),
            "elapsed_ms_max": round(elapsed[-1], 1),
            "cost_usd_mean": round(statistics.fmean(r.cost_usd for r in turns), 5),
            "calls_mean": round(statistics.fmean(r.calls for r in turns), 2),
        }

    return {
        "conversations": len({r.conversation for r in results}),
        "turns": len(results),
        "failed_or_cancelled": sum(1 for r in results if r.mode in ("failed", "cancelled")),
        "errors": sum(1 for r in results if r.error),
        "classifier": {
            "mode_accuracy": rate([r.checks["mode"] for r in classified]),
            "rewrite_fidelity": rate([r.checks["standalone_contains"] for r in rewrites]),
        },
        "discuss": {
            "grounded": rate([r.checks["grounded"] for r in discussions]),
        },
        "gate": {
            "asked_where_expected": rate([r.checks["clarify"] for r in gate_expected]),
            "silent_elsewhere": rate([r.checks["no_clarify"] for r in gate_silent]),
        },
        "turns_passing_every_check": rate([r.passed for r in results]),
        "degraded_turns": sum(1 for r in results if r.degraded),
        # Informational, as in the one-shot harness: stating a limit while still
        # answering is the behaviour SAD §4.1 asks for.
        "analyses_that_stated_some_limit": rate(
            [r.stated_limit for r in results if r.mode in ("first", "analyse")]
        ),
        "by_mode": by_mode,
    }


def render(results: list[TurnResult], summary: dict[str, Any]) -> str:
    lines = ["", "=" * 78, "CeyNex multi-turn evaluation", "=" * 78, ""]
    lines.append(f"{'conv':5s} {'#':>2s} {'mode':8s} {'ok':3s} {'frames':>6s} {'ms':>8s}  query")
    lines.append("-" * 78)
    for r in results:
        ok = "ok" if r.passed else "FAIL"
        lines.append(
            f"{r.conversation:5s} {r.index + 1:2d} {r.mode:8s} {ok:3s} {r.frames:6d} "
            f"{r.elapsed_ms:8.1f}  {r.query[:38]}"
        )
    lines += ["", "-" * 78, json.dumps(summary, indent=2), ""]
    failed = [r for r in results if not r.passed]
    if failed:
        lines.append("Turns that missed an expectation:")
        for r in failed:
            missed = sorted(k for k, v in r.checks.items() if not v)
            detail = f" standalone={r.standalone_query!r}" if "standalone_contains" in missed else ""
            lines.append(f"  {r.conversation} turn {r.index + 1}: {missed}{detail}")
        lines.append("")
    return "\n".join(lines)


# --- driving the runner ----------------------------------------------------------


async def _run_turn(runtime, query: str, conversation_id: int, *, skip_clarify: bool = False):
    from ceynex.api import turn_runner

    started = turn_runner.start_turn(
        turn_runner.TurnRequest(
            runtime=runtime, query=query, typed=query, user_email=EVAL_USER,
            conversation_id=conversation_id, skip_clarify=skip_clarify,
        )
    )
    begun = time.perf_counter()
    if started.task is not None:
        done, _ = await asyncio.wait({started.task}, timeout=turn_runner.STREAM_BUDGET_S + 15)
        if not done:
            started.task.cancel()
    elapsed_ms = (time.perf_counter() - begun) * 1000
    frames = [{"event": f.event, "data": f.data} for f in started.frames]
    return frames, elapsed_ms


async def run_conversation(conversation: dict[str, Any], runtime) -> list[TurnResult]:
    """Drive one conversation to its end. Never raises; a failure is a result."""
    from ceynex.chat import clarify, store

    conversation_id = await store.create(EVAL_USER, title=f"eval {conversation['id']}")
    results: list[TurnResult] = []
    try:
        for index, turn in enumerate(conversation["turns"]):
            try:
                frames, elapsed_ms = await _run_turn(runtime, turn["query"], conversation_id)
            except Exception as exc:  # noqa: BLE001 - a crash is a result
                log.warning("%s turn %d raised: %s", conversation["id"], index + 1, exc)
                frames, elapsed_ms = [{"event": "error", "data": {"message": str(exc)}},
                                      {"event": "done", "data": {"failed": True}}], 0.0
            result = score_turn(conversation["id"], index, turn, frames, elapsed_ms)
            results.append(result)
            log.info("%s %d %s %s (%.0f ms)", conversation["id"], index + 1, result.mode,
                     "ok" if result.passed else "FAIL", elapsed_ms)

            # A clarifying question the set expected: answer it the way the
            # resume route does, so the composed query and the gate-free path
            # are exercised too. `answer` names the option; default the first.
            # (Until 2026-09-12 the default was the last, which the template
            # made "both". "both" is no longer offered: see chat/clarify.py.)
            if result.clarified and turn.get("expect", {}).get("clarify"):
                pending = next((f["data"] for f in frames if f["event"] == "clarify"), None)
                if pending and pending.get("pending_id") is not None:
                    claimed = await store.resolve_clarification(int(pending["pending_id"]), EVAL_USER)
                    if claimed is not None:
                        options = list(pending.get("options") or [])
                        chosen = turn.get("expect", {}).get("answer") or (options[0] if options else "")
                        composed = clarify.Clarification.compose(
                            str(claimed["original_query"]), [chosen] if chosen else []
                        )
                        frames, elapsed_ms = await _run_turn(
                            runtime, composed, conversation_id, skip_clarify=True
                        )
                        follow = score_turn(
                            conversation["id"], index, {"query": composed, "expect": {"mode": "analyse"}},
                            frames, elapsed_ms,
                        )
                        follow.method = "clarify-resume"
                        results.append(follow)
    finally:
        try:
            await store.delete(conversation_id, EVAL_USER)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not delete eval conversation %s: %s", conversation_id, exc)
    return results


async def run_all(conversations: list[dict[str, Any]], *, use_llm: bool) -> list[TurnResult]:
    from ceynex.agents.common import AgentDeps
    from ceynex.api.deps import Runtime
    from ceynex.kg.client import KnowledgeGraphClient
    from ceynex.llm import FakeLLMClient, LLMReasoningClient
    from ceynex.orchestrator.graph import build_graph
    from ceynex.retrieval.client import PolicyRetriever

    llm = LLMReasoningClient() if use_llm else FakeLLMClient(available=False)
    policy = PolicyRetriever.from_settings()
    if policy is not None:
        await policy.warmup()

    results: list[TurnResult] = []
    async with KnowledgeGraphClient() as kg:
        deps = AgentDeps(kg=kg, llm=llm, extras={"policy": policy})
        graph = build_graph(deps, use_llm_router=use_llm and getattr(llm, "available", False))
        runtime = Runtime(kg=kg, llm=llm, deps=deps, graph=graph, policy=policy)
        for conversation in conversations:
            results.extend(await run_conversation(conversation, runtime))
    if policy is not None:
        await policy.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the CeyNex multi-turn evaluation set.")
    parser.add_argument("--conversations", type=Path, default=CONVERSATIONS)
    parser.add_argument("--id", help="run a single conversation by id")
    parser.add_argument("--degraded", action="store_true",
                        help="force the LLM unavailable: keyword classifier, template gate")
    parser.add_argument("--json", type=Path, help="write full results here")
    parser.add_argument("--cold", action="store_true",
                        help="clear the prompt cache first, so every model call is paid for")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cold:
        from eval.repeat import clear_prompt_cache

        print(f"cleared {clear_prompt_cache()} cached completions")
    for noisy in ("httpx", "httpcore", "neo4j", "ceynex.observability", "ceynex.api.turn_runner"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    conversations = load_conversations(args.conversations)
    if args.id:
        conversations = [c for c in conversations if c["id"] == args.id]
    if not conversations:
        print("no conversations matched", file=sys.stderr)
        return 2

    results = asyncio.run(run_all(conversations, use_llm=not args.degraded))
    summary = report(results)
    print(render(results, summary))

    if args.json:
        args.json.write_text(json.dumps(
            {"degraded_run": args.degraded, "summary": summary,
             "results": [asdict(r) for r in results]},
            indent=2,
        ))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
