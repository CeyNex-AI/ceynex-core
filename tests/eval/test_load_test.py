"""The load harness's own logic, over a scripted server — never the network.

What a load test reports is only as good as how it reads responses and how it
paces requests, and both are testable without a server: the SSE reading, the
pacing, the accounting of failures and 429s, and the pre-registered verdict.
"""

from __future__ import annotations

import json

import httpx
import pytest

from eval import load_test

SSE_ANSWERED = (
    ": heartbeat\n\n"
    'id: 1\nevent: start\ndata: {"request_id": "r1"}\n\n'
    'id: 2\nevent: node_start\ndata: {"node": "route"}\n\n'
    'id: 3\nevent: done\ndata: {"failed": false, "answer": {"degraded": true, "answer": "x"}}\n\n'
)


def _serve(monkeypatch, handler):
    """Every client the harness opens talks to `handler` instead of a server."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        load_test, "_client",
        lambda base_url, users, timeout_s: httpx.AsyncClient(base_url=base_url, transport=transport),
    )


QUESTIONS = [
    {"id": "S01", "category": "single_sector", "question": "cinnamon price trend"},
    {"id": "X01", "category": "cross_sector", "question": "tea versus apparel"},
]


# --- reading a chat turn ---------------------------------------------------------


async def test_a_chat_turn_times_its_first_frame_and_reads_degraded_from_done(monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(200, text=SSE_ANSWERED))
    results, _ = await load_test.run_sustained("http://api", 1, 5.0, endpoint="chat",
                                               per_user=2, pace_s=0.0, questions=QUESTIONS)
    assert [r.status for r in results] == [200, 200]
    assert all(r.error is None and r.degraded and r.endpoint == "chat" for r in results)
    assert all(r.first_frame_ms is not None and r.first_frame_ms <= r.elapsed_ms for r in results)


async def test_a_turn_whose_done_says_failed_is_a_failure_under_a_200(monkeypatch):
    body = 'id: 1\nevent: error\ndata: {"message": "x"}\n\nid: 2\nevent: done\ndata: {"failed": true}\n\n'
    _serve(monkeypatch, lambda request: httpx.Response(200, text=body))
    results, wall = await load_test.run_sustained("http://api", 1, 5.0, endpoint="chat",
                                                  per_user=1, pace_s=0.0, questions=QUESTIONS)
    assert results[0].status == 200 and results[0].error
    assert load_test.summarize(results, wall)["failed"] == 1


async def test_a_stream_that_ends_without_done_is_a_failure(monkeypatch):
    body = 'id: 1\nevent: start\ndata: {}\n\n'
    _serve(monkeypatch, lambda request: httpx.Response(200, text=body))
    results, _ = await load_test.run_sustained("http://api", 1, 5.0, endpoint="chat",
                                               per_user=1, pace_s=0.0, questions=QUESTIONS)
    assert results[0].error == "stream ended without a done frame"


# --- pacing ----------------------------------------------------------------------


async def test_a_sustained_user_never_asks_sooner_than_the_pace(monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(200, json={"degraded": False}))
    pace = 0.05
    results, _ = await load_test.run_sustained("http://api", 3, 5.0, per_user=3, pace_s=pace,
                                               questions=QUESTIONS)
    assert len(results) == 9
    for user in range(3):
        sent = sorted(r.sent_at_s for r in results if r.user == user)
        assert all(b - a >= pace - 0.005 for a, b in zip(sent, sent[1:], strict=False)), sent


async def test_a_sustained_run_ends_at_its_duration(monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(200, json={"degraded": False}))
    results, wall = await load_test.run_sustained("http://api", 2, 5.0, duration_s=0.12,
                                                  pace_s=0.05, questions=QUESTIONS)
    assert 2 <= len(results) <= 6
    assert all(r.sent_at_s < 0.12 for r in results)


async def test_a_sustained_run_needs_a_bound():
    with pytest.raises(ValueError, match="per-user or --duration"):
        await load_test.run_sustained("http://api", 1, 5.0)


# --- identity ----------------------------------------------------------------------


def test_a_signed_in_user_sends_its_token_and_an_anonymous_one_does_not():
    assert load_test._headers(3, "tok") == {"X-Real-IP": "10.50.0.4", "Authorization": "Bearer tok"}
    assert "Authorization" not in load_test._headers(3, None)


async def test_every_burst_user_is_its_own_address(monkeypatch):
    """M3's premise, kept: 50 users are 50 identities, not one caller's allowance."""
    seen = []

    def handler(request):
        seen.append(request.headers["X-Real-IP"])
        return httpx.Response(200, json={"degraded": False})

    _serve(monkeypatch, handler)
    await load_test.run("http://api", 5, 5.0)
    assert len(set(seen)) == 5


# --- accounting ------------------------------------------------------------------


def _result(user, status, ms, category="single_sector", error=None):
    return load_test.Result(user, "S01", category, "q", status, ms, False, error)


def test_a_429_is_counted_as_a_failure_and_as_rate_limited():
    results = [_result(0, 200, 100.0), _result(1, 429, 5.0, error="HTTP 429")]
    summary = load_test.summarize(results, 1.0, mode="sustained", pace_s=2.5)
    assert summary["failed"] == 1 and summary["rate_limited"] == 1
    assert summary["offered_max_per_min"] == 2 * 60 / 2.5


# --- the pre-registered verdict ------------------------------------------------------


def _summary(p95_single, p95_cross, failed=0, rate_limited=0):
    return {
        "failed": failed,
        "rate_limited": rate_limited,
        "latency_by_category": {
            "single_sector": {"p95_ms": p95_single},
            "cross_sector": {"p95_ms": p95_cross},
        },
    }


def test_the_verdict_passes_inside_budget_with_no_failures():
    outcome = load_test.verdict(_summary(6_000, 12_000), _summary(5_000, 10_000))
    assert outcome["passed"]
    assert outcome["categories"]["single_sector"]["ratio"] == 1.2
    assert outcome["categories"]["single_sector"]["material_increase"] is False


def test_the_verdict_fails_on_a_single_failure_or_429():
    assert not load_test.verdict(_summary(6_000, 12_000, failed=1), _summary(5_000, 10_000))["passed"]
    outcome = load_test.verdict(_summary(6_000, 12_000, rate_limited=1), _summary(5_000, 10_000))
    assert not outcome["passed"] and outcome["checks"]["no_rate_limits"] is False


def test_the_verdict_fails_a_budget_breached_under_load():
    outcome = load_test.verdict(_summary(13_100, 12_000), _summary(6_000, 10_000))
    assert not outcome["passed"]
    assert outcome["categories"]["single_sector"]["material_increase"] is True


def test_a_breach_already_there_at_one_user_is_named_as_such():
    """Concurrency cannot be blamed for a budget the baseline already breaks."""
    outcome = load_test.verdict(_summary(11_000, 12_000), _summary(10_500, 10_000))
    assert outcome["categories"]["single_sector"]["breached_at_one_user"] is True


def test_the_verdict_reads_two_saved_runs(tmp_path, capsys):
    load = tmp_path / "load.json"
    base = tmp_path / "base.json"
    load.write_text(json.dumps({"summary": _summary(6_000, 12_000)}))
    base.write_text(json.dumps({"summary": _summary(5_000, 10_000)}))
    assert load_test.main(["--verdict", str(load), str(base)]) == 0
    assert '"passed": true' in capsys.readouterr().out


def test_a_baseline_never_shares_an_account_with_the_load_after_it():
    """Found by the first sustained run: sharing one meant sharing a rate-limit window."""
    names = {mode: load_test.LOAD_ACCOUNT.format(mode=mode, index=0)
             for mode in ("burst", "sequential", "sustained")}
    assert len(set(names.values())) == 3
