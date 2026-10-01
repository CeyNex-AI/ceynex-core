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
    duration_s, pace_s = 0.12, 0.05
    results, wall = await load_test.run_sustained("http://api", 2, 5.0, duration_s=duration_s,
                                                  pace_s=pace_s, questions=QUESTIONS)
    assert 2 <= len(results) <= 6
    # A tight `< duration_s` bound is over-precise: the boundary check runs
    # against the real clock, but real dispatch still follows a few Python
    # statements later, and asyncio.sleep's own wake-up overshoot (worse on
    # Windows' coarser timer resolution) occasionally carries that past the
    # boundary by a few ms. Bound it by one full pace interval instead --
    # generous against that jitter, while still catching a real bug (duration_s
    # ignored outright would send every per_user request, not stop early).
    assert all(r.sent_at_s < duration_s + pace_s for r in results)


async def test_a_sustained_run_needs_a_bound():
    with pytest.raises(ValueError, match="per-user or --duration"):
        await load_test.run_sustained("http://api", 1, 5.0)


# --- identity ----------------------------------------------------------------------


def test_a_user_sends_its_token_and_its_own_address():
    assert load_test._headers(3, "tok") == {"X-Real-IP": "10.50.0.4", "Authorization": "Bearer tok"}


# --- accounts: every run is signed in (SRS 3.1.11) ---------------------------------


def _one_result():
    return load_test.Result(
        user=0,
        id="Q01",
        category="single_sector",
        question="q",
        status=200,
        elapsed_ms=100.0,
        degraded=False,
        error=None,
    )


@pytest.fixture
def fake_accounts(monkeypatch):
    """Account creation and deletion recorded, and the run itself faked."""
    log = {"created": [], "deleted": [], "tokens": []}

    def create(count, mode="burst"):
        accounts = [(100 + i, f"tok-{mode}-{i}") for i in range(count)]
        log["created"].extend(accounts)
        return accounts

    async def run(base_url, users, timeout_s, *, endpoint="query", tokens=None):
        log["tokens"].extend(tokens or [])
        return [_one_result()], 1.0

    monkeypatch.setattr(load_test, "create_accounts", create)
    monkeypatch.setattr(
        load_test, "delete_accounts", lambda accounts: log["deleted"].extend(accounts)
    )
    monkeypatch.setattr(load_test, "run", run)
    return log


def test_the_accounts_file_round_trips_and_is_owner_only(tmp_path):
    path = tmp_path / "load.json"
    load_test.write_accounts(path, [(7, "tok-a"), (8, "tok-b")], "sustained")
    assert load_test.read_accounts(path) == [(7, "tok-a"), (8, "tok-b")]
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_run_creates_its_own_accounts_and_deletes_them(fake_accounts):
    assert load_test.main(["--users", "3"]) == 0
    assert fake_accounts["tokens"] == ["tok-burst-0", "tok-burst-1", "tok-burst-2"]
    assert fake_accounts["deleted"] == fake_accounts["created"]


def test_emit_tokens_writes_the_accounts_and_runs_nothing(fake_accounts, tmp_path):
    path = tmp_path / "load.json"
    assert load_test.main(["--users", "2", "--mode", "sustained", "--emit-tokens", str(path)]) == 0
    assert load_test.read_accounts(path) == [(100, "tok-sustained-0"), (101, "tok-sustained-1")]
    assert fake_accounts["tokens"] == [] and fake_accounts["deleted"] == []


def test_a_run_on_minted_tokens_leaves_the_accounts_to_their_owner(fake_accounts, tmp_path):
    path = tmp_path / "load.json"
    load_test.write_accounts(path, [(5, "minted-0"), (6, "minted-1"), (7, "minted-2")], "burst")
    assert load_test.main(["--users", "2", "--tokens", str(path)]) == 0
    assert fake_accounts["tokens"] == ["minted-0", "minted-1"]
    assert fake_accounts["created"] == [] and fake_accounts["deleted"] == []


def test_too_few_minted_tokens_is_refused(fake_accounts, tmp_path):
    path = tmp_path / "load.json"
    load_test.write_accounts(path, [(5, "minted-0")], "burst")
    with pytest.raises(SystemExit, match="holds 1 accounts"):
        load_test.main(["--users", "2", "--tokens", str(path)])


def test_delete_accounts_deletes_what_the_file_names(fake_accounts, tmp_path):
    path = tmp_path / "load.json"
    load_test.write_accounts(path, [(5, "minted-0"), (6, "minted-1")], "burst")
    assert load_test.main(["--delete-accounts", str(path)]) == 0
    assert fake_accounts["deleted"] == [(5, "minted-0"), (6, "minted-1")]


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
