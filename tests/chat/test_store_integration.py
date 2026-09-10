"""The chat store against a real Postgres (deviation D13).

Marked `integration` and excluded from `make test-unit`, following the existing
`tests/api/test_history_integration.py`. It exists because the unit tests for the
routes drive an in-memory fake: that proves the *routes* scope by user, and
proves nothing at all about whether the SQL parses, whether `Jsonb` round-trips a
forecast, or whether `ON DELETE CASCADE` is actually declared.

Cleans up before *and* after, scoped to test-only addresses, so a failed run
cannot leave rows behind that make the next one pass for the wrong reason.
"""

from __future__ import annotations

import psycopg
import pytest

from ceynex.api import history
from ceynex.chat import store
from ceynex.observability import ledger
from ceynex.observability.trace import TraceEvent
from ceynex.settings import postgres_dsn

pytestmark = pytest.mark.integration

TEST_USER = "chat-store-test@ceynex.invalid"
OTHER_USER = "chat-store-other@ceynex.invalid"


@pytest.fixture(autouse=True)
def clean_user():
    # The transcript read joins `query_history` for the save state, so the
    # table the API's lifespan creates has to exist here too.
    history.ensure_table()
    store.ensure_table()
    ledger.ensure_table()
    _purge()
    yield
    _purge()


def _purge() -> None:
    with psycopg.connect(postgres_dsn(), connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chat_conversation WHERE user_email = ANY(%s)",
            ([TEST_USER, OTHER_USER],),
        )
        cur.execute("DELETE FROM query_history WHERE user_email = ANY(%s)",
                    ([TEST_USER, OTHER_USER],))
        cur.execute("DELETE FROM llm_usage WHERE user_email = ANY(%s)",
                    ([TEST_USER, OTHER_USER],))
        # Trace events written against no conversation are not reachable by the
        # cascade above, so they are purged by request id instead.
        cur.execute("DELETE FROM chat_trace_event WHERE request_id = ANY(%s)",
                    (["rq", "req-1", "r1", "r2"],))
        conn.commit()


def _answer_message(request_id: str = "req-1") -> store.Message:
    return store.Message(
        role="assistant",
        content="Cinnamon exports to Germany rose 12%.",
        mode="analyse",
        request_id=request_id,
        confidence=0.61,
        confidence_band="Moderate",
        degraded=False,
        agents_used=["export_analytics"],
        route=["export_analytics"],
        sectors=["agriculture"],
        unanswered=["district breakdown"],
        evidence=[{"source_id": "KG", "claim": "c", "detail": "MATCH (n) RETURN n"}],
        forecast=[{"period": "2026", "point": 4.8, "lower": 4.1, "upper": 5.5, "unit": "USD m"}],
        graph={"nodes": [], "edges": [], "focus_id": None, "queries": [], "truncated": False},
        elapsed_ms=1234.5,
        usage={"calls": 2, "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.001},
    )


async def test_the_schema_applies_and_a_turn_round_trips():
    """The whole answer payload survives, not just the prose — reopening a chat
    has to redisplay the evidence panel, the forecast and the graph."""
    conversation_id = await store.create(TEST_USER, "cinnamon")
    await store.append(
        conversation_id,
        TEST_USER,
        [store.Message(role="user", content="how did cinnamon do"), _answer_message()],
    )

    messages = await store.messages(conversation_id, TEST_USER)
    assert [m.role for m in messages] == ["user", "assistant"]

    answer = messages[1]
    assert answer.confidence == pytest.approx(0.61)
    assert answer.evidence[0]["detail"] == "MATCH (n) RETURN n"
    assert answer.forecast[0]["point"] == pytest.approx(4.8)
    assert answer.graph["nodes"] == []
    assert answer.unanswered == ["district breakdown"]
    assert answer.usage["tokens_in"] == 100


async def test_seq_is_continuous_across_separate_appends():
    """Two turns written by two requests must not both start at 1, or the
    transcript reorders itself on read."""
    conversation_id = await store.create(TEST_USER)
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="one"), _answer_message("r1")])
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="two"), _answer_message("r2")])

    messages = await store.messages(conversation_id, TEST_USER)
    assert [m.seq for m in messages] == [1, 2, 3, 4]
    assert [m.content for m in messages][::2] == ["one", "two"]


async def test_another_user_cannot_read_or_write():
    conversation_id = await store.create(TEST_USER)
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="mine")])

    assert await store.messages(conversation_id, OTHER_USER) is None
    assert await store.owns(conversation_id, OTHER_USER) is False
    assert await store.append(conversation_id, OTHER_USER,
                              [store.Message(role="user", content="theirs")]) == []
    assert await store.update(conversation_id, OTHER_USER, title="mine now") is False
    assert await store.delete(conversation_id, OTHER_USER) is False

    # ...and none of that damaged the real owner's transcript.
    messages = await store.messages(conversation_id, TEST_USER)
    assert [m.content for m in messages] == ["mine"]


async def test_deleting_a_conversation_cascades_to_its_messages():
    """SRS 3.10 — a delete that leaves the messages behind has not deleted anything."""
    conversation_id = await store.create(TEST_USER)
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="x"), _answer_message()])

    assert await store.delete(conversation_id, TEST_USER) is True

    with psycopg.connect(postgres_dsn(), connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chat_message WHERE conversation_id = %s",
                    (conversation_id,))
        assert cur.fetchone()[0] == 0


async def test_listing_orders_pinned_first_then_recent():
    older = await store.create(TEST_USER, "older")
    newer = await store.create(TEST_USER, "newer")
    await store.update(older, TEST_USER, pinned=True)

    listed = await store.list_for_user(TEST_USER)
    assert [c.id for c in listed] == [older, newer]


async def test_archived_conversations_are_excluded_by_default():
    conversation_id = await store.create(TEST_USER, "old")
    await store.update(conversation_id, TEST_USER, archived=True)

    assert await store.list_for_user(TEST_USER) == []
    assert len(await store.list_for_user(TEST_USER, include_archived=True)) == 1


async def test_message_count_counts_turns_not_rows():
    """A sidebar saying "4 messages" for two exchanges is confusing; it counts
    what the user asked."""
    conversation_id = await store.create(TEST_USER)
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="one"), _answer_message()])
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="two"), _answer_message("r2")])

    listed = await store.list_for_user(TEST_USER)
    assert listed[0].message_count == 2


async def test_a_title_is_set_once_and_never_over_a_chosen_one():
    conversation_id = await store.create(TEST_USER)

    assert await store.set_title_if_unset(conversation_id, TEST_USER, "auto name") is True
    assert await store.set_title_if_unset(conversation_id, TEST_USER, "second try") is False

    listed = await store.list_for_user(TEST_USER)
    assert listed[0].title == "auto name"

    await store.update(conversation_id, TEST_USER, title="chosen by hand")
    assert await store.set_title_if_unset(conversation_id, TEST_USER, "auto again") is False
    assert (await store.list_for_user(TEST_USER))[0].title == "chosen by hand"


async def test_a_trace_round_trips_in_order():
    conversation_id = await store.create(TEST_USER)
    events = [
        TraceEvent(seq=1, request_id="rq", ts=1.0, kind="route",
                   node="route", payload={"route": ["export_analytics"]}),
        TraceEvent(seq=2, request_id="rq", ts=2.0, kind="kg_query",
                   node="export_analytics", payload={"cypher": "MATCH (n)", "row_count": 3}),
    ]
    await store.save_trace("rq", conversation_id, events)

    replayed = await store.trace_for("rq")
    assert [e["seq"] for e in replayed] == [1, 2]
    assert replayed[1]["cypher"] == "MATCH (n)"
    assert replayed[1]["node"] == "export_analytics"


async def test_the_usage_ledger_records_and_rolls_up():
    from ceynex.observability.context import LLMCall

    await ledger.record(
        request_id="rq-usage",
        user_email=TEST_USER,
        conversation_id=None,
        calls=[
            LLMCall(role="router", model="gpt-4o-mini", provider="openai",
                    tokens_in=1000, tokens_out=70, cost_usd=0.0002),
            LLMCall(role="merge", model="gpt-4o", provider="cache", cache_hit=True),
        ],
    )

    recorded = await ledger.for_request("rq-usage")
    assert [c.role for c in recorded] == ["router", "merge"]
    # A cache hit is 0 tokens and $0 — correct accounting, not a gap.
    assert recorded[1].cache_hit is True
    assert recorded[1].tokens_in == 0
    assert recorded[1].cost_usd == 0.0

    by_role = {r.key: r for r in await ledger.by_role_and_model(TEST_USER)}
    assert any("router" in key for key in by_role)
    assert any("(cached)" in key for key in by_role), "cache hits must stay separable"


async def test_deleting_a_conversation_also_removes_its_trace():
    """Found by a leaking test purge, which is how the missing foreign key
    surfaced. Without the cascade, deleting a conversation left every Cypher
    query, token count and timing from its analyses in the database for good —
    exactly what the delete endpoint's own docstring says it does not do
    (SRS 3.10).
    """
    conversation_id = await store.create(TEST_USER)
    await store.save_trace(
        "rq-cascade",
        conversation_id,
        [TraceEvent(seq=1, request_id="rq-cascade", ts=1.0, kind="route",
                    node="route", payload={})],
    )
    assert await store.trace_for("rq-cascade")

    await store.delete(conversation_id, TEST_USER)
    assert await store.trace_for("rq-cascade") == []


async def test_a_pending_clarification_can_only_be_claimed_once():
    """The one-round cap is a row, not a counter, and this is why.

    `resolve_clarification` is a conditional `UPDATE ... RETURNING`, so under the
    deployed image's two uvicorn workers the first request to claim the row gets
    the query back and the second gets nothing — rather than both deciding the
    question is unanswered and running the five-agent fan-out twice. An
    in-memory fake can be written to pass either way, so this is asserted against
    real Postgres.
    """
    conversation_id = await store.create(TEST_USER)
    pending_id = await store.record_clarification(
        conversation_id, TEST_USER, "tea and cinnamon?", {"question": "Which?"}
    )

    assert (await store.open_clarification(conversation_id, TEST_USER)) is not None
    assert (await store.resolve_clarification(pending_id, TEST_USER)) is not None
    assert (await store.resolve_clarification(pending_id, TEST_USER)) is None
    # And it is no longer open, so a reload does not re-ask.
    assert (await store.open_clarification(conversation_id, TEST_USER)) is None


async def test_another_user_cannot_claim_a_pending_clarification():
    conversation_id = await store.create(TEST_USER)
    pending_id = await store.record_clarification(
        conversation_id, TEST_USER, "tea and cinnamon?", {"question": "Which?"}
    )
    assert (await store.resolve_clarification(pending_id, OTHER_USER)) is None
    assert (await store.resolve_clarification(pending_id, TEST_USER)) is not None


async def test_deleting_a_conversation_also_removes_its_pending_question():
    """Same reasoning as the trace cascade above: a question about a conversation
    that no longer exists has nothing to resume into."""
    conversation_id = await store.create(TEST_USER)
    pending_id = await store.record_clarification(
        conversation_id, TEST_USER, "tea and cinnamon?", {"question": "Which?"}
    )
    await store.delete(conversation_id, TEST_USER)
    assert (await store.resolve_clarification(pending_id, TEST_USER)) is None


# --- what a reopened transcript must still show (W0) -----------------------


async def test_the_working_behind_the_score_survives_a_reload():
    """"Why this confidence?" was shown on the live turn and silently missing on
    the same turn reopened a minute later."""
    conversation_id = await store.create(TEST_USER)
    answer = _answer_message()
    answer.confidence_breakdown = {"base": 0.7, "staleness": -0.05, "final": 0.61}
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="q"), answer])

    reloaded = (await store.messages(conversation_id, TEST_USER))[1]
    assert reloaded.confidence_breakdown == {"base": 0.7, "staleness": -0.05, "final": 0.61}


async def test_a_withheld_discussion_is_still_marked_as_withheld():
    conversation_id = await store.create(TEST_USER)
    discussion = store.Message(role="assistant", content="withheld", mode="discuss",
                               grounded=False)
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="summarise"), discussion])

    reloaded = (await store.messages(conversation_id, TEST_USER))[1]
    assert reloaded.grounded is False


async def test_the_transcript_keeps_what_was_typed_beside_what_was_run():
    """A follow-up is rewritten before the graph runs. The reader typed "now do
    rubber", and a reopened chat must not show them a question they never asked."""
    conversation_id = await store.create(TEST_USER)
    typed = store.Message(role="user", content="now do rubber",
                          effective_query="What are Sri Lanka's rubber export trends?")
    await store.append(conversation_id, TEST_USER, [typed, _answer_message()])

    reloaded = (await store.messages(conversation_id, TEST_USER))[0]
    assert reloaded.content == "now do rubber"
    assert reloaded.asked == "What are Sri Lanka's rubber export trends?"


async def test_the_save_state_is_read_from_the_history_row_it_links_to():
    """One saved flag, read in two places. The chat star and the History panel
    cannot disagree because there is only one row to disagree about."""
    history_id = history.record(user_email=TEST_USER, query="q", answer="a",
                                confidence=0.6, degraded=False)
    assert isinstance(history_id, int)

    conversation_id = await store.create(TEST_USER)
    answer = _answer_message()
    answer.query_history_id = history_id
    await store.append(conversation_id, TEST_USER,
                       [store.Message(role="user", content="q"), answer])
    assert (await store.messages(conversation_id, TEST_USER))[1].saved is False

    history.set_saved(history_id, TEST_USER, saved=True)
    assert (await store.messages(conversation_id, TEST_USER))[1].saved is True


async def test_a_regenerated_answer_links_to_the_version_it_replaces():
    conversation_id = await store.create(TEST_USER)
    first_ids = await store.append(conversation_id, TEST_USER,
                                   [store.Message(role="user", content="q"),
                                    _answer_message("r1")])
    second = _answer_message("r2")
    second.regenerated_from = first_ids[1]
    await store.append(conversation_id, TEST_USER, [second])

    messages = await store.messages(conversation_id, TEST_USER)
    assert [m.role for m in messages] == ["user", "assistant", "assistant"]
    assert messages[2].regenerated_from == messages[1].id
    # Both versions are kept: Regenerate adds, it never overwrites.
    assert {m.request_id for m in messages[1:]} == {"r1", "r2"}
