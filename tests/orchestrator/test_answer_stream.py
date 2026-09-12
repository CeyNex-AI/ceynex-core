"""The answer streams a sentence at a time, and never shows an unsourced figure.

SRS 3.1.3 is enforced on the whole prose by `merger._reject_ungrounded_prose`.
Streaming could have undone it — a figure shown is shown, whatever happens
next — so the property pinned here is the strong one: **nothing the gate
releases could have been rejected by the whole-prose check**, at any chunk
boundary, and a draft that will not be served is withdrawn before the answer
that replaces it arrives.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from ceynex.contracts import AgentOutput, Evidence, new_state
from ceynex.llm import FakeLLMClient
from ceynex.observability import context, trace
from ceynex.orchestrator.answer_stream import SentenceGate
from ceynex.orchestrator.grounding import numbers_in, ungrounded_figures
from ceynex.orchestrator.merger import merge

CORPUS = [
    "Tea exports were USD 1,234,567 in 2024.",
    "Cinnamon grew 4.25 percent to USD 88,200.",
    "The HHI was 1,850.",
]

GROUNDED = (
    "Tea exports were USD 1,234,567 in 2024. "
    "Cinnamon grew 4.25 percent to USD 88,200, led by Mexico. "
    "Concentration is moderate, with an HHI of 1,850 [2]. "
    "The U.S. market was not covered."
)

#: Sentence two states a figure no finding supports.
INVENTED = (
    "Tea exports were USD 1,234,567 in 2024. "
    "Rubber reached USD 9,999,999 last year. "
    "Cinnamon grew 4.25 percent."
)


class Recorder:
    def __init__(self):
        self.deltas: list[str] = []
        self.resets: list[str] = []

    def gate(self, corpus=CORPUS) -> SentenceGate:
        return SentenceGate(
            corpus,
            emit_delta=lambda text, index: self.deltas.append(text),
            emit_reset=self.resets.append,
        )


def _feed(gate: SentenceGate, text: str, cuts: list[int]) -> None:
    last = 0
    for cut in [*sorted(cuts), len(text)]:
        gate.feed(text[last:cut])
        last = cut


# --- the property ------------------------------------------------------------


@pytest.mark.parametrize("seed", range(200))
def test_released_text_is_always_a_prefix_of_a_grounded_answer(seed):
    """Any chunking the provider might choose — boundaries inside words, inside
    figures, inside "4.25" — releases a prefix of the answer, and closing
    releases the rest."""
    rng = random.Random(seed)
    cuts = rng.sample(range(1, len(GROUNDED)), k=rng.randint(0, 40))
    record = Recorder()
    gate = record.gate()

    _feed(gate, GROUNDED, cuts)
    assert GROUNDED.startswith(gate.released)
    gate.close(accepted=True)
    assert gate.released == GROUNDED
    assert record.resets == []


@pytest.mark.parametrize("seed", range(200))
def test_nothing_released_could_have_failed_the_whole_prose_check(seed):
    """The gate's promise, stated as the merger's own check: every figure the
    reader was shown is one `ungrounded_figures` accepts."""
    rng = random.Random(seed)
    cuts = rng.sample(range(1, len(INVENTED)), k=rng.randint(0, 40))
    record = Recorder()
    gate = record.gate()

    _feed(gate, INVENTED, cuts)
    shown = gate.released
    assert ungrounded_figures(shown, CORPUS) == []
    assert "9,999,999" not in shown
    assert gate.held


@pytest.mark.parametrize("seed", range(50))
def test_a_figure_is_never_split_across_two_releases(seed):
    """A boundary is whitespace after terminal punctuation, and no figure
    contains whitespace — so every figure in every release is a whole figure of
    the answer: never "1,234" shown now and "567" later."""
    rng = random.Random(seed)
    cuts = rng.sample(range(1, len(GROUNDED)), k=rng.randint(1, 40))
    record = Recorder()
    gate = record.gate()
    _feed(gate, GROUNDED, cuts)
    gate.close(accepted=True)

    whole = numbers_in(GROUNDED)
    for delta in record.deltas:
        assert numbers_in(delta) <= whole, f"a partial figure was released: {delta!r}"


def test_decimals_and_abbreviations_do_not_end_a_sentence():
    record = Recorder()
    gate = record.gate(["Growth was 4.25 percent in the U.S. market."])
    gate.feed("Growth was 4.25 percent in the U.S. market. Next")
    assert record.deltas == ["Growth was 4.25 percent in the U.S. market. "]


# --- withdrawal ------------------------------------------------------------------


def test_the_stream_stops_at_the_first_unsourced_sentence():
    record = Recorder()
    gate = record.gate()
    gate.feed(INVENTED)
    assert record.deltas == ["Tea exports were USD 1,234,567 in 2024. "]
    gate.feed(" More text that must never be shown.")
    assert len(record.deltas) == 1


def test_a_rejected_draft_is_withdrawn_before_the_answer_arrives():
    record = Recorder()
    gate = record.gate()
    gate.feed(INVENTED)
    gate.close(accepted=False, reason="ungrounded")
    assert record.resets == ["ungrounded"]


def test_nothing_shown_means_nothing_to_withdraw():
    record = Recorder()
    gate = record.gate()
    gate.feed("Rubber reached USD 9,999,999 last year. Then more.")
    gate.close(accepted=False, reason="ungrounded")
    assert record.deltas == [] and record.resets == []


def test_a_retry_withdraws_the_attempt_it_replaces():
    """A timed-out first attempt must not be spliced onto the second."""
    record = Recorder()
    gate = record.gate()
    gate.feed("Tea exports were USD 1,234,567 in 2024. Cinnamon")
    gate.restart()
    gate.feed(GROUNDED)
    gate.close(accepted=True)

    assert record.resets == ["retry"]
    assert gate.released == GROUNDED


def test_restarting_before_anything_was_shown_says_nothing():
    record = Recorder()
    gate = record.gate()
    gate.restart()
    assert record.resets == []


def test_a_model_that_stops_partway_withdraws_its_draft_as_degraded():
    record = Recorder()
    gate = record.gate()
    gate.feed("Tea exports were USD 1,234,567 in 2024. Cinn")
    gate.close(accepted=False, reason="degraded")
    assert record.resets == ["degraded"]


# --- wired into the merge ------------------------------------------------------------


def _merge_state():
    st = new_state("How did tea exports do in 2024?", "tester")
    st["agent_outputs"] = {
        "export_analytics": AgentOutput(
            agent="export_analytics",
            summary="Tea exports were USD 1,234,567 in 2024.",
            figures={"tea_export_value_usd": 1234567.0},
            assumptions=[],
            evidence=[Evidence(source_id="KG", claim="Tea exports were USD 1,234,567 in 2024.",
                               detail="MATCH (n) RETURN n", period="2024")],
            confidence=0.8,
            degraded=False,
        )
    }
    st["route"] = ["export_analytics"]
    st["relevance"] = {"export_analytics": 1.0}
    return st


async def _merge_with_sink(llm):
    sink = trace.TraceSink(request_id="r", loop=asyncio.get_running_loop())
    token = context.install(context.RequestObservability(request_id="r", trace=sink))
    try:
        result = await merge(_merge_state(), llm)
    finally:
        context.reset(token)
    queued = []
    while not sink.queue.empty():
        queued.append(sink.queue.get_nowait())
    return result, sink, queued


async def test_merge_without_a_listener_never_streams():
    """`POST /api/query` and `make eval` have no sink. Their merge call must be
    exactly the call it always was — no stream argument at all."""
    llm = FakeLLMClient(response="Tea exports were USD 1,234,567 in 2024.")
    await merge(_merge_state(), llm)
    assert llm.streamed == [False]


async def test_a_listened_merge_streams_its_prose_sentence_by_sentence():
    prose = "Tea exports were USD 1,234,567 in 2024. They were the largest item."
    result, sink, queued = await _merge_with_sink(FakeLLMClient(response=prose))

    deltas = [e.payload["text"] for e in queued if e.kind == "answer_delta"]
    assert "".join(deltas) == prose
    assert result.answer.startswith(prose)
    # The answer text is not written to the stored trace — the message row is
    # its durable copy.
    assert not any(e.kind == "answer_delta" for e in sink.history)


async def test_an_ungrounded_merge_is_withdrawn_and_replaced():
    prose = "Tea exports were USD 1,234,567 in 2024. Rubber hit USD 9,999,999."
    result, sink, queued = await _merge_with_sink(FakeLLMClient(response=prose))

    shown = "".join(e.payload["text"] for e in queued if e.kind == "answer_delta")
    assert "9,999,999" not in shown
    assert [e.payload for e in sink.history if e.kind == "answer_reset"] == [
        {"reason": "ungrounded"}
    ]
    assert "9,999,999" not in result.answer, "the deterministic composition replaced it"
    assert result.ungrounded


# --- the direction rule keeps the gate and the whole-prose check agreeing ---------

FELL_CORPUS = ["Losing GSP+ changes export value by USD -161,815,198."]
#: "decrease" shares the figure's sentence, so the figure is grounded.
FELL_SAID = (
    "Losing GSP+ would cut export value. "
    "The decrease would be about USD 161,815,198 a year. "
    "Rubber is not affected."
)
#: The word is a sentence away from the figure, so it is not.
FELL_ELSEWHERE = (
    "Export value would decrease. "
    "The change would be USD 161,815,198 a year. "
    "Rubber is not affected."
)


@pytest.mark.parametrize("seed", range(100))
def test_the_direction_rule_gives_the_gate_the_whole_prose_verdict(seed, monkeypatch):
    """The rule reads a figure's own sentence, and the gate sees one sentence at
    a time. Both must cut sentences in the same places, or a sentence could pass
    the gate and fail the whole-prose check, or the reverse."""
    monkeypatch.setenv("CEYNEX_GROUNDING", "direction")
    for text, grounded in ((FELL_SAID, True), (FELL_ELSEWHERE, False)):
        rng = random.Random(seed)
        cuts = rng.sample(range(1, len(text)), k=rng.randint(0, 40))
        gate = Recorder().gate(FELL_CORPUS)
        _feed(gate, text, cuts)
        assert (ungrounded_figures(text, FELL_CORPUS) == []) is grounded
        assert ungrounded_figures(gate.released, FELL_CORPUS) == []
        if grounded:
            gate.close(accepted=True)
            assert gate.released == text
        else:
            assert gate.held and "161,815,198" not in gate.released
