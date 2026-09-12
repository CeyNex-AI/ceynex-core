"""Releasing an answer sentence by sentence, each grounded first — SRS 3.1.3, D12.

Streaming the answer text is the obvious next step once the trace streams, and
the obvious way to do it would break the one guarantee this project is built on.
`merger._reject_ungrounded_prose` checks the *whole* prose for figures no finding
supports, and throws it away if it finds one. Stream tokens straight to the
screen and the reader has already seen the invented number by the time that
check runs. Retracting it afterwards is not the same as never having shown it:
the whole claim is that a figure on this screen traces to a source, and a
fabricated one shown for two seconds is still shown.

So the answer is released a **sentence** at a time, and each sentence is held
until it passes the same check, against the same corpus, that the merge applies
to the whole:

- **Exact, not approximate.** `grounding.NUMBER` cannot span whitespace, and a
  sentence boundary here is always whitespace after terminal punctuation, so no
  figure is ever cut in two. Every figure in the prose therefore lands wholly
  inside exactly one sentence, and "every released sentence is grounded" is
  implied by the whole-prose check the merge already runs. The gate never shows
  a figure the merge would reject.
- **The first failing sentence stops the stream.** Nothing after it is shown.
  The merge then rejects the prose exactly as it always did, and `close()` emits
  `answer_reset` so the reader is told the draft was withdrawn and why — before
  the `done` frame delivers the deterministic composition in its place.
- **A retry starts over, visibly.** When the client retries a timed-out call or
  falls back to the failsafe, `restart()` withdraws what the first attempt
  showed rather than splicing two attempts together.

A sentence is the smallest unit that can be checked honestly. A token cannot:
"exports fell by" is only grounded or not once the number arrives. The cost is
latency to the first word — the first sentence rather than the first token —
which is still most of a merge call sooner than the whole answer.

The `done` frame stays authoritative. The guards that append to a merged answer
(`_ensure_conflicts_stated` and its siblings) run after the model finishes, so
their sentences arrive with `done`, which replaces the draft outright.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from ceynex.observability import trace
from ceynex.orchestrator.grounding import ungrounded_figures

log = logging.getLogger(__name__)

#: Terminal punctuation, any closing quotes or brackets, whitespace, then the
#: start of another sentence. Whitespace is the whole point: no figure contains
#: any, so a split here can never cut one. Missing a boundary only makes one
#: release larger; it never makes one ungrounded.
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def _emit_delta(text: str, index: int) -> None:
    trace.emit_live("answer_delta", text=text, index=index)


def _emit_reset(reason: str) -> None:
    trace.emit("answer_reset", reason=reason)


class SentenceGate:
    """Holds streamed text until a whole, grounded sentence can be released.

    Satisfies `llm.client.TextStream`. `corpus` must be exactly what the final
    whole-prose check uses — the merger builds both from `_grounding_corpus`.
    """

    def __init__(
        self,
        corpus: list[str],
        *,
        emit_delta: Callable[[str, int], None] = _emit_delta,
        emit_reset: Callable[[str], None] = _emit_reset,
    ) -> None:
        self._corpus = corpus
        self._emit_delta = emit_delta
        self._emit_reset = emit_reset
        self._buffer = ""
        self._released: list[str] = []
        self._held = False

    @property
    def released(self) -> str:
        """Everything shown so far, exactly as shown."""
        return "".join(self._released)

    @property
    def held(self) -> bool:
        """True once a sentence failed the check; nothing more will be shown."""
        return self._held

    def feed(self, chunk: str) -> None:
        if self._held or not chunk:
            return
        self._buffer += chunk
        while True:
            boundary = SENTENCE_END.search(self._buffer)
            if boundary is None:
                return
            sentence, rest = self._buffer[: boundary.end()], self._buffer[boundary.end():]
            if not self._release(sentence):
                return
            self._buffer = rest

    def restart(self) -> None:
        """A new attempt is starting. Withdraw anything the last one showed."""
        if self._released:
            self._emit_reset("retry")
        self._buffer = ""
        self._released = []
        self._held = False

    def close(self, *, accepted: bool, reason: str = "ungrounded") -> None:
        """The model has finished and the whole-prose verdict is in.

        `accepted` means the prose will be served. Its last sentence has no
        boundary after it, so it is released here — through the same check, for
        uniformity, even though an accepted prose cannot fail it. Otherwise any
        released draft is withdrawn: `reason` is "ungrounded" when the merge
        rejected the prose, "degraded" when the model stopped partway.
        """
        if accepted:
            if self._held:
                # Cannot happen — a failing sentence makes the whole prose fail
                # the same check — and if it ever does, the reader must not be
                # left looking at a draft the answer does not match.
                log.warning("gate held a sentence the whole-prose check accepted")
                self._emit_reset("ungrounded")
            elif self._buffer.strip():
                self._release(self._buffer)
            self._buffer = ""
            return
        if self._released:
            self._emit_reset(reason)
        self._buffer = ""

    def _release(self, sentence: str) -> bool:
        missing = ungrounded_figures(sentence, self._corpus)
        if missing:
            log.info("holding a streamed sentence with unsourced figure(s): %s", missing)
            self._held = True
            return False
        self._emit_delta(sentence, len(self._released))
        self._released.append(sentence)
        return True


__all__ = ["SENTENCE_END", "SentenceGate"]
