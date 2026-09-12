"""Assertions for conversation naming (deviation D13, SRS 3.4.3).

Naming runs after the answer has already been delivered, so the bar is that it
never costs anything: not a failed turn, not a slower one, and never a name the
user chose being quietly replaced.
"""

from __future__ import annotations

from ceynex.chat.store import MAX_TITLE
from ceynex.chat.titles import FALLBACK_CHARS, fallback_title, title_for
from ceynex.llm import FakeLLMClient


class ScriptedLLM:
    available = True

    def __init__(self, reply):
        self.reply = reply
        self.roles: list[str] = []

    async def generate(self, role, system, user, *, json_mode=False):
        self.roles.append(role)
        return self.reply


# --- the fallback, which is also the degraded path ------------------------


def test_a_short_question_becomes_the_title_unchanged():
    assert fallback_title("cinnamon exports") == "cinnamon exports"


def test_a_long_question_is_cut_on_a_word_boundary():
    """A name ending mid-word reads as a bug rather than as a summary."""
    title = fallback_title("how did Sri Lankan cinnamon exports to Germany change during 2025")
    assert len(title) <= FALLBACK_CHARS + 1
    assert not title.rstrip("…").endswith(" ")
    assert "…" in title
    # the cut fell between words, not inside one
    assert title.rstrip("…").split()[-1] in ["how", "did", "Sri", "Lankan", "cinnamon", "exports", "to", "Germany", "change", "during", "2025"]


def test_an_empty_question_still_produces_a_name():
    assert fallback_title("   ") == "New conversation"


def test_whitespace_is_normalised():
    assert fallback_title("cinnamon\n\n  exports") == "cinnamon exports"


async def test_no_llm_falls_back_to_the_question():
    name = await title_for("cinnamon exports", "It rose.", FakeLLMClient(available=False))
    assert name == "cinnamon exports"


async def test_a_raising_model_falls_back():
    class Exploding:
        available = True

        async def generate(self, *args, **kwargs):
            raise RuntimeError("provider on fire")

    name = await title_for("cinnamon exports", "It rose.", Exploding())
    assert name == "cinnamon exports"


# --- the model's name -----------------------------------------------------


async def test_a_good_name_is_used():
    llm = ScriptedLLM("Cinnamon exports to Germany")
    name = await title_for("how did cinnamon exports to Germany change", "It rose.", llm)

    assert name == "Cinnamon exports to Germany"
    assert llm.roles == ["title"], "own role, so its spend is separable"


async def test_surrounding_quotes_and_trailing_stops_are_stripped():
    name = await title_for("q", "a", ScriptedLLM('"Cinnamon exports to Germany."'))
    assert name == "Cinnamon exports to Germany"


async def test_only_the_first_line_is_taken():
    """Models like to answer a naming request with a sentence and then the name."""
    name = await title_for("q", "a", ScriptedLLM("Cinnamon markets\n\nLet me know if..."))
    assert name == "Cinnamon markets"


async def test_a_model_that_writes_a_sentence_falls_back():
    llm = ScriptedLLM("This conversation is about how cinnamon exports to Germany changed in 2025")
    name = await title_for("cinnamon exports", "It rose.", llm)
    assert name == "cinnamon exports"


async def test_an_over_long_name_falls_back():
    name = await title_for("cinnamon exports", "a", ScriptedLLM("x" * (MAX_TITLE + 1)))
    assert name == "cinnamon exports"


async def test_an_empty_reply_falls_back():
    assert await title_for("cinnamon exports", "a", ScriptedLLM("")) == "cinnamon exports"
    assert await title_for("cinnamon exports", "a", ScriptedLLM(None)) == "cinnamon exports"
