"""Assertions for the tariff-rate extractor — the one place a figure comes from prose.

`docs/EVALUATION.md` §5 records the limitation these tests exist to hold the line
on: `orchestrator/grounding.py` compares digit strings, so a real percentage
lifted from the wrong sentence passes every runtime check and is still wrong.
S06 is the live example of that class of failure.

So the tests that matter most here are the ones asserting it returns **None** —
each refusal is a wrong answer the system will not give.
"""

from ceynex.retrieval.rates import extract_tariff_rate
from ceynex.retrieval.schema import PolicyChunk

APPAREL = ("61", "6109")
TEA = ("0902",)


def chunk(text: str) -> PolicyChunk:
    return PolicyChunk(
        doc_id="USA-USTR-TARIFF-ACTIONS",
        chunk_index=0,
        text=text,
        title="Presidential Tariff Actions",
        publisher="USTR",
        url="https://example.invalid/tariff",
        page=3,
    )


# --- what it accepts -----------------------------------------------------


def test_reads_a_rate_stated_for_the_goods_in_question():
    found = extract_tariff_rate(
        [chunk("Knitted apparel under HS 61 faces an MFN duty of 16.5% on entry.")], APPAREL
    )

    assert found is not None
    assert found.rate == 0.165
    assert "16.5" in found.claim
    assert found.chunk.url == "https://example.invalid/tariff"


def test_the_claim_quotes_the_sentence_the_rate_came_from():
    """The citation has to carry the sentence, not just the document.

    A page reference alone leaves a reader hunting for which of the page's
    numbers was used, which is the difference between checkable and merely
    attributed.
    """
    found = extract_tariff_rate(
        [chunk("Cotton T-shirts, HS 6109, carry a tariff of 16.5% ad valorem.")], APPAREL
    )

    assert found is not None
    assert "6109" in found.claim
    assert "USTR" in found.claim


# --- what it refuses -----------------------------------------------------


def test_refuses_a_rate_stated_for_different_goods():
    """The condition that does the real work.

    A tariff schedule lists dozens of rates. Without requiring the goods to be
    named in the same sentence, the extractor returns whichever appears first
    and attaches a footwear duty to a question about tea.
    """
    found = extract_tariff_rate(
        [chunk("Footwear under HS 64 faces an MFN duty of 37.5% on entry.")], TEA
    )

    assert found is None


def test_refuses_a_percentage_that_is_not_about_a_tariff():
    found = extract_tariff_rate(
        [chunk("Knitted apparel under HS 61 grew 16.5% year on year.")], APPAREL
    )

    assert found is None


def test_refuses_when_two_different_rates_both_qualify():
    """Ambiguity is reported as no-rate, never resolved by picking the first.

    Two candidates means the passage does not say which applies, and choosing
    one produces a figure with a real citation that the citation does not
    support.
    """
    found = extract_tariff_rate(
        [
            chunk("Knitted apparel, HS 61, faces an MFN duty of 16.5%."),
            chunk("Duties on HS 61 knitted garments are 32.0% under the general column."),
        ],
        APPAREL,
    )

    assert found is None


def test_the_same_rate_stated_twice_is_not_ambiguous():
    found = extract_tariff_rate(
        [
            chunk("Knitted apparel, HS 61, faces an MFN duty of 16.5%."),
            chunk("The applied tariff for HS 61 knitwear remains 16.5% ad valorem."),
        ],
        APPAREL,
    )

    assert found is not None
    assert found.rate == 0.165


def test_refuses_an_implausible_rate():
    """A "percentage" of 400 is a share, an index or a typo — never an import duty."""
    assert extract_tariff_rate([chunk("HS 61 duty collections rose to 400%.")], APPAREL) is None


def test_refuses_a_zero_rate():
    """"0% duty" states a preference, not a rate to re-impose on losing it.

    Accepting it would model losing duty-free access as a shock of zero, i.e.
    report that nothing happens — a confidently wrong answer to the exact
    question the agent is being asked.
    """
    assert extract_tariff_rate([chunk("HS 61 enters at 0% duty under GSP.")], APPAREL) is None


def test_no_chunks_is_no_rate():
    assert extract_tariff_rate([], APPAREL) is None
