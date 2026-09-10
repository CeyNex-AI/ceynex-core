"""`countries_in` — full names and adjectival/people forms.

The demonym support exists because a question phrased "What do the Chinese
import rules say about Sri Lankan cinnamon?" otherwise resolved no destination
market and `trade_economics` answered it as a no-country coverage question
(live audit, E09-adjacent).
"""

from __future__ import annotations

import pytest

from ceynex.retrieval.tagging import countries_in


@pytest.mark.parametrize(
    "text,expected",
    [
        ("What does China's trade policy say about Sri Lankan tea?", "CHN"),
        ("What do the Chinese import rules say about Sri Lankan cinnamon?", "CHN"),
        ("How do Indian tariffs on Sri Lankan tea compare?", "IND"),
        ("the German market for knitted apparel", "DEU"),
        ("American duties on textiles", "USA"),
        ("British DCTS eligibility", "GBR"),
        ("Dutch demand for cinnamon", "NLD"),
        ("Japanese phytosanitary measures", "JPN"),
        ("Vietnamese competition in woven garments", "VNM"),
    ],
)
def test_a_demonym_resolves_to_the_country(text, expected):
    assert expected in countries_in(text)


def test_a_full_name_and_a_demonym_for_the_same_country_do_not_double_count():
    got = countries_in("China's policy and Chinese practice")
    assert got.count("CHN") == 1


@pytest.mark.parametrize(
    "text",
    [
        "fishing rights in the Indian Ocean",
        "the classic case of Dutch disease in a commodity economy",
        "a North American supply chain",
        "South American coffee exporters",
        "nail polish exports from the EU",
    ],
)
def test_a_demonym_inside_a_false_positive_phrase_is_ignored(text):
    got = countries_in(text)
    assert "IND" not in got
    assert "NLD" not in got
    assert "USA" not in got
    assert "POL" not in got


def test_indian_ocean_does_not_mask_a_real_india_mention_elsewhere():
    got = countries_in("India's exports and the Indian Ocean shipping lanes")
    assert "IND" in got  # matched from the full name "India", not the blanked phrase


def test_plain_english_country_names_still_work():
    got = countries_in("exports from Sri Lanka to Germany and the United States")
    assert got[:1] == ("DEU",) or "DEU" in got
    assert "USA" in got and "LKA" in got
