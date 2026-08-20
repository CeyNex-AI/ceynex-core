"""Assertions for SRS 3.1.8 / 3.6.1 country and HS code reconciliation.

The EU-aggregate tests are the ones that matter. Every other failure here is
loud; double-counting a bloc alongside its member states is silent, and it makes
every market-share figure in the system wrong without raising anything.
"""

import pandas as pd
import pytest

from ceynex.data.crosswalk import (
    CrosswalkError,
    _countries,
    aggregate_partner_label,
    country_name,
    dim_country_rows,
    dim_hs_rows,
    drop_aggregate_partners,
    hs_chapter,
    hs_description,
    hs_sector,
    is_aggregate_partner,
    is_in_scope,
    is_known_country,
    known_aliases,
    market_to_iso3,
    normalize_hs,
    to_iso3,
    to_m49,
)

# --- countries -----------------------------------------------------------


def test_sri_lanka_is_lka_144():
    """The one mapping the whole project is built on."""
    assert to_m49("LKA") == 144
    assert to_iso3(144) == "LKA"
    assert country_name("LKA") == "Sri Lanka"


@pytest.mark.parametrize("country", _countries(), ids=lambda c: c.iso3)
def test_every_country_round_trips(country):
    """iso3 -> m49 -> iso3 for all 255 rows, historical entries included."""
    assert to_m49(country.iso3) == country.m49
    assert to_iso3(country.m49) == country.iso3


def test_to_iso3_accepts_what_it_is_already():
    """Sources mix conventions in one column; callers should not have to branch."""
    assert to_iso3("LKA") == to_iso3("lka") == to_iso3(144) == to_iso3("144") == "LKA"


def test_to_iso3_resolves_a_country_name():
    assert to_iso3("Bangladesh") == "BGD"
    assert to_iso3("Viet Nam") == "VNM"


def test_unknown_identifiers_raise_rather_than_return_none():
    """A silent None becomes a NULL partner and an orphan graph node."""
    for bad in ("XXX", "Atlantis", 9999):
        with pytest.raises(CrosswalkError):
            to_iso3(bad)
    assert not is_known_country("XXX")


def test_historical_reporters_still_resolve():
    """Ten years of trade series reach back past several dissolutions."""
    assert to_iso3(891) == "SCG"
    assert to_m49("ANT") == 530


# --- the double-counting trap -------------------------------------------


def test_eu_aggregate_is_not_a_country():
    """partner=97 is reported alongside DEU, FRA and ITA. Counting it doubles them."""
    assert is_aggregate_partner(97)
    assert aggregate_partner_label(97) == "EU (as reported)"
    with pytest.raises(CrosswalkError, match="aggregate partner"):
        to_iso3(97)


def test_world_partner_is_not_a_country():
    """partner=0 is the total across all partners."""
    assert is_aggregate_partner(0)
    with pytest.raises(CrosswalkError, match="aggregate partner"):
        to_iso3(0)


def test_real_partners_are_not_flagged_as_aggregates():
    for iso3 in ("DEU", "USA", "GBR", "BGD", "IND"):
        assert not is_aggregate_partner(to_m49(iso3)), iso3


def test_drop_aggregate_partners_keeps_only_real_countries():
    frame = pd.DataFrame(
        {
            "partner_m49": [276, 97, 250, 0, 380, 899],
            "export_value_usd": [100.0, 250.0, 90.0, 500.0, 60.0, 5.0],
        }
    )
    kept = drop_aggregate_partners(frame)

    assert sorted(kept["partner_m49"]) == [250, 276, 380]
    # 500 (World) + 250 (EU) would have inflated a 250 total to 1005.
    assert kept["export_value_usd"].sum() == 250.0


def test_drop_aggregate_partners_says_which_column_it_wanted():
    with pytest.raises(KeyError, match="partner_m49"):
        drop_aggregate_partners(pd.DataFrame({"partner": [276]}))


# --- HS codes ------------------------------------------------------------


def test_leading_zero_is_restored():
    """Comtrade returns codes as integers; 902 is tea, not chapter 90."""
    assert normalize_hs(902, digits=4) == "0902"
    assert normalize_hs(906, digits=4) == "0906"
    assert normalize_hs("90611", digits=6) == "090611"


@pytest.mark.parametrize(
    ("code", "digits", "expected"),
    [
        ("610910", 6, "610910"),
        ("610910", 4, "6109"),
        ("610910", 2, "61"),
        ("6109", 4, "6109"),
        ("6109", 2, "61"),
        ("090611", 4, "0906"),
        ("090611", 2, "09"),
    ],
)
def test_truncation_walks_the_hs_hierarchy(code, digits, expected):
    """HS is hierarchical: truncate, never round."""
    assert normalize_hs(code, digits=digits) == expected


def test_chapter_helper_matches_two_digit_truncation():
    assert hs_chapter("610910") == "61"
    assert hs_chapter(902) == "09"


def test_cannot_invent_precision():
    """4 digits cannot become 6. Padding here would fabricate a subheading."""
    with pytest.raises(CrosswalkError, match="cannot express"):
        normalize_hs("6109", digits=6)


def test_rejects_things_that_are_not_hs_codes():
    for bad in ("cinnamon", "61A9", ""):
        with pytest.raises(CrosswalkError):
            normalize_hs(bad)
    with pytest.raises(ValueError, match="2, 4, 6"):
        normalize_hs("6109", digits=5)


@pytest.mark.parametrize(
    ("code", "sector"),
    [
        ("0902", "agriculture"),   # tea
        ("090611", "agriculture"), # cinnamon, resolved via its 4-digit heading
        ("4001", "agriculture"),   # rubber
        ("151311", "agriculture"), # coconut oil
        ("6109", "apparel"),       # the SRS 3.1.9 GSP+ worked example
        ("620342", "apparel"),     # resolved via chapter 62
    ],
)
def test_sector_resolves_by_walking_up_the_hierarchy(code, sector):
    assert hs_sector(code) == sector


def test_out_of_scope_codes_are_rejected_not_guessed():
    """SRS 2.4 fixes scope to agriculture and apparel."""
    for code in ("8703", "2709", "7108"):  # cars, crude oil, gold
        assert not is_in_scope(code)
        with pytest.raises(CrosswalkError, match="outside CeyNex's sector scope"):
            hs_sector(code)


def test_descriptions_exist_for_the_headline_codes():
    assert "Tea" in hs_description("0902")
    assert "Cinnamon" in hs_description("0906")
    assert "T-shirts" in hs_description("6109")


# --- dimension seeds -----------------------------------------------------


def test_dim_country_rows_are_unique_and_current_only():
    rows = dim_country_rows()
    assert len({r[0] for r in rows}) == len(rows), "duplicate iso3 would break the primary key"
    assert len({r[1] for r in rows}) == len(rows), "duplicate m49 would break the unique index"
    assert "SCG" not in {r[0] for r in rows}, "historical entries do not belong in dim_country"
    assert ("LKA", 144, "Sri Lanka") in rows


def test_dim_hs_rows_carry_only_the_two_contracted_sectors():
    rows = dim_hs_rows()
    assert {r[2] for r in rows} == {"agriculture", "apparel"}
    assert len({r[0] for r in rows}) == len(rows)


# --- free-text market names (EDB / JAAF) ----------------------------------


def test_market_to_iso3_resolves_a_plain_name():
    assert market_to_iso3("Germany") == ("DEU", 276)


def test_market_to_iso3_resolves_common_aliases_and_abbreviations():
    # 840, not Comtrade's non-standard partner code 842 — see the module
    # docstring on `_partner_aliases()`. EDB/JAAF market names resolve to the
    # same official UN M49 that every other source in fact_trade uses.
    assert market_to_iso3("USA") == ("USA", 840)
    assert market_to_iso3("U.S.A.") == ("USA", 840)
    assert market_to_iso3("UK") == ("GBR", 826)
    assert market_to_iso3("Great Britain") == ("GBR", 826)


def test_market_to_iso3_strips_edb_style_parenthetical_alt_names():
    # Confirmed against the real EDB EPI PDFs (2023/2024 editions).
    assert market_to_iso3("Croatia (Hrvatska)") == ("HRV", 191)
    assert market_to_iso3("Czech Republic (Czechia)") == ("CZE", 203)
    assert market_to_iso3("Korea South (Korea, Republic of)") == ("KOR", 410)
    assert market_to_iso3("Iran (Islamic Republic of)") == ("IRN", 364)


def test_market_to_iso3_handles_edb_comma_style_names_without_parens():
    assert market_to_iso3("Taiwan, Province of China") == ("TWN", 158)
    assert market_to_iso3("Tanzania, United Republic of") == ("TZA", 834)


def test_market_to_iso3_unresolvable_name_returns_none_none_not_a_guess():
    assert market_to_iso3("Not Specified") == (None, None)
    assert market_to_iso3("Wakanda") == (None, None)


def test_known_aliases_round_trip_through_market_to_iso3():
    for alias in known_aliases():
        iso3, m49 = market_to_iso3(alias)
        assert iso3 is not None, f"{alias!r} is a known alias but did not resolve"
        assert to_m49(iso3) == m49
