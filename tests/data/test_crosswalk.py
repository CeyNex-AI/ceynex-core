from ceynex.data.crosswalk import market_to_iso3


def test_resolves_a_plain_name():
    assert market_to_iso3("Germany") == ("DEU", 276)


def test_resolves_common_aliases_and_abbreviations():
    assert market_to_iso3("USA") == ("USA", 842)
    assert market_to_iso3("U.S.A.") == ("USA", 842)
    assert market_to_iso3("UK") == ("GBR", 826)
    assert market_to_iso3("Great Britain") == ("GBR", 826)


def test_strips_edb_style_parenthetical_alt_names():
    # Confirmed against the real EDB EPI PDFs (2023/2024 editions).
    assert market_to_iso3("Croatia (Hrvatska)") == ("HRV", 191)
    assert market_to_iso3("Czech Republic (Czechia)") == ("CZE", 203)
    assert market_to_iso3("Korea South (Korea, Republic of)") == ("KOR", 410)
    assert market_to_iso3("Iran (Islamic Republic of)") == ("IRN", 364)


def test_handles_edb_comma_style_names_without_parens():
    assert market_to_iso3("Taiwan, Province of China") == ("TWN", 158)
    assert market_to_iso3("Tanzania, United Republic of") == ("TZA", 834)


def test_unresolvable_name_returns_none_none_not_a_guess():
    assert market_to_iso3("Not Specified") == (None, None)
    assert market_to_iso3("Wakanda") == (None, None)
