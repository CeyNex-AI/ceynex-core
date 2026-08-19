"""Assertions for the Comtrade connector (SRS 3.1.7).

No network. The fixture is a real 20-row response for Sri Lankan tea exports in
2023, chosen to contain every trap at once: the World aggregate (partner 0), two
Comtrade variant codes (842 USA, 699 India, 251 France), and ordinary partners.

The alias test is a regression test for a bug that dropped USD 6.66bn of US
apparel exports with nothing but a log line.
"""

import json
import shutil
from pathlib import Path

import pytest

from ceynex.data.connectors.comtrade import (
    FACT_TRADE_COLUMNS,
    SOURCE_ID,
    ComtradeConnector,
)

FIXTURE = Path(__file__).parent / "fixtures" / "comtrade_0902_2023.json"


@pytest.fixture
def connector(tmp_path):
    """An offline connector whose cache is pre-seeded with the fixture."""
    cache = tmp_path / "comtrade"
    (cache / "2023-01-01").mkdir(parents=True)
    shutil.copy(FIXTURE, cache / "2023-01-01" / "0902_2023.json")
    return ComtradeConnector(
        hs_codes=("0902",), years=(2023,), cache_root=cache, offline=True, api_key=None
    )


def test_it_reads_the_cache_rather_than_the_network(connector):
    raw = connector.fetch()
    assert len(raw) == 20
    assert connector.manifest().source_id == SOURCE_ID
    assert connector.manifest().row_count == 20


def test_offline_with_no_cache_returns_empty_rather_than_raising(tmp_path):
    """R2: a missing key must not stop the pipeline, only shrink it."""
    connector = ComtradeConnector(
        hs_codes=("0902",), years=(2023,), cache_root=tmp_path / "empty", offline=True
    )
    assert connector.fetch().empty
    assert connector.to_fact_trade(connector.fetch()).empty


# --- the two silent-corruption traps -------------------------------------


def test_the_world_aggregate_is_dropped(connector):
    """partner 0 is the total across all partners; keeping it doubles everything."""
    raw = connector.fetch()
    assert 0 in set(raw["partnerCode"]), "fixture must contain the trap it guards"

    facts = connector.to_fact_trade(raw)
    assert 0 not in set(facts["partner_m49"])


def test_comtrade_variant_partner_codes_survive(connector):
    """Regression: 842/699/251 are the USA, India and France, not unknown codes.

    Dropping them removed USD 6.66bn of US exports alone from a three-year pull.
    """
    raw = connector.fetch()
    facts = connector.to_fact_trade(raw)

    resolved = dict(zip(facts["partner_m49"], facts["partner_iso3"], strict=True))
    assert resolved[842] == "USA"
    assert resolved[699] == "IND"
    assert resolved[251] == "FRA"


def test_no_export_value_is_lost_except_to_aggregates(connector):
    """Every non-aggregate row in must be a row out. Silence is the failure mode."""
    raw = connector.fetch()
    facts = connector.to_fact_trade(raw)

    non_aggregate = raw[raw["partnerCode"] != 0]
    assert len(facts) == len(non_aggregate), "a partner was dropped without being an aggregate"


# --- contract conformance ------------------------------------------------


def test_output_has_exactly_the_fact_trade_columns(connector):
    facts = connector.to_fact_trade(connector.fetch())
    assert list(facts.columns) == FACT_TRADE_COLUMNS


def test_non_nullable_contract_columns_are_populated(connector):
    facts = connector.to_fact_trade(connector.fetch())
    for column in (
        "source_id",
        "sector",
        "item",
        "hs_code",
        "reporter_iso3",
        "reporter_m49",
        "period_start",
        "period_end",
        "frequency",
    ):
        assert facts[column].notna().all(), f"{column} has nulls"


def test_hs_codes_keep_their_leading_zero(connector):
    """Comtrade returns 902 for tea; joining on that matches nothing."""
    facts = connector.to_fact_trade(connector.fetch())
    assert set(facts["hs_code"]) == {"0902"}


def test_sri_lanka_is_always_the_reporter(connector):
    facts = connector.to_fact_trade(connector.fetch())
    assert set(facts["reporter_iso3"]) == {"LKA"}
    assert set(facts["reporter_m49"]) == {144}


def test_annual_periods_span_the_whole_year(connector):
    facts = connector.to_fact_trade(connector.fetch())
    assert set(facts["frequency"]) == {"A"}
    assert set(facts["period_start"]) == {"2023-01-01"}
    assert set(facts["period_end"]) == {"2023-12-31"}


def test_unit_price_is_value_over_volume_or_none(connector):
    facts = connector.to_fact_trade(connector.fetch())
    priced = facts[facts["price"].notna()]
    assert not priced.empty
    for row in priced.to_dict("records"):
        assert row["price"] == pytest.approx(row["export_value_usd"] / row["export_volume"])
        assert row["price_unit"] == "USD/kg"


def test_the_tea_figure_matches_the_source(connector):
    """One human-verified number, per the team rule that numbers are not trusted
    unsupervised. The full 2023 pull sums to USD 1.27bn against a published ~1.3bn;
    this fixture is a 19-partner subset of it, so it must be smaller but the same
    order of magnitude."""
    facts = connector.to_fact_trade(connector.fetch())
    total = facts["export_value_usd"].sum()
    assert 1e8 < total < 1.3e9


# --- endpoint selection --------------------------------------------------


def test_no_key_means_the_public_preview_endpoint(tmp_path):
    connector = ComtradeConnector(cache_root=tmp_path, api_key=None)
    assert connector.uses_subscription is False
    assert "preview" in connector.endpoint


def test_a_key_means_the_subscription_endpoint(tmp_path):
    connector = ComtradeConnector(cache_root=tmp_path, api_key="a-key")
    assert connector.uses_subscription is True
    assert "preview" not in connector.endpoint


def test_manifest_before_fetch_is_an_error_not_a_lie(tmp_path):
    with pytest.raises(RuntimeError, match="fetch"):
        ComtradeConnector(cache_root=tmp_path).manifest()


def test_the_fixture_still_contains_the_traps_it_guards():
    """If someone regenerates the fixture, this fails rather than the tests above
    quietly passing against data with nothing left to catch."""
    rows = json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]
    codes = {row["partnerCode"] for row in rows}
    assert 0 in codes, "lost the World aggregate"
    assert {842, 699, 251} <= codes, "lost the Comtrade variant partner codes"
