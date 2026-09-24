"""Assertions for the FX connector (SRS 3.1.7).

No network. The fixture is a real World Bank API response for Sri Lanka's
PA.NUS.FCRF indicator (fetched 2026-09-24), chosen because it contains the
trap this connector has to handle: the two most recent years (2024, 2025)
report `value: null` since the World Bank hasn't published them yet.
"""

import shutil
from pathlib import Path

import pytest

from ceynex.data.connectors.fx import ITEM, REPORTER_ISO3, SOURCE_ID, FXConnector

FIXTURE = Path(__file__).parent / "fixtures" / "wb_fx_lka.json"


@pytest.fixture
def connector(tmp_path):
    """An offline connector whose cache is pre-seeded with the real fixture."""
    cache = tmp_path / "fx"
    (cache / "2026-09-24").mkdir(parents=True)
    shutil.copy(FIXTURE, cache / "2026-09-24" / "wb_fx.json")
    return FXConnector(cache_root=cache, offline=True)


def test_it_reads_the_cache_rather_than_the_network(connector):
    raw = connector.fetch()
    assert len(raw) == 66
    assert connector.manifest().source_id == SOURCE_ID
    assert connector.manifest().row_count == 66


def test_offline_with_no_cache_returns_empty_rather_than_raising(tmp_path):
    connector = FXConnector(cache_root=tmp_path / "empty", offline=True)
    assert connector.fetch().empty
    assert connector.to_fact_trade(connector.fetch()).empty


def test_null_recent_years_are_dropped_not_written_as_zero(connector):
    """2024 and 2025 report value: null in the real response -- must not
    become a fabricated 0.0 rate in fact_trade."""
    raw = connector.fetch()
    fact_trade = connector.to_fact_trade(raw)
    years = fact_trade["period_start"].dt.year.tolist()
    assert 2024 not in years
    assert 2025 not in years
    assert (fact_trade["fx_usd_lkr"] > 0).all()


def test_maps_a_known_real_year_correctly(connector):
    fact_trade = connector.to_fact_trade(connector.fetch())
    row = fact_trade[fact_trade["period_start"].dt.year == 2023].iloc[0]
    assert row["source_id"] == SOURCE_ID
    assert row["sector"] == "macro"
    assert row["item"] == ITEM
    assert row["reporter_iso3"] == REPORTER_ISO3
    assert row["hs_code"] is None
    assert row["partner_iso3"] is None
    assert row["export_value_usd"] is None
    assert round(float(row["fx_usd_lkr"]), 2) == 327.51


def test_years_filter_trims_the_result(tmp_path):
    cache = tmp_path / "fx"
    (cache / "2026-09-24").mkdir(parents=True)
    shutil.copy(FIXTURE, cache / "2026-09-24" / "wb_fx.json")
    connector = FXConnector(years=(2020, 2023), cache_root=cache, offline=True)

    fact_trade = connector.to_fact_trade(connector.fetch())

    years = sorted(fact_trade["period_start"].dt.year.tolist())
    assert years == [2020, 2021, 2022, 2023]


def test_is_repeatable(connector):
    first = connector.to_fact_trade(connector.fetch())
    second = connector.to_fact_trade(connector.fetch())
    assert first["fx_usd_lkr"].tolist() == second["fx_usd_lkr"].tolist()
    assert (first["source_hash"] == second["source_hash"]).all()
