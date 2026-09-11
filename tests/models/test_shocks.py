"""The shock formulas, written once (D17).

The refactor that created `ceynex/models/shocks.py` moved arithmetic out of
`agents/trade_economics.py`, and a moved formula that changed by one sign or one
rounding would change every simulation figure the evaluation has published.
The golden values below were computed from the agent's code *before* the move
(`git show d71083d:ceynex/agents/trade_economics.py`), so the check is against
the formulas as they were, not against the module checking itself.
"""

from __future__ import annotations

import asyncio

import pytest

from ceynex.agents import trade_economics as te
from ceynex.models import shocks
from ceynex.settings import elasticity_config

BASELINE = 1_000_000.0

# (delta_usd, pct, detail) from the pre-refactor `_simulate_fx` / `_simulate_tariff`.
GOLDEN = {
    ("fx", "agriculture", 0.05): (
        -5999.999999999998, -0.005999999999999998,
        "agriculture: FX pass-through 0.60 and export demand elasticity -0.80 applied linearly "
        "to a 5.0% depreciation. Volume effect +2.4%, price effect -3.0%.",
    ),
    ("fx", "agriculture", -0.03): (
        3599.999999999999, 0.003599999999999999,
        "agriculture: FX pass-through 0.60 and export demand elasticity -0.80 applied linearly "
        "to a -3.0% depreciation. Volume effect -1.4%, price effect +1.8%.",
    ),
    ("fx", "apparel", 0.10): (
        8000.0, 0.008,
        "apparel: FX pass-through 0.40 and export demand elasticity -1.20 applied linearly "
        "to a 10.0% depreciation. Volume effect +4.8%, price effect -4.0%.",
    ),
    ("tariff", "agriculture", 0.10): (
        -40000.00000000001, -0.04000000000000001,
        "agriculture: 10.0% tariff with exporter incidence 0.50 and demand elasticity -0.80. "
        "Buyer price +5.0%.",
    ),
    ("tariff", "apparel", 0.05): (
        -30000.0, -0.03,
        "apparel: 5.0% tariff with exporter incidence 0.50 and demand elasticity -1.20. "
        "Buyer price +2.5%.",
    ),
    ("tariff", "apparel", -0.03): (
        18000.0, 0.018,
        "apparel: -3.0% tariff with exporter incidence 0.50 and demand elasticity -1.20. "
        "Buyer price -1.5%.",
    ),
}

COVERAGE_ROWS = [
    {"agreement": "GSP+", "agreement_type": "unilateral_preference", "matched_on": "61",
     "agreement_verified": "verified"},
    {"agreement": "UK DCTS", "agreement_type": "unilateral_preference", "matched_on": "61",
     "agreement_verified": "unverified"},
]

GOLDEN_AGREEMENT = {
    "apparel": (
        -56999.99999999999, -0.056999999999999995,
        "apparel: preference coverage resolved from the knowledge graph (GSP+, UK DCTS, matched "
        "on HS 61, status unverified/verified). Loss modelled as an MFN tariff of 9.5% from "
        "config/elasticities.yaml (a literature constant, not a queried tariff schedule — "
        "deviation D9), with exporter incidence 0.50 and demand elasticity -1.20.",
    ),
    "agriculture": (
        -22000.000000000004, -0.022000000000000002,
        "agriculture: preference coverage resolved from the knowledge graph (GSP+, UK DCTS, "
        "matched on HS 61, status unverified/verified). Loss modelled as an MFN tariff of 5.5% "
        "from config/elasticities.yaml (a literature constant, not a queried tariff schedule — "
        "deviation D9), with exporter incidence 0.50 and demand elasticity -0.80.",
    ),
}


@pytest.fixture
def config():
    return elasticity_config()


# --- parity with the formulas as they were ---------------------------------------


@pytest.mark.parametrize(("shock", "sector", "magnitude"), sorted(GOLDEN))
def test_the_shared_formula_matches_the_agents_old_output_to_the_cent(
    config, shock, sector, magnitude
):
    fn = shocks.fx_shock if shock == "fx" else shocks.tariff_shock
    outcome = fn(sector, BASELINE, magnitude, config)
    delta, pct, detail = GOLDEN[(shock, sector, magnitude)]
    assert outcome.revenue_change_usd == pytest.approx(delta, abs=1e-9)
    assert outcome.revenue_change_pct == pytest.approx(pct, abs=1e-12)
    assert outcome.detail == detail


@pytest.mark.parametrize(("shock", "sector", "magnitude"), sorted(GOLDEN))
def test_the_agent_wrapper_returns_exactly_the_same_tuple(config, shock, sector, magnitude):
    """The agent's `(delta, pct, detail)` shape survives, byte for byte."""
    wrapper = te._simulate_fx if shock == "fx" else te._simulate_tariff
    delta, pct, detail = GOLDEN[(shock, sector, magnitude)]
    got = wrapper(sector, BASELINE, magnitude, config)
    assert got[0] == pytest.approx(delta, abs=1e-9)
    assert got[1] == pytest.approx(pct, abs=1e-12)
    assert got[2] == detail


@pytest.mark.parametrize("sector", sorted(GOLDEN_AGREEMENT))
def test_agreement_loss_through_the_agent_matches_its_old_output(config, sector):
    class KG:
        async def run(self, cypher, params=None):
            return COVERAGE_ROWS, cypher

    class Deps:
        kg = KG()

    item = "apparel_knit" if sector == "apparel" else "tea"
    (delta, pct, detail), _cypher = asyncio.run(
        te._simulate_agreement_loss(Deps(), sector, item, BASELINE, config)
    )
    want = GOLDEN_AGREEMENT[sector]
    assert delta == pytest.approx(want[0], abs=1e-9)
    assert pct == pytest.approx(want[1], abs=1e-12)
    assert detail == want[2]


def test_agreement_loss_from_the_workbench_side_matches_the_agent(config):
    """Same rows, same config: the slider path and the agent path agree."""
    outcome = shocks.agreement_loss_shock(
        "apparel", BASELINE, config, coverage=shocks.describe_coverage(COVERAGE_ROWS)
    )
    delta, pct, detail = GOLDEN_AGREEMENT["apparel"]
    assert outcome.revenue_change_usd == pytest.approx(delta, abs=1e-9)
    assert outcome.revenue_change_pct == pytest.approx(pct, abs=1e-12)
    assert outcome.detail == detail
    assert [p.name for p in outcome.parameters] == [
        "agreement_loss_mfn_tariff", "tariff_incidence", "export_demand_elasticity"
    ]


# --- the arithmetic's own properties --------------------------------------------


def test_fx_revenue_is_the_volume_effect_plus_the_price_effect(config):
    outcome = shocks.fx_shock("agriculture", BASELINE, 0.05, config)
    assert outcome.revenue_change_pct == pytest.approx(
        outcome.volume_change_pct + outcome.price_change_pct
    )
    assert outcome.price_change_pct < 0, "a depreciation lowers the foreign price"
    assert outcome.volume_change_pct > 0, "and demand rises in response"


def test_a_tariff_lowers_revenue_and_a_negative_tariff_raises_it(config):
    assert shocks.tariff_shock("apparel", BASELINE, 0.05, config).revenue_change_usd < 0
    assert shocks.tariff_shock("apparel", BASELINE, -0.05, config).revenue_change_usd > 0


def test_losing_a_preference_can_never_raise_revenue(config):
    for sector in shocks.SECTORS:
        outcome = shocks.agreement_loss_shock(sector, BASELINE, config, coverage="GSP+")
        assert outcome.revenue_change_usd < 0


# --- provenance travels with every parameter --------------------------------------


def test_every_parameter_carries_its_basis_and_source_verbatim(config):
    """The config's `TBD` placeholders must reach the page as `TBD`, not be
    tidied into something that looks sourced."""
    outcome = shocks.fx_shock("agriculture", BASELINE, 0.05, config)
    by_name = {p.name: p for p in outcome.parameters}
    assert by_name["fx_pass_through"].basis == "literature_range"
    assert by_name["fx_pass_through"].source.startswith("TBD")
    assert by_name["export_demand_elasticity"].source.startswith("TBD")
    assert all(not p.overridden for p in outcome.parameters)


def test_an_override_changes_only_the_named_parameter_and_says_so(config):
    plain = shocks.fx_shock("agriculture", BASELINE, 0.05, config)
    moved = shocks.fx_shock(
        "agriculture", BASELINE, 0.05, config, overrides={"fx_pass_through": 1.0}
    )
    pt = {p.name: p for p in moved.parameters}["fx_pass_through"]
    assert pt.overridden and pt.value == 1.0 and pt.basis == "override"
    assert pt.default == {p.name: p for p in plain.parameters}["fx_pass_through"].value
    assert {p.name: p for p in moved.parameters}["export_demand_elasticity"] == (
        {p.name: p for p in plain.parameters}["export_demand_elasticity"]
    )
    # Full pass-through of a 5% depreciation: price -5%, volume +4%, revenue -1%.
    assert moved.revenue_change_pct == pytest.approx(-0.01)


def test_an_overridden_mfn_rate_is_not_called_a_literature_constant(config):
    """The detail line the page shows must say whose number it is."""
    outcome = shocks.agreement_loss_shock(
        "apparel", BASELINE, config, coverage="GSP+",
        overrides={"agreement_loss_mfn_tariff": 0.15},
    )
    assert "set in the scenario workbench" in outcome.detail
    assert "overriding the 9.5% literature constant" in outcome.detail
    assert "a literature constant, not a queried" not in outcome.detail


def test_a_none_override_is_no_override(config):
    plain = shocks.fx_shock("apparel", BASELINE, 0.05, config)
    same = shocks.fx_shock("apparel", BASELINE, 0.05, config, overrides={"fx_pass_through": None})
    assert same == plain


def test_a_missing_config_entry_falls_back_and_is_labelled_as_such():
    outcome = shocks.tariff_shock("apparel", BASELINE, 0.05, config={})
    by_name = {p.name: p for p in outcome.parameters}
    assert by_name["tariff_incidence"].value == shocks.DEFAULTS["tariff_incidence"]
    assert by_name["tariff_incidence"].basis == "fallback"
    assert by_name["export_demand_elasticity"].value == shocks.DEFAULTS["export_demand_elasticity"]


def test_base_assumptions_are_never_empty_and_name_the_shock(config):
    """SRS 3.1.5 — the simulation must state its assumptions."""
    assumptions = shocks.base_assumptions(config, "fx", 0.05)
    assert assumptions and assumptions[0] == "Shock modelled: fx, magnitude 5.0%."
    assert te._base_assumptions(config, "fx", 0.05) == assumptions
