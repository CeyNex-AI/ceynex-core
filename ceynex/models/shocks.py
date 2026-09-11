"""Implements SRS 3.1.5 — the shock formulas, written once — deviation D17.

`agents/trade_economics.py` simulates three shocks: a rupee depreciation, a
tariff, and the loss of a unilateral preference. The scenario workbench
(`api/routes/scenario.py`) lets a reader move the same parameters with sliders
and re-run in place. Two copies of a formula are two formulas that can drift,
so both call the functions here, and `tests/models/test_shocks.py` asserts the
agent and the workbench agree to the cent on the same inputs.

**Every parameter carries its provenance.** `config/elasticities.yaml` records a
`basis` and a `source` beside each value, and several of those sources are still
`TBD` placeholders. The agent reads only the value; the workbench shows all
three, because a slider over a number nobody has sourced must say so rather
than look like a fitted estimate. An overridden value is marked as such and
keeps the configured default beside it.

**Pure arithmetic, no I/O, no model call.** Baselines and coverage come from the
knowledge graph through the caller; nothing here reaches a database or an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SHOCKS = ("fx", "tariff", "agreement")
SECTORS = ("agriculture", "apparel")

#: Config groups a caller may override, and which key of each group a sector
#: reads. `tariff_incidence` has one entry for both sectors.
OVERRIDABLE = {
    "fx_pass_through": "sector",
    "export_demand_elasticity": "sector",
    "tariff_incidence": "default",
    "agreement_loss_mfn_tariff": "sector",
}

#: Code defaults, used only when the config table has no entry at all. The
#: agent has always tolerated a missing entry this way; the value is then
#: reported with basis `fallback` so nobody mistakes it for a sourced one.
DEFAULTS = {
    "fx_pass_through": 0.5,
    "export_demand_elasticity": -1.0,
    "tariff_incidence": 0.5,
    "agreement_loss_mfn_tariff": 0.095,
}


@dataclass(frozen=True)
class Parameter:
    """One elasticity as it entered a calculation, with where it came from."""

    name: str
    value: float
    #: The configured value, kept beside an override so the page can show both.
    default: float
    #: `literature_range` / `assumption` from the config, `override` when the
    #: caller supplied the value, `sourced` when a policy document did, and
    #: `fallback` when the config had no entry.
    basis: str
    source: str
    overridden: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "default": self.default,
            "basis": self.basis,
            "source": self.source,
            "overridden": self.overridden,
        }


@dataclass(frozen=True)
class ShockOutcome:
    """What a shock does to one sector's export revenue, and how it was worked out."""

    shock: str
    sector: str
    baseline_usd: float
    revenue_change_usd: float
    revenue_change_pct: float
    #: The price the buyer faces, as a fraction. Negative for a depreciation.
    price_change_pct: float
    #: The volume response to that price move.
    volume_change_pct: float
    parameters: tuple[Parameter, ...]
    #: The one-sentence working the agent records as an assumption, verbatim.
    detail: str

    def as_tuple(self) -> tuple[float, float, str]:
        """The shape the agent has always passed around: `(delta, pct, detail)`."""
        return self.revenue_change_usd, self.revenue_change_pct, self.detail

    def as_dict(self) -> dict[str, Any]:
        return {
            "shock": self.shock,
            "sector": self.sector,
            "baseline_usd": self.baseline_usd,
            "revenue_change_usd": self.revenue_change_usd,
            "revenue_change_pct": self.revenue_change_pct,
            "price_change_pct": self.price_change_pct,
            "volume_change_pct": self.volume_change_pct,
            "parameters": [p.as_dict() for p in self.parameters],
            "detail": self.detail,
        }


def parameter(
    config: dict[str, Any],
    group: str,
    key: str,
    default: float | None = None,
    *,
    overrides: dict[str, float | None] | None = None,
) -> Parameter:
    """Read one elasticity from the config table, tolerating a missing entry.

    The value logic is exactly the agent's old `_value()`; what is new is that
    the entry's `basis` and `source` come along, and an override — keyed by
    group, the way the workbench's sliders are — replaces the value while
    keeping the configured default and the source it is overriding.
    """
    fallback = DEFAULTS[group] if default is None else default
    entry = config.get(group, {}).get(key)
    if isinstance(entry, dict) and "value" in entry:
        configured = float(entry["value"])
        basis = str(entry.get("basis", "unstated"))
        source = str(entry.get("source", "unstated"))
    elif isinstance(entry, int | float):
        configured, basis, source = float(entry), "unstated", "config/elasticities.yaml"
    else:
        configured, basis = fallback, "fallback"
        source = "code default; no entry in config/elasticities.yaml"

    override = (overrides or {}).get(group)
    if override is not None:
        return Parameter(group, float(override), configured, "override", source, True)
    return Parameter(group, configured, configured, basis, source)


def fx_shock(
    sector: str,
    baseline: float,
    depreciation: float,
    config: dict[str, Any],
    *,
    overrides: dict[str, float | None] | None = None,
) -> ShockOutcome:
    """A rupee depreciation makes exports cheaper abroad, so volume rises.

    Two parameters, both from the config table:

    - **pass-through**: how much of the currency move reaches the foreign-currency
      price. Contract-priced apparel passes less through than spot-traded
      agricultural commodities.
    - **export demand elasticity**: how much volume responds to that price change.

    USD revenue change ≈ pass_through × depreciation × (−elasticity − 1).
    The −1 is the price effect: each unit earns fewer dollars, which offsets part
    of the volume gain. Omitting it is the classic error that reports a
    depreciation as pure upside.
    """
    pass_through = parameter(config, "fx_pass_through", sector, overrides=overrides)
    elasticity = parameter(config, "export_demand_elasticity", sector, overrides=overrides)

    price_change = -pass_through.value * depreciation  # foreign price falls
    volume_change = elasticity.value * price_change  # demand rises as price falls
    revenue_change = volume_change + price_change

    detail = (
        f"{sector}: FX pass-through {pass_through.value:.2f} and export demand elasticity "
        f"{elasticity.value:.2f} applied linearly to a {depreciation * 100:.1f}% depreciation. "
        f"Volume effect {volume_change * 100:+.1f}%, price effect {price_change * 100:+.1f}%."
    )
    return ShockOutcome(
        "fx", sector, baseline, baseline * revenue_change, revenue_change,
        price_change, volume_change, (pass_through, elasticity), detail,
    )


def tariff_shock(
    sector: str,
    baseline: float,
    tariff: float,
    config: dict[str, Any],
    *,
    overrides: dict[str, float | None] | None = None,
) -> ShockOutcome:
    """An importing country's tariff raises the buyer's price by the incidence share.

    The exporter's own unit price is held: the whole revenue effect is the
    volume lost to the higher landed price.
    """
    incidence = parameter(config, "tariff_incidence", "default", overrides=overrides)
    elasticity = parameter(config, "export_demand_elasticity", sector, overrides=overrides)

    price_change = incidence.value * tariff
    revenue_change = elasticity.value * price_change

    detail = (
        f"{sector}: {tariff * 100:.1f}% tariff with exporter incidence {incidence.value:.2f} and "
        f"demand elasticity {elasticity.value:.2f}. Buyer price {price_change * 100:+.1f}%."
    )
    return ShockOutcome(
        "tariff", sector, baseline, baseline * revenue_change, revenue_change,
        price_change, revenue_change, (incidence, elasticity), detail,
    )


def agreement_loss_shock(
    sector: str,
    baseline: float,
    config: dict[str, Any],
    *,
    coverage: str,
    mfn_tariff: Parameter | None = None,
    rate_basis: str | None = None,
    overrides: dict[str, float | None] | None = None,
) -> ShockOutcome:
    """Losing a preference re-imposes the MFN tariff: a tariff shock at that rate.

    `coverage` is the caller's description of what the knowledge graph found —
    the caller has the rows, and no simulation may run without them (SAD §4.1).
    `mfn_tariff` is the rate to apply when a policy document supplied one; left
    None, the configured constant (or an override of it) stands, and
    `rate_basis` defaults to the sentence saying so.
    """
    rate = mfn_tariff or parameter(config, "agreement_loss_mfn_tariff", sector, overrides=overrides)
    basis_text = rate_basis or describe_config_rate(rate)
    incidence = parameter(config, "tariff_incidence", "default", overrides=overrides)
    elasticity = parameter(config, "export_demand_elasticity", sector, overrides=overrides)

    price_change = incidence.value * rate.value
    revenue_change = elasticity.value * price_change

    detail = (
        f"{sector}: preference coverage resolved from the knowledge graph ({coverage}). "
        f"Loss modelled as an {basis_text}, with exporter incidence {incidence.value:.2f} "
        f"and demand elasticity {elasticity.value:.2f}."
    )
    return ShockOutcome(
        "agreement", sector, baseline, baseline * revenue_change, revenue_change,
        price_change, revenue_change, (rate, incidence, elasticity), detail,
    )


def describe_coverage(preferences: list[dict[str, Any]]) -> str:
    """The graph's coverage rows as the agent has always described them."""
    names = ", ".join(sorted({r["agreement"] for r in preferences}))
    verified = {r.get("agreement_verified", "unverified") for r in preferences}
    return (
        f"{names}, matched on HS {preferences[0]['matched_on']}, "
        f"status {'/'.join(sorted(verified))}"
    )


def describe_config_rate(rate: Parameter) -> str:
    """How the agent names the D9 fallback constant when no document sourced a rate."""
    return (
        f"MFN tariff of {rate.value * 100:.1f}% from config/elasticities.yaml "
        f"(a literature constant, not a queried tariff schedule — deviation D9)"
    )


def base_assumptions(config: dict[str, Any], shock: str, magnitude: float) -> list[str]:
    """SRS 3.1.5 requires these to be stated. They must never be empty."""
    model = config.get("model", {})
    assumptions = [
        f"Shock modelled: {shock}, magnitude {magnitude * 100:.1f}%.",
        f"Functional form is {model.get('form', 'linear')} over a "
        f"{model.get('horizon_months', 12)}-month horizon; effects do not compound.",
        "Elasticities are literature ranges recorded in config/elasticities.yaml, "
        "not estimates fitted to Sri Lankan data.",
    ]
    if note := model.get("note"):
        assumptions.append(str(note))
    return assumptions


__all__ = [
    "DEFAULTS",
    "OVERRIDABLE",
    "SECTORS",
    "SHOCKS",
    "Parameter",
    "ShockOutcome",
    "agreement_loss_shock",
    "base_assumptions",
    "describe_config_rate",
    "describe_coverage",
    "fx_shock",
    "parameter",
    "tariff_shock",
]
