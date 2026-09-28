# Elasticity sources: `config/elasticities.yaml`

The trade-economics agent and the scenario workbench (D17) simulate three shocks
(an LKR depreciation, a tariff, the loss of a preference) from the parameters in
`config/elasticities.yaml`. Every parameter needs a `source` (SRS 3.1.5). Since
Day 7 all seven were `TBD`. This document records the sources for each one, the
value each source supports, and how that value was mapped onto the way
`ceynex/models/shocks.py` uses the parameter.

**Researched 2026-09-28 (M2); all four proposed changes signed off by Thisen the same day and applied. Tariff incidence stays one value for now (§3's open item).**

---

## The convention trap, stated first

The literature reports "exchange rate pass-through" in two opposite conventions,
and both appear in the sources below:

- **Importer-currency pass-through**: the share of a depreciation that reaches
  the buyer's price in the buyer's currency. **This is our `fx_pass_through`**:
  `shocks.fx_shock` computes `price_change = −pass_through × depreciation` on the
  USD price.
- **Producer-currency response**: how much the exporter's own-currency price
  rises. When prices are sticky in USD, this is close to 1 and the USD price
  barely moves. **Our parameter is 1 minus this number.**

A figure of "0.84" or "0.95" means opposite things under the two conventions.
Every estimate below is converted to ours, and each conversion is stated.

---

## 1. `fx_pass_through`: share of a depreciation that reaches the USD price within 12 months

### Agriculture (tea, cinnamon, rubber): current 0.6, **proposed 0.1**

The current comment says spot-traded commodities pass *more* through than
contract-priced apparel. That gets the mechanism backwards. A commodity whose USD
price is set on a world market has a flexible *rupee* price, and that is the price
that absorbs a depreciation. For a price-taker, the law of one price leaves the USD
price where the world market puts it.

**Evidence from our own data.** The 2022 crisis is a clean, very large shock:
LKR/USD went from 198.88 (2021) to 322.63 (2022), so the rupee lost 38% of its USD
value (World Bank `fx_usd_lkr`, already in `fact_trade`). At 0.6, USD export
prices should have fallen about 23%. The UN Comtrade unit values in `fact_trade`
(local Parquet mirror, all partners) show:

| USD per kg | 2021 | 2022 | 2023 | 2021→22 |
|---|---|---|---|---|
| Tea (0902) | 4.93 | 5.30 | 5.41 | **+7.5%** |
| Cinnamon (0906) | 13.15 | 12.70 | 10.73 | **−3.4%** |
| Rubber (4001) | 2.89 | 2.71 | 2.25 | **−6.2%** |

The implied one-year pass-through is roughly 0 to 0.15. Tea is confounded by the
2021–22 fertiliser-ban supply shortfall, which pushed prices up. Rubber tracks the
world rubber price. Cinnamon, where Sri Lanka has real pricing power in true
cinnamon, falls a further 15.5% in 2023 while the rate held: a lagged, partial
pass-through. That lag is outside the model's 12-month horizon.

**Proposed: 0.1.** A price-taker's zero, nudged up for cinnamon's pricing power.
The basis becomes `own_data` rather than `literature_range`.

### Apparel: current 0.4, **proposed 0.3**

- **Gopinath, Boz, Casas, Díez, Gourinchas & Plagborg-Møller (2020), "Dominant
  Currency Paradigm", *American Economic Review* 110(3), §4.** This is the direct
  measurement for a small, USD-invoicing exporter of manufactures: firm-level
  Colombian export prices, excluding commodities, against the peso/USD rate. The
  paper measures prices **in pesos** (the producer-currency convention). The peso
  price rises 0.84 on impact and a cumulative 0.56 after two years, so the USD
  price falls 0.16 on impact and 0.44 after two years. At four quarters that
  interpolates to **about 0.3 in our convention**.
- **Imported inputs cap it.** A depreciation lowers only the rupee-cost share of a
  garment's USD cost. Sri Lankan apparel imports most of its fabric and trims, so
  even full cost pass-through could lower the USD price by at most the domestic
  value-added share. (The Gopinath–Itskhoki review, *Handbook of International
  Economics* Vol. VI ch. 2, 2022, sets out the mechanism.)
- **Why not the ~0.95 figure (Hossen 2023, arXiv:2303.04101, Bangladesh
  firm-level)?** Checked at source. It *is* in our convention: ERPT = 1 − β, with
  unit values in BDT. But it is identified with year fixed effects over 2005–13,
  when BDT/USD was stable, so it comes mostly from cross-currency moves against a
  USD invoice. A USD-invoiced good passes those through almost completely by
  construction. It measures a different shock from a taka/USD (or rupee/USD)
  depreciation, which is the only shock `fx_shock` models. Adopting it would take
  apparel's 5%-depreciation revenue effect from +0.40% to +0.95%.

**Proposed: 0.3** (range 0.16–0.44 from the same source), with basis `literature`.

---

## 2. `export_demand_elasticity`: % volume change per 1% change in the buyer's landed price

No Sri Lanka-specific estimate turned up. What the sources give is a range
bounded by horizon and aggregation:

- **Senhadji & Montenegro (1999), "Time Series Analysis of Export Demand
  Equations: A Cross-Country Analysis", *IMF Staff Papers* 46(3).** Aggregate
  export demand for 53 countries. The short-run (one-year) price elasticity
  averages **−0.21** (median −0.17). The long-run elasticity averages **−1.0**
  (median −0.76), and Asian exporters sit significantly above the average. Sri
  Lanka is not in the sample.
- **Substitution between sources.** Phillips (2024), "Upstream, Downstream: An
  Estimation of Armington Elasticities at Different Stages of Production" (USITC
  Economics Working Paper), estimates **σ ≈ 3.6–3.8** for downstream textiles and
  apparel (HS 3.557, NAICS 3.846). Ahmad, Montgomery & Schreiber (2020, USITC
  working paper) find apparel "consistently on the high end" of Armington
  estimates across the literature. A small supplier facing substitution to
  Bangladesh or Vietnam sees a demand elasticity near −σ in the long run.
- **Tea.** Own-price demand in major importing markets is reported as inelastic,
  roughly −0.1 to −0.5 (US, UK, Canada). **These figures are cited second-hand**:
  the source, a *Journal of Applied Sciences* 2006 tea-demand study, could not be
  opened to check them. Demand facing *Sri Lankan* tea specifically is more
  elastic, because Kenyan and Indian teas substitute for it.

**Proposed: keep −1.2 (apparel) and −0.8 (agriculture)**, with basis
`assumption`. Their sources state the bracket: above the one-year aggregate
(~−0.2), below the steady-state product-level substitution (about −3.6 for apparel).
The honest statement is that these are judgment calls inside a sourced range, not
estimates.

---

## 3. `tariff_incidence`: exporter's share of a tariff (D18)

- **Cirera (2014), "Who captures the price rent? The impact of European Union
  trade preferences on export prices", *Review of World Economics* 150(3).**
  Exporters' share of EU preference margins ranges from **0.17 to 0.8**, depending
  on margin size and product type. It is 0.48–0.59 where the margin exceeds 4%,
  and 0.68–0.8 where the product is also differentiated. Apparel's EU margin is
  about 12%, and apparel is differentiated.
- **Özden & Sharma (2006), *WBER* 20(2):** Caribbean apparel exporters to the US
  captured 0.647. **Olarreaga & Özden (2005), *World Economy* 28(1):** AGOA apparel
  exporters captured 0.3; smaller exporters captured less.
- **New tariffs are a different case.** US tariffs in 2018–19 passed almost
  entirely into US import prices, so the exporter's share was about 0 (Amiti,
  Redding & Weinstein 2019, *JEP*; Fajgelbaum, Goldberg, Kennedy & Khandelwal 2020,
  *QJE*; Cavallo, Gopinath, Neiman & Tang 2021, *AER: Insights*, about 0.95 at the
  border).

**Proposed: keep 0.5** with these sources. **Open question:** one `default` value
serves both `tariff_shock` (new tariff: evidence ≈ 0–0.1) and
`agreement_loss_shock` (preference loss: evidence ≈ 0.5–0.7 for apparel). Splitting
it into two keys is a small change to `shocks.OVERRIDABLE` and the workbench
slider.

---

## 4. `agreement_loss_mfn_tariff`: the EU MFN rate that returns if GSP+ is lost

Rates are from the World Bank WITS / UNCTAD TRAINS API (reporter EU 918, MFN,
2023 schedule, HS6 simple averages) and WTO *World Tariff Profiles 2019*
(European Union, Part A.2).

### Apparel: current 0.095, **proposed 0.115**

The WTO profile gives EU MFN applied duties on Clothing (ch. 61–62): **average
11.5%, max 12%, 0% duty-free lines**. WITS confirms 12% on T-shirts (6109.10),
women's and men's cotton trousers (6204.62, 6203.42), jerseys (6110.20) and
knitted underwear (6108.21). The exception is bras (6212.10) at 6.5%, which pulls
a Sri Lanka trade-weighted rate below 12%. **Proposed: 0.115**, the WTO simple
average for the product group.

### Agriculture: current 0.055, **proposed 0.0**

CeyNex's agriculture items are tea (0902), cinnamon (0906) and natural rubber
(4001). Their EU MFN rates (WITS, 2023) are:
- black tea (0902.30, 0902.40), cinnamon (0906.11/.19/.20) and natural rubber
  (4001.21/.22) are all **0%**;
- only green tea in small packs (0902.10) carries **3.2%**, a negligible share of
  Sri Lanka's exports.

Losing GSP+ therefore changes no tariff on these goods. The current 5.5% produced
a −4.95% revenue loss for a preference that is worth nothing on them. **Proposed:
0.0.** Coconut oil (1513, 2.5–12.8%) would matter, but it is not a CeyNex item.

---

## Before / after, at the configured defaults

Revenue change as % of baseline, D18 tariff arithmetic, one sector at a time:

| Sector | Shock | Now | Proposed |
|---|---|---|---|
| Agriculture | 5% LKR depreciation | −0.60% | **−0.10%** |
| Agriculture | 5-pt tariff | −4.50% | −4.50% |
| Agriculture | Preference loss | −4.95% | **0.00%** |
| Apparel | 5% LKR depreciation | +0.40% | **+0.30%** |
| Apparel | 5-pt tariff | −5.50% | −5.50% |
| Apparel | Preference loss | −10.45% | **−12.65%** |

The tariff shocks don't move because the proposal keeps incidence and the demand
elasticities. The depreciation effects shrink, which matches what 2022 actually
did to USD export prices. Agriculture's preference loss becomes zero because the
EU has no tariff to restore on tea, cinnamon or rubber.

## Limits worth stating

- The pass-through evidence is short-horizon. Cinnamon's 2023 fall suggests more
  arrives in year two, and the model's 12-month horizon doesn't capture it.
- Our own-data check is one episode (2021→22), with a supply shock on tea, and it
  uses UN Comtrade unit values, which move with product mix as well as price.
- No elasticity here is a fitted Sri Lankan estimate. The demand elasticities in
  particular stay `assumption`, now with the range they sit in written down.
