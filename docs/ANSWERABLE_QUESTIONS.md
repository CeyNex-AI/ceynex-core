# CeyNex — what the current system can answer correctly

**Compiled:** 2026-09-03
**Basis:** `ceynex-core` on `main` — the 5 agents, the router, the two graded
eval sets (`eval/questions.yaml` 30 Q, `eval/policy_questions.yaml` 15 Q), and
the 2026-08-28 evaluation runs in `docs/EVALUATION.md`.
**Live-verified:** 2026-09-03 against `https://35.200.228.142/api/query` — 23
example questions fired at the deployed backend. Corrections from that run are
folded in below and flagged **[live 2026-09-03]**. Raw results:
`scratchpad/verify_live.py`.

> ⚠️ **§9's limitations 1, 2, 6, 7 and 10 were fixed after this was written and
> are no longer accurate.** The catalogue of what CeyNex *answers* still stands;
> the list of what it gets wrong is now partly historical. What changed:
>
> | §9 item | Was | Now |
> |---|---|---|
> | 1. Cross-sector under-fan (X04/X06) | `export_analytics` alone, answered about tea at 0.9 | fans to both sector agents + `export_analytics`; also no longer flagged out-of-scope by the keyword router |
> | 2. Spurious `unanswered[]` caveat | every non-tea agriculture answer carried one | the merger drops a decline another finding already answered, and the agent's wording no longer claims "only tea is covered" |
> | 6. Forecasts run off the drift baseline | true, and served at High confidence | the apparel models can now load at all (the registry lookup was pinned to `sector="agriculture"`), and the baseline scores Moderate, not High. **Still requires the deploy step** — see `ceynex-infra/DEPLOY_ORDER.md §2b` |
> | 7. Merge prose surfaces source hedges | narrated as "a discrepancy between sources" | a scope/period difference is no longer presented as a disagreement to reconcile |
> | 10. Shipping/freight mis-worded refusal | "names a sector CeyNex does not cover" | names freight as the reason; see `BUG_router_shipping_out_of_scope.md` |
>
> **Also corrected here:** §4d's S06 cinnamon price is not a unit-handling
> problem. `annual_series()` summed the price column, and cinnamon prices in
> `fact_trade` are Comtrade per-partner unit values already in USD/kg — 71 rows
> for 2015 summing to exactly the 854.01 reported. Prices are now averaged
> (volume-weighted) and read from a single named source. Re-verify the live
> figure: it should be single-digit USD/kg, or an honest "too few observations"
> decline if FAOSTAT rows were never loaded on that host.
>
> §9 items 3, 4, 5, 11, 12, 13 and 14 are unchanged and still accurate.

This is a catalogue of the question **shapes** CeyNex handles, with concrete
example phrasings that route and answer correctly today, plus an honest list of
what it gets wrong or thin. "Correctly" here means: routed to the right agent(s),
returned figures backed by evidence, and either answered or refused as
appropriate.

### Live verification summary (2026-09-03)

| Result | Questions |
|---|---|
| Answered correctly, grounded | S02, S03, S05, S07, S08, S10, CO1, X01, X02, X09, M01, M02, M03 |
| Correct refusal / partial-with-stated-gap | S06, S12, X11, X12, P03, "who was Euler" |
| **Under-fanned** (report §9 confirmed live) | X04 — routed to `export_analytics` alone, confidence 0.9 |
| **Weaker than the report claimed** | S04 (no district data live), P08 (no tariff specifics live), P10 (ran a Japan hypothetical instead of answering Germany) |
| **Report facts corrected** | data window is **2015–2025** not 2015–2024; forecasts run off the **drift baseline**, not the registered models; coconut data **is** present |

None of the 23 fabricated a figure. Every miss was an over-hedge, an
over-confident non-answer, or a routing under-fan — not a hallucination.

---

## 1. Scope — the outer boundary

CeyNex answers questions about **Sri Lanka's merchandise exports** in two sectors
(SRS 2.4):

| Sector | Items in scope | HS codes |
|---|---|---|
| Agriculture | tea, cinnamon, natural rubber, coconut\* | 0902, 0906, 4001, 0801/1513\* |
| Apparel | knitted garments, woven garments | HS 61, HS 62 |

Plus **destination-market trade policy** for the countries whose policy documents
are in the corpus (see §6).

\* **Coconut** was added to the Comtrade pull on 2026-08-26. **[live 2026-09-03]**
It is now loaded and answers correctly — "How are Sri Lanka's coconut exports
doing and which markets buy them?" returned USD 183 M for 2025, US the largest
buyer at 26.6%, 104 markets, HHI 0.11. Every item has **2015–2025** data.

**Anything outside this returns a correct refusal, not an answer** — see §8.

---

## 2. Data actually loaded (what the figures rest on)

| Store | Contents | Coverage |
|---|---|---|
| PostgreSQL `fact_trade` | Sri Lanka export flows, reporter/partner/HS/period/volume/value | **2015–2025** [live 2026-09-03], HS 0902/0906/4001/0801/1513/61/62 |
| Neo4j graph | `(Country)-[EXPORTS_TO]-(Commodity/ApparelCategory)` edges, trade-agreement nodes (GSP+, UK DCTS, ISFTA, SAFTA, APTA) | derived from `fact_trade` |
| Forecast registry | 5 LightGBM models (tea, cinnamon, rubber, apparel_knit, apparel_woven), 2 versions each, target `export_value_usd` | 9 rows/model. **[live 2026-09-03] the deployed backend has no registered export-value model — every forecast query falls back to the drift baseline** ("mean year-on-year change + bootstrap interval") and says so in its assumptions. §10's MAPE table describes the models as evaluated locally, not the live path. |
| M1 agriculture series | tea export **volume** (Tea Board, 15 obs), cinnamon **producer price** (FAOSTAT, 34 obs) | annual |
| Qdrant policy corpus | 901 chunks from **6 documents**: Sri Lanka, UK, Canada, USA, India, Italy | trade-policy / strategy docs |

**Known holes:** 2018 is missing from Comtrade at source (every series has a
gap); no tariff-rate table (WITS ingestion was cut — deviation D9); no national
accounts / GDP model; no freight, logistics or shipping-cost data; no services
exports.

---

## 3. The five agents and the question shapes each answers

| Agent | Answers | Source |
|---|---|---|
| `export_analytics` | market share, destination concentration (HHI), growth / CAGR, rankings ("largest", "fastest-growing"), district breakdowns | Neo4j Cypher |
| `agriculture_commodity` | tea/cinnamon/rubber/coconut **price** and **export-volume** trends and their drivers; cinnamon district production share | Neo4j + M1 series |
| `apparel_manufacturing` | HS 61/62 buyer-market demand, per-partner export value, sector overview, single-buyer exposure | Neo4j Cypher |
| `trade_economics` | (a) **simulations**: FX moves, tariff changes, agreement loss (GSP+/DCTS); (b) **descriptions**: what a destination market's trade policy/strategy says | Neo4j + Qdrant corpus |
| `forecast` | forward export-value projections with 80% intervals for the 5 modelled items | LightGBM registry (+ drift fallback) |

Every answer also carries: a **confidence score** (Very low <0.30 / Low
0.30–0.50 / Moderate 0.50–0.75 / High ≥0.75), an **evidence panel** with the
literal Cypher / source pages, and — for out-of-scope parts — an explicit
statement of what was skipped.

---

## 4. Single-sector questions it answers correctly

These are the strongest category — 100% returned grounded content in the eval
run. Swap tea/cinnamon/rubber/coconut or knitted/woven freely; the intent parser
resolves the item, an optional country, an optional year, and an optional region
(continent).

### 4a. Market share / "who buys it"
- Which country takes the largest share of Sri Lanka's tea exports? *(eval S01)*
- Which markets buy the most Sri Lankan knitted apparel? *(eval S07)*
- Who are the top destinations for Ceylon cinnamon?
- What share of rubber exports goes to Germany?
- Which markets buy the most Sri Lankan woven apparel?

### 4b. Destination concentration
- How concentrated are Sri Lanka's rubber export destinations? *(eval S03)*
- How exposed is Sri Lanka's apparel sector to a single buyer market? *(eval S11)*
- Is Sri Lanka's tea export base dependent on a few countries?

### 4c. Growth / trend over time
- How fast have cinnamon exports grown over the last five years? *(eval S02)*
- Has woven apparel export value grown or fallen since 2020? *(eval S08)*
- Which importing country has shown the fastest-growing demand for Sri Lankan apparel? *(eval S09)*
- How have tea export earnings changed over the last decade?
- What is the CAGR of rubber exports to India?

### 4d. Price movement *(agriculture only)*
- What is driving the recent movement in cinnamon prices? *(eval S06)*
- How have cinnamon producer prices moved over the last few years?
- What has happened to tea export volumes recently?

> **[live 2026-09-03]** S06 answers with the price *movement* (FAOSTAT producer
> price, 2015→2025) but states plainly it has *no* information on the drivers —
> it is a trend answer, not a causal one. The reported unit (`USD/kg` at a
> value near 1,200) looks unit-suspect and matches the S06 flag in
> EVALUATION.md §1 — verify before quoting the number.

### 4e. District / regional production — **NOT available on the live system**
- ~~Which district contributes the largest share of cinnamon production?~~ *(eval S04)*

> **[live 2026-09-03]** S04 returns a correct refusal: *"no district-level
> production data recorded, and no numerical production or export share is
> available for any district."* District data is M1's to load and is not in the
> deployed graph. Routing is correct (`export_analytics` + `agriculture_commodity`);
> there is simply nothing to answer from. Do not demo district questions.

### 4f. Forecasts (see §10 for accuracy)
- Forecast tea export earnings for the next three years. *(eval S05)*
- Forecast knitted apparel export value for the next two years. *(eval S10)*
- What's the outlook for cinnamon exports next year?

> **[live 2026-09-03]** Both S05 and S10 answered (High confidence, forecast
> attached, 80% intervals) — but via the **drift baseline**, not a registered
> model; the answer says so ("No registered export-value model for this item
> yet"). S10 labels the 2025 actual as a "forecast" for 2025 — cosmetic.

---

## 5. Cross-sector questions it answers correctly

The system fans out to both sector agents plus `export_analytics` and merges into
one coherent answer (not a concatenation). **Caveat:** the broadest three-way
comparisons sometimes under-fan — see §9.

- Compare the growth of tea and apparel exports over the last five years. *(X01)*
- Which sector is more dependent on the European Union, agriculture or apparel? *(X02)*
- Do agriculture and apparel exports go to the same destination markets? *(X03)*
- Compare cinnamon and knitted apparel as export earners. *(X05)*
- Which sector recovered faster after 2020, agriculture or apparel? *(X07)*
- Compare the destination concentration of rubber and woven apparel. *(X10)*
- Forecast both tea and apparel exports and say which is expected to grow faster. *(X08)*
- Which of Sri Lanka's export sectors is most concentrated in a single market? *(X04 — **[live 2026-09-03] under-fanned**: routed to `export_analytics` alone, dropped both sector agents, answered about tea only at confidence 0.9)*
- Is Sri Lanka's export base becoming more or less diversified across sectors? *(X06 — same caveat, not re-tested live)*
- Which sector would be hurt more by losing access to the United States market? *(X09 — **[live 2026-09-03] answered well**: routed to `trade_economics`, apparel −USD 170 M / −5.7% vs agriculture −USD 31 M / −2.2%)*

---

## 6. Simulation questions it answers correctly

`trade_economics` classifies the shock (`fx` / `tariff` / `agreement` /
`policy`) and computes a revenue impact from graph baselines.

### 6a. Currency (FX) shocks — strongest simulation type
- How would a 5% depreciation of the Sri Lankan rupee affect apparel exports compared to agriculture? *(M01)*
- How would a 10% rupee depreciation change cinnamon export earnings? *(M04)*
- What happens to tea export revenue if the rupee appreciates 8%?

### 6b. Tariff shocks (magnitude supplied in the question)
- What if the European Union raised tariffs on Sri Lankan tea by 10%? *(M03)*
- If the United States raised tariffs on Sri Lankan apparel by 15%, what is the revenue impact? *(policy P14 — control case)*

### 6c. Trade-agreement loss
- What happens to apparel export revenue if Sri Lanka loses GSP+? *(M02)*
- What happens to tea export revenue if the European Union suspends GSP+? *(policy P12)*
- How much would apparel export revenue fall if the United Kingdom ended DCTS preferences? *(policy P13)*
- If the United States withdrew duty-free access for Sri Lankan knitted apparel, what tariff would apply? *(policy P11 — falls back to a 9.5% literature constant, and says so)*

### 6d. Demand shocks
- If global demand for knitted apparel fell 15%, what would that do to export revenue? *(M05)*

> **Simulation caveat:** the computed impact figure is sometimes stated in the
> prose but not repeated in an evidence entry, so it reads as "ungrounded" in
> QA (EVALUATION.md §1, grounding class 1). The number is real; the panel just
> doesn't echo it yet.

---

## 7. Destination-market policy questions (Qdrant corpus)

Answerable **only for the 6 corpus countries: Sri Lanka, UK, Canada, USA, India,
Italy.** Routing to `trade_economics` for these is 73% exact after the D10 fix;
retrieval adds +8 points of grounding.

- Which trade agreement gives Sri Lankan cinnamon preferential access to the European Union? *(P01)*
- Does the United Kingdom's trade strategy keep preferential access for Sri Lankan tea? *(P02)*
- What does India's Foreign Trade Policy say about imports from Sri Lanka? *(P03 — **[live 2026-09-03]** precise refusal: anchors on `IND-DGFT-FTP-2023`, "does not contain relevant passages". Correct, but note this is a refusal, not a substantive answer)*
- What non-tariff measures does the European Union apply to imported spices? *(P04)*
- Does the Netherlands' foreign trade policy identify Sri Lanka as a priority market? *(P05 — answers precisely "no documents held for NLD")*
- Compare the preferential access Sri Lankan apparel receives in the United Kingdom and the European Union. *(P07)*
- Which of Sri Lanka's largest apparel markets has the most restrictive import tariffs? *(P08 — **[live 2026-09-03] weaker than expected**: names the markets (US, UK, IT, DE, NL) but returns "the findings do not specify which... has the most restrictive tariffs" at confidence 0.8. The corpus did not supply tariff specifics. Treat as partial.)*
- Which sector, agriculture or apparel, is more exposed to losing European Union preferences? *(P09)*

---

## 8. Questions it handles correctly by **refusing** (this is the right answer)

CeyNex is graded 100% (3/3) on refusing the unanswerable in the main set. A
correct refusal names the gap and answers whatever part it can. **[live
2026-09-03]** S12, X11, X12 and "who was Euler" all refused correctly; M06, P06,
P15 were not re-tested live.

| Question | Correct behaviour |
|---|---|
| What were Sri Lanka's tea exports in 2035? *(S12)* | "That period is not in the record" — or an explicit forecast with an interval; never a figure stated as history |
| How does Sri Lanka's tea sector compare with its fisheries sector? *(X11)* | Answer for tea, state that fisheries is outside scope |
| Which is larger, Sri Lanka's apparel exports or its software services exports? *(X12)* | State that services exports are not held; no comparison |
| Which sector should Sri Lanka prioritise for the next decade, and by how much will that raise GDP? *(M06)* | Answer the export comparison; state the GDP figure cannot be estimated |
| What tariff will the United States apply to Sri Lankan tea in 2030? *(P06)* | "No document states a 2030 rate" — do not present today's schedule as 2030 |
| How do Japan's tariffs on Sri Lankan tea compare with Germany's? *(P10)* | Answer for Germany, state nothing is held for Japan. **[live 2026-09-03]** partial miss: said "no specific data comparing Japan's and Germany's tariffs" (good — no fabrication) but then ran a hypothetical Japan tariff sim and gave no Germany specifics. No hallucination, but not the intended partial answer. |
| By how much would GDP grow if every top-10 destination removed all tariffs? *(P15)* | Give per-sector export effects; state GDP cannot be modelled |
| General-knowledge / small talk ("who is Euler", "what's 4+4") | "This question does not name anything CeyNex covers" + scope statement |

---

## 9. What it does **not** answer reliably (known limitations)

Be cautious quoting these; they are documented failures, not surprises.

1. **Broad three-way cross-sector comparisons under-fan.** **[live 2026-09-03]**
   X04 ("most concentrated sector") confirmed: routed to `export_analytics`
   alone, sectors `["cross_sector"]`, answered about tea only, confidence 0.9.
   (EVALUATION.md §1)
2. **`agriculture_commodity` injects a spurious caveat into `unanswered[]` on
   any non-tea agriculture question.** **[live 2026-09-03]** S03 (rubber), CO1
   (coconut), X02 and M01 (apparel) all returned a *correct* answer from
   `export_analytics` **plus** an `unanswered` line like *"No sourced export
   volume series is available for rubber… only tea is covered."* The frontend
   renders that array as a limitation note, so a correct answer looks
   half-failed. The agent should stay silent when another agent covered the
   question. Worth a bug ticket of its own.
3. **Simulation figures** — EVALUATION.md §1 flags these as sometimes missing
   from the evidence panel. **[live 2026-09-03]** in this run M01/M02/M03 all
   stated their impact figures in the prose; still verify the panel echoes them.
4. **`agriculture_commodity` refuses production and substitution claims by
   design** — e.g. "how much tea does Sri Lanka produce?" or "is rubber
   substituting for tea?" get a decline, not an estimate. Only price and
   export-volume trends are supported.
5. **District / sub-national production questions** — no data on the live graph
   (S04). Correct refusal, but not answerable.
6. **Forecasts run off the drift baseline on the live backend** — no registered
   model is loaded, so §10's MAPE figures are not what the deployed system uses.
7. **Merge prose surfaces source-reconciliation hedges.** **[live 2026-09-03]**
   S07, S08, X02 answers contain phrases like *"two different analyses reporting
   USD X and USD Y, indicating a discrepancy between sources"* (Comtrade 2025 vs
   EDB 2024). The headline answer is right; the prose is noisier than a demo
   wants.
9. **Coconut** — **[live 2026-09-03]** now loaded and answering; the earlier
   "verify coverage" caveat is lifted.
10. **No shipping / freight / logistics cost data** — questions about shipping
    costs currently get a (mis-worded) out-of-scope refusal. See
    `docs/BUG_router_shipping_out_of_scope.md`.
11. **Policy questions outside the 6 corpus countries** (Netherlands, Germany,
    Japan, Australia, China, …) — refused for lack of a document, which is
    correct behaviour but limits coverage.
12. **No tariff-rate lookup** — a tariff *simulation* needs the rate in the
    question, or it falls back to a literature constant (9.5% MFN).
13. **Single-sector latency breaches the 10 s SRS budget** (p95 ~15–29 s) and can
    exceed the API's 25 s timeout on the slowest LLM calls. **[live 2026-09-03]**
    this run was faster — single-sector answers came back in 5–23 s, cross-sector
    up to 30 s; still over budget on the tail.
14. **Grounding check is string-based** — it catches invented numbers but not a
    correctly-sourced number attached to the wrong claim (see S06).

---

## 10. Forecast accuracy (for §4f / §6c answers)

> **[live 2026-09-03] the deployed backend does not use these models** — it
> falls back to a drift baseline for every forecast (see §2). The table below is
> the local model evaluation from `docs/EVALUATION.md §3`, kept for reference
> and for when the registry is loaded onto the VM.

Backtest MAPE, 3 rolling-origin folds, 9 annual observations per item:

| Item | Best model | MAPE | 80% interval coverage |
|---|---|---:|---:|
| tea | SARIMA(1,1,0) | **5.3%** | 1.00 |
| cinnamon | SARIMA(1,1,0) | **6.3%** | 1.00 |
| apparel (woven) | LightGBM | **14.1%** | 1.00 |
| apparel (knit) | LightGBM | **15.8%** | 0.67 |
| rubber | SARIMA(1,1,0) | **21.9%** | 0.67 |

Rubber is the weak one and should be presented with that caveat. Small sample
(9 rows, 3 folds) — the honest answer if asked about reliability.

---

## 11. Phrasing tips (to keep a question on the answerable path)

- **Name the commodity or "apparel / knitted / woven"** — a question with no
  in-scope noun ("how are exporters doing?") falls to the out-of-scope path.
- **Name a magnitude for a shock** — "raise tariffs by 10%", "5% depreciation".
  A tariff question with no rate has nothing to compute.
- **Use a real destination country** for policy questions, and one of the 6 in
  the corpus for "what does its policy say".
- **Ask for a forecast explicitly** ("forecast", "outlook", "next N years") to
  reach the forecast agent rather than a historical-trend answer.
- **A future year with no "forecast" cue** (e.g. "tea exports in 2035") is
  treated as unanswerable-as-history — which is correct.

---

## 12. Quick self-check commands

```bash
cd ceynex-core
python -m ceynex.orchestrator.demo "Which country takes the largest share of Sri Lanka's tea exports?"
python -m ceynex.orchestrator.demo --no-llm "..."   # degraded (no-LLM) path
make eval            # 30-question set, LLM live
make eval-policy     # 15-question policy set
```

`eval/questions.yaml` and `eval/policy_questions.yaml` are the 45 canonical
graded questions — every example above is drawn from them or is a same-shape
variant.
