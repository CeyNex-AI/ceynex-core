# Evaluation — M2, Core Systems and Orchestration

Measured on **2026-08-28** against PostgreSQL 18 and Neo4j 5.26 holding 4,625
`fact_trade` rows and 4,625 `EXPORTS_TO` edges of live UN Comtrade data covering
2015–2024 for HS 0902, 0906, 4001, 61 and 62.

> ⚠️ **These figures predate the production re-ingest and do not describe the
> deployed system.** Verified live on 2026-09-03: every answer from the backend
> VM now reaches **2025** (tea export value USD 1,431,567,471 for 2025, "141
> markets in 2025"), so `fact_trade` there covers **2015–2025** and holds more
> than the 4,625 rows above. Coconut (HS 0801/1513) was also added to the
> Comtrade pull on 2026-08-26 and is loaded there. A local checkout still reads
> 4,625 rows over 2015–2024 unless you re-run `make ingest`, so treat every
> number in this document as measuring the local stack on the date given, not
> production. Re-measure before quoting any of it as current:
>
> ```sql
> SELECT count(*), min(period_start), max(period_start) FROM fact_trade;
> ```
>
> §3's model accuracy table needs the same caution for a different reason — see
> the note there.

**This supersedes the 2026-08-19 run**, which is not comparable to it. Three
things changed in between, and the differences below are mostly attributable to
them rather than to the orchestrator:

1. **Both sector agents were stubs on 19 Aug.** M1's `agriculture_commodity` and
   M3's `apparel_manufacturing` landed 26 Aug. The old run measured a system two
   agents smaller.
2. **No LLM key was configured on 19 Aug; one is now**, plus an OpenRouter
   free-tier failsafe. Every latency figure in the old run was the SRS 3.4.3
   degraded path, which is why it posted 1.36 s single-sector p50 and this run
   posts 8.4 s.
3. **A refusal metric was wrong.** See §2.

Reproduce with:

```bash
make eval            # 30 questions, LLM live
make eval-degraded   # the same set, SRS 3.4.3 path
make backtest SECTOR=agriculture ITEM=cinnamon
```

**Read the denominators.** Several rates here rest on 3 observations. They are
reported with their `of` counts throughout because "100% correct" out of three
is a weaker claim than the percentage implies, and rounding that away would be
the most misleading thing in this document.

> **Clear `.cache/llm` before any run whose latency you intend to quote.**
> `LLMReasoningClient` keeps a content-addressed prompt cache on disk with a
> 168-hour TTL, so a second run of the same 30 questions serves them from disk.
> Measured 2026-08-28: the cached re-run returned fluent prose at a p50 of
> **65 ms** and reported `within_budget: true` everywhere. Those are cache-hit
> times, not response times, and they would have been the most flattering and
> most wrong numbers in this document. Every latency figure below is from a
> cold cache.

---

## 1. Orchestrator: the 30-question set

`eval/questions.yaml` holds 30 questions with their expected agent routes — 12
single-sector, 12 cross-sector, 6 simulation, of which 3 are unanswerable and 1
is partially answerable. **The file was committed before the harness was ever
run** (commit `1b4d831`, preceding the harness at `f209d02`), because questions
written after watching the system answer them describe the system rather than
test it.

### Headline

| Metric | LLM live | Degraded (SRS 3.4.3) | Denominator | 19 Aug |
|---|---|---|---|---|
| Questions completed without crashing | **100%** | 100% | 30 | 100% |
| Routing — exact agent-set match | **53.3%** | 40% | 30 | 40% |
| Routing — recall of expected agents | **0.856** | 0.881 | 30 | 0.88 |
| Routing — never returned an empty route | **100%** | 100% | 30 | 100% |
| Answers with every figure traceable to evidence | **77.8%** | 92.6% | 27 answerable | 92.6% |
| Ungrounded figures across the whole run | **6** | 2 | — | 2 |
| Mean evidence entries per answer | **3.67** | 4.48 | 27 | 3.67 |
| Answers with no evidence at all | **2** | 3 | 27 | 0 |
| Unanswerable questions correctly refused | **100%** | 100% | 3 | 100% |
| Answerable questions returning no content | **0%** | 0% | 27 | 0% |

The degraded column is not a worse version of the live one. It is better on
grounding and evidence density and worse on routing, and both differences have
the same cause: the deterministic composer restates agent summaries verbatim,
so every figure it prints is by construction one an agent produced, while the
keyword router alone cannot resolve the queries §1's routing discussion covers.

### Latency (SRS 3.4.1) — one budget is breached

| Query type | p50 | p95 | Budget | Within budget |
|---|---:|---:|---:|---|
| Single-sector | 8,397 ms | **14,638 ms** | 10,000 ms | **no** |
| Cross-sector | 6,076 ms | 17,394 ms | 20,000 ms | yes |
| Simulation | 7,658 ms | 9,557 ms | 20,000 ms | yes |

**Single-sector p95 is 46% over its budget.** The 19 Aug run passed every budget
with 5–15× headroom for one reason: no LLM key was configured, so it never paid
for a model call. That headroom was never real; it was the degraded path being
measured and reported as though it were the system.

Two consequences worth stating plainly:

- **The API would not merely be slow on these queries, it would fail.**
  `ceynex/api/routes/query.py` sets `REQUEST_TIMEOUT_S = 25.0`. The slowest
  query in the earlier of the two cold runs took 29.0 s, which is past that
  wall. Latency variance between the two cold runs was substantial (single-sector
  p95 of 29.0 s and 14.6 s on the same 30 questions), so the tail is not stable
  and one run is not enough to characterise it.
- **The budget is the requirement, not the p50.** Single-sector p50 (8.4 s) is
  inside 10 s, and quoting only that would pass a requirement the system fails.

This is a single-user measurement. SRS 3.4.2's 50 concurrent users remains
untested and is recorded in [DEFERRED.md](DEFERRED.md); rate limiting (SRS
3.4.6) now caps any one caller at 30 queries/minute, so that figure means 50
distinct users rather than one script.

### Routing: 53.3% exact, and why that is not the whole story

Exact match counts a route as correct only if the agent set matches the
pre-written label exactly. Recall — did the router include every agent that was
needed — is **0.856**. The two move in opposite directions from 19 Aug (exact
40% → 53.3%, recall 0.881 → 0.856), and that is the LLM router doing what a
sharper router should: fanning out less.

Eight of the fourteen non-matches are still the router adding one *more* agent
than the label listed:

```
S07  expected [export_analytics]  ->  [apparel_manufacturing, export_analytics]
S08  expected [export_analytics]  ->  [apparel_manufacturing, export_analytics]
```

"Which markets buy the most Sri Lankan knitted apparel?" is labelled as pure
export analytics; the router also sends it to the apparel agent. That is
defensible and arguably better than the label. **The labels were not edited to
match** — doing so after seeing the results is precisely the failure the
pre-commit was meant to prevent.

The remaining six are misses, and the shape of them has changed. On 19 Aug the
misses were shocks the keyword list could not name. Now they are the opposite —
the router **drops** agents a broad comparison needs:

| Id | Question | Dropped |
|---|---|---|
| X04, X06 | cross-sector comparisons labelled for all three of `agriculture_commodity`, `apparel_manufacturing`, `export_analytics` | routed to `export_analytics` alone |
| X09 | "Which sector would be hurt more by losing access to the United States market?" | `agriculture_commodity`, `trade_economics` |
| M01, M06 | multi-agent simulation questions | `export_analytics` |
| S12 | "What were Sri Lanka's tea exports in 2035?" | routed to `forecast`, which is arguably right for a future year |

X09 was a miss on 19 Aug too and remains one. **Under-fanning is the more
dangerous error**: an extra agent costs latency and produces a defensible
answer, while a dropped one produces a confident answer to half the question.
X04 and X06 are the cases to fix first.

### Evidence grounding — the number that got worse

Every figure appearing in a merged answer is checked against the claims and
Cypher of the `Evidence` entries attached to it. **77.8% of answers are fully
grounded, with 6 ungrounded figures**, against 92.6% and 2 on the degraded path.
Switching prose generation on is what cost the 15 points, and the two failure
classes behind it are different problems needing different fixes:

**1. A computed figure the agent never put in its evidence (M01–M05).** M02
answers "apparel export revenue is expected to decrease by approximately USD
161,815,198", and that number appears in no evidence entry — it is a
`trade_economics` simulation output living in the agent's `figures` dict. This
is the same defect fixed for the forecast agent in `e0ed5ac`, recurring in the
simulation agents. **The fix is in the agents, not the merger:** an agent that
computes a figure must restate it in an `Evidence` entry, or the answer cannot
be checked.

**2. A sourced number attached to a wrong claim (S06).** The answer says
cinnamon prices show "a 12.5% rise over this period" and then "an increase of
106.8 USD/kg over nine years" — against a series whose 2024 value is 6.81
USD/kg. The two sentences contradict each other and the second is not a price
movement at all. This one matters more than its single-figure weight suggests,
because it is the limitation §5 has always claimed and never demonstrated: the
runtime guard (`ceynex/orchestrator/grounding.py`) compares digit strings and
**cannot** see that a plausible number has been given the wrong unit and the
wrong claim. Verify S06 by hand before quoting anything from it.

The check is deliberately crude and over-reports: it compares digit strings, so
a figure rounded differently in prose than in evidence is flagged. For a metric
whose job is catching hallucinated numbers, a false alarm costs a manual check
and a miss costs the claim.

**Two answers carried no evidence at all** (X03, M06), against 0 on 19 Aug.
X03 is the one answer in the run that fell back to degraded mode mid-flight;
M06 is the routing miss above. Both are regressions worth chasing before the
Testing and Evaluation Document.

### Degraded mode (SRS 3.4.3)

The same 30 questions with the LLM forced unavailable: **0 crashes, all 30
flagged `degraded=True`, 100% of unanswerable questions refused, and grounding
*higher* at 92.6%.** Three answers carry no evidence.

The degraded path is not a fallback that limps. On every metric except routing
it is the equal or better of the live path, because a deterministic composer
cannot invent a figure and cannot misattribute one. What it loses is fluency and
the LLM router's precision.

---

## 2. What the evaluation changed

### 2026-08-28: the harness was measuring its own vocabulary

The 28 Aug run scored refusals at **100% in degraded mode and 33% with the LLM
composing** — the same system, the same 30 questions, the same three
unanswerable ones. Nothing about the behaviour differed. `is_refusal` grepped
the answer for decline phrasing against a fixed marker list, and that list had
been written from the deterministic composer's wording. X11 answered:

> "…data for the fisheries sector is not available for comparison, so a direct
> comparison between the tea and fisheries sectors cannot be made."

That is a textbook correct refusal and it matched no marker. The metric was
punishing paraphrase.

`merger._gap_already_stated` records the identical lesson from the other side,
twice, on 26 and 27 Aug — a fixed vocabulary cannot keep up with open-ended LLM
paraphrasing. The harness had the same bug and nobody had looked, because with
no LLM configured the deterministic composer was the only thing it ever scored.

**Fixed by asking the orchestrator instead of the prose.** `merge()` already
computes what it could not cover; `demo.answer` now returns that `unanswered`
list through the same helper the API route uses, and `is_refusal` reads it. The
marker list survives only as a fallback for older `--json` dumps. Refusal is
**100% (3 of 3)** on both paths once measured this way.

The general point is worth keeping for the report: **a metric written against
one implementation of a component silently becomes a measurement of that
implementation.** The fix was not a better keyword list.

### 2026-08-19: four defects on the harness's first run

All four are fixed, and the before/after is the clearest evidence that the
evaluation did work rather than just describing a system that already passed.

| | v1 baseline | after fixes |
|---|---:|---:|
| Routing recall | 0.647 | **0.881** |
| Answers with no evidence at all | 9 of 30 | **0** |
| Mean evidence per answer | 1.96 | **3.67** |
| Fully grounded answers | 81.5% | **92.6%** |
| Unanswerable correctly refused | 66.7% | **100%** |

**1. Nine of thirty answers never touched the graph** (commit `8563476`). The
keyword router added `export_analytics` only when an explicitly analytical word
appeared. Any question naming a sector went to the sector agent alone — and
those are M1's and M3's, still stubs — so it returned in about 2 ms with no
figures and no evidence. "Which markets buy the most Sri Lankan knitted apparel?"
is answerable from the graph today and was answering nothing.

**2. Out-of-scope questions were answered without saying what was skipped**
(commits `8563476`, `8de3b41`). Two bugs stacked. The router suppressed its
out-of-scope flag whenever the query *also* named a covered sector — exactly the
mixed case that most needs flagging — and the merger never read the flag back
out of state. "How does Sri Lanka's tea sector compare with its fisheries
sector?" returned a confident tea answer that never mentioned fisheries. A
reader had no way to tell half the question was dropped. **Omitting the limit is
the same failure as inventing the figure**, and it is the one that looks fine.

**3. Forecast figures were traceable to nothing** (commit `e0ed5ac`). The
forecast evidence named the model that produced the numbers but never restated
them, so every forecast figure in an answer counted as ungrounded — 12 of the 14
ungrounded figures in the baseline run.

**4. Two models registered in the same second silently overwrote each other.**
Version stamps are second-granular, and a loop over model families collides.
Found by the smoke test, not by the harness, but the same class of thing: it
failed by losing data quietly rather than by raising.

---

## 3. Forecast models

Rolling-origin backtest, expanding window, 3 folds at horizon 1, on the real
Comtrade series (9 annual observations per item; **2018 is absent from Comtrade
at source**, not dropped by the pipeline).

> ⚠️ **This table describes the models as evaluated locally. It is not the
> accuracy of the deployed system.** Verified live 2026-09-03: the backend VM has
> no registered model at all, so every forecast it serves is the drift baseline
> ("mean year-on-year change + bootstrap interval") and says so in its
> assumptions — S05 (tea) and S10 (knit apparel) both confirmed. The artifacts
> under `models/` are git-ignored, excluded from the build context, and excluded
> from the deploy tar, so they reach a VM only through the `models_data` volume.
> See `ceynex-infra/docs/12-deployment.md` for the step that puts them there.
>
> Quoting 5.3% MAPE for tea while the live path is a drift baseline is the
> specific claim to avoid. Check what a host actually holds with
> `GET /admin/models`.

| Sector | Item | Model | MAPE | RMSE (USD) | Coverage |
|---|---|---|---:|---:|---:|
| agriculture | tea | SARIMA(1,1,0) | **5.3%** | 72,716,541 | 1.00 |
| agriculture | tea | LightGBM | 5.7% | 81,859,333 | 0.67 |
| agriculture | cinnamon | SARIMA(1,1,0) | **6.3%** | 15,883,043 | 1.00 |
| agriculture | cinnamon | LightGBM | 12.2% | 31,677,535 | 1.00 |
| agriculture | rubber | SARIMA(1,1,0) | **21.9%** | 8,196,694 | 0.67 |
| agriculture | rubber | LightGBM | 30.3% | 9,554,970 | 0.67 |
| apparel | apparel_knit | LightGBM | **15.8%** | 643,264,054 | 0.67 |
| apparel | apparel_knit | SARIMA(1,1,0) | 25.7% | 910,240,924 | 0.33 |
| apparel | apparel_woven | LightGBM | **14.1%** | 282,366,222 | 1.00 |
| apparel | apparel_woven | SARIMA(1,1,0) | 18.6% | 382,489,539 | 0.67 |

**No family wins everywhere, which is why both are in the registry.** SARIMA
takes agriculture; LightGBM takes both apparel categories, by 10 and 4.5
percentage points. Because of that split the forecast agent selects by score
rather than by recency (`load_best`) — before that change it served whichever
model happened to be saved second, which on cinnamon meant serving 12.2% MAPE
when 6.3% was on disk.

Three readings worth making explicitly:

**Rubber is the hard case at 21.9%**, and it should be. Natural rubber is a
volatile world-priced commodity and Sri Lanka is a small producer in it; a
short annual series has little to go on. Reporting it next to tea's 5.3% is
more informative than reporting an average of the two.

**Coverage is the number to distrust here.** It is computed over 3 held-out
observations, so it can only take the values 0, 0.33, 0.67 or 1.00. Nominal is
0.80, which is not even attainable. `apparel_knit` under SARIMA at 0.33 is a
real signal of overconfident intervals; every 1.00 in the column means "all
three landed inside", not "well calibrated".

**LightGBM fits year-on-year changes, not levels.** Trees cannot predict outside
their training range, so fitted on levels the model returned a flat line the
moment a series trended past its own history — MAPE 0.12 and coverage 0.00 on a
synthetic trending series. Differencing is what makes this family usable at all,
and `test_the_boosted_model_can_forecast_above_its_training_range` fails if
anyone reverts it.

---

## 3A. M1 agriculture source-series evaluation

Measured on 2026-08-21 from the M1 dated raw snapshots, rather than the
partner-level UN Comtrade series in §3. These are different targets and should
not be compared as if they were the same experiment.

### Sufficiency decision

| Target | Source | Frequency | Observations | Window | Decision |
|---|---|---:|---:|---|---|
| Tea export volume | Tea Board total exports | annual | 15 | 2011–2025 | Short annual series: baselines first |
| Cinnamon producer price | FAOSTAT USD producer price | annual | 34 | 1991–2024 | Short annual series: baselines first |

Both series are below the plan's 40-observation threshold. Backtests use an
expanding window with three one-year-ahead folds, not a random split. Coverage
has only three held-out observations, so values of 0.33, 0.67, and 1.00 are
coarse diagnostics rather than calibrated probability estimates.

### Results

| Target | Model | MAPE | RMSE | 80% interval coverage | Selected |
|---|---|---:|---:|---:|---|
| Tea export volume (kg) | annual naive | **3.18%** | 8,550,826 kg | 1.00 | yes |
| Tea export volume (kg) | annual drift | 3.95% | 11,907,238 kg | 1.00 | no |
| Tea export volume (kg) | SARIMA/ETS | 6.30% | 20,160,918 kg | 0.67 | no |
| Cinnamon producer price (USD/kg) | annual naive | **12.05%** | 1.172 USD/kg | 0.33 | yes |
| Cinnamon producer price (USD/kg) | annual drift | 13.33% | 1.316 USD/kg | 0.33 | no |
| Cinnamon producer price (USD/kg) | SARIMA/ETS | 12.32% | 1.191 USD/kg | 0.33 | no |
| Cinnamon producer price (USD/kg) | GBM | 15.91% | 1.605 USD/kg | 0.33 | no |

GBM is retained only when it improves MAPE by at least 5% relative to the best
simple candidate. It does not meet that threshold, so the annual naïve baseline
is selected for both targets.

### Fold-level forecast-error analysis

`python -m eval.agriculture_forecast_errors` writes the three held-out
predictions behind each selected annual-naïve model to a local, git-ignored JSON
record. The following values were reproduced from the dated M1 snapshots on
2026-09-07. Each fold trains only through the stated prior year and predicts the
next year; it is not a random split.

| Target | Test year | Actual | Forecast | Absolute percentage error | Inside 80% interval? |
|---|---:|---:|---:|---:|---|
| Tea export volume (kg) | 2023 | 241,912,000 | 250,191,000 | 3.42% | yes |
| Tea export volume (kg) | 2024 | 245,787,000 | 241,912,000 | 1.58% | yes |
| Tea export volume (kg) | 2025 | 257,440,000 | 245,787,000 | 4.53% | yes |
| Cinnamon producer price (USD/kg) | 2022 | 9.9371 | 11.2900 | 13.61% | no |
| Cinnamon producer price (USD/kg) | 2023 | 8.9257 | 9.9400 | 11.36% | yes |
| Cinnamon producer price (USD/kg) | 2024 | 10.0533 | 8.9300 | 11.17% | no |

Tea's largest held-out miss was 11,653,000 kg in 2025 (4.53%); all three
actuals were inside the model's 80% intervals. This is **not** evidence of a
calibrated 100% coverage rate: with only three folds, it can only indicate that
the intervals were wide enough for these three outcomes. Cinnamon's 2022 price
fall and 2024 rebound were both outside the intervals, giving 1/3 coverage
against the nominal 80%. This undercoverage is a reason to present the interval
as a limitation, not a guarantee.

Forecast confidence now includes an interval-coverage penalty in addition to
the existing MAPE, training-observation, and staleness terms. For valid coverage
`c < 0.80`, the penalty is `min(0.15, 0.30 * (0.80 - c))`; it is zero at or
above nominal coverage. Thus cinnamon's 1/3 coverage reduces its self-reported
forecast confidence by **0.14**. A model without a valid coverage metric is
penalised by 0.10 rather than assumed calibrated. The forecast evidence and
assumptions state this limitation whenever the penalty applies.

### Cinnamon benchmark limitation

The Liyanage/Silva/Marasinghe purchasing-price panel is unavailable. Therefore
the cinnamon result above uses FAOSTAT's annual USD producer-price fallback and
is **not a reproduction of the published benchmark**. Any comparison in the
report must quote the paper's reported MAPE with this target, frequency, and
source difference stated beside it; it must not imply the same train/test split
or data were used.

### Agriculture agent end-to-end smoke evaluation

Measured on **2026-09-06** against the local PostgreSQL `fact_trade` records,
Neo4j graph, and registered M1 models. `python -m eval.agriculture_agent_e2e`
runs five representative questions directly through the Agriculture & Commodity
agent and writes its full local JSON record under `eval/results/`. It forces the
LLM unavailable to make the run repeatable and to exercise SRS 3.4.3; therefore
these are agent-level deterministic/degraded results, **not** a substitute for
the orchestrator's 30-question evaluation.

| Check | Required behaviour | Result |
|---|---|---|
| Cinnamon trend | Source-backed price trend with figures and two evidence records | Pass: 10.05 USD/kg in 2024, up 382.8% from 1991 |
| Cinnamon forecast | Registered producer-price forecast with an 80% interval | Pass: 2025 point forecast 10.05 USD/kg; 8.96-11.15 interval; annual-frequency caveat stated |
| Cinnamon districts | Do not invent a largest district without a sourced share | Pass: lists Matara, Galle, and Ratnapura; explicitly refuses a largest-share claim |
| Tea export trend | Tea Board export-volume trend with figures and two evidence records | Pass: 257,440,000 kg in 2025, down 20.3% from 2011 |
| Tea-to-rubber substitution | Do not infer a relationship without evidence | Pass: explicitly reports that the effect cannot be estimated responsibly |

All **5 of 5** checks passed, with a mean of **2.0 evidence records** per
answer. Each answer was correctly marked `degraded=True`, because no LLM prose
was requested. The two refusal cases are passes, not missing functionality:
they show the agent preserves evidence boundaries instead of manufacturing a
district share or substitution effect.

### Agriculture cross-source validation

`python -m eval.agriculture_validation --write-flags` validates only
semantically equivalent, connector-normalised annual export-volume totals. It
aggregates partner-level UN Comtrade rows to a national total, keeps an existing
Tea Board or DEA/EAC world-total row as-is, and compares tea (`TEA_BOARD` vs
`UN_COMTRADE`) and cinnamon (`CINNAMON` vs `UN_COMTRADE`) by item and year.
The run is non-destructive: source facts are only read, and only material
(5-20%) or severe (>20%) discrepancies are inserted into `dq_flag`. Exact
existing flags are not inserted twice.

The run measured on **2026-09-07**, after the documented 2015--2024 UN
Comtrade import, found 121 FAOSTAT, 15 Tea Board, 5 Cinnamon, 2,413 UN
Comtrade, and 0 EDB agriculture facts. It evaluated 12 overlapping annual
commodity-source pairs: 11 were minor differences, one was material, and none
were severe. The material finding was tea export volume for 2020: Tea Board
reported 265,569,000 kg and the partner-aggregated UN Comtrade total was
279,710,426.42 kg (5.32\% difference). The run inserted this one material
finding as a `dq_flag`; source facts were not altered. Repeating the command
does not insert the same flag again.

FAOSTAT's current `fact_trade` rows are producer prices, so they are not
compared with export volumes; Pink Sheet is an auction-price series and is
likewise not an export-volume comparator. The configured EDB connector is
apparel-only. WITS tariff ingestion remains deliberately deferred and is
reported as unavailable rather than treated as validated agriculture data.

### Agriculture testing and evaluation record

The agriculture checks are intentionally separated by failure type so a passing
unit test cannot be mistaken for a validated external figure:

| Evidence | Reproducible command or scope | Outcome on 2026-09-07 |
|---|---|---|
| Forecast-error analysis | `python -m eval.agriculture_forecast_errors` | 6 held-out predictions recorded; no source or model artifact changed |
| Model selection | `tests/models/agriculture/test_evaluation.py` and `tests/models/agriculture/test_baseline.py` | annual-naïve selected for tea and cinnamon; interval contains each point forecast |
| Backtest rules | `tests/eval/test_backtest.py` | expanding windows, error metrics, and interval coverage checked |
| Agent behaviour | `tests/eval/test_agriculture_agent_e2e.py` | five planned questions passed in deterministic degraded mode; see the smoke-evaluation table above |
| Cross-source validation | `tests/eval/test_agriculture_validation.py` and `python -m eval.agriculture_validation --write-flags` | 12 comparable pairs; 11 minor, 1 material, 0 severe; one material flag retained without altering facts |

These checks do not validate WITS or reproduce the unavailable published
cinnamon purchasing-price benchmark. Those are explicit deferred/limitation
states, rather than passing results.

### Registry release procedure

The selected models are registered only from a clean, committed checkout:

```bash
python -m ceynex.models.agriculture.evaluation --register
```

This writes one local (git-ignored) versioned artifact for each of
`agriculture/tea/export_volume` and `agriculture/cinnamon/producer_price`.
Its `metadata.json` records the annual training window and row count, source,
`AnnualNaiveModel` class, all three-fold rolling-origin metrics, 80% interval
level, and the exact Git SHA. Registration fails if either target does not
select the annual-naïve model, its forecast interval excludes the point, or the
checkout is dirty; recording a SHA that cannot reproduce the artifact would be
misleading.

## 4. Merge coherence — not yet measured

SRS 3.1.2 forbids answers that concatenate per-agent responses, and no automated
metric can detect the failure: "The agriculture agent says X. The apparel agent
says Y." is well-formed, correctly routed, fully grounded, and exactly the thing
the requirement prohibits.

`eval/coherence.py` produces the instrument — blind rating sheets with agent
attribution stripped, question ids replaced by opaque labels, and rows shuffled
under a fixed seed so a rater cannot infer how many agents contributed and score
the machinery instead of the prose:

```bash
python -m eval.coherence sheet --results results.json --out coherence_sheet.csv
python -m eval.coherence score r1.csv r2.csv r3.csv --key coherence_sheet.key.json
```

**This requires three human raters and has not been run. It is now unblocked** —
it was waiting on the two sector agents, which landed 26 Aug, and on prose
generation, which is on. It is the longest-lead item left before the Testing and
Evaluation Document (activity 084, due 20 Sept), because it needs three people's
calendars rather than a command. Book it first and score it later.

The scoring reports
inter-rater spread alongside the mean, because three raters agreeing on 4 and
three splitting 2/4/5 produce nearly the same average and mean entirely
different things. If it is run with fewer than three raters the tool warns, and
the rater count must be recorded here as a limitation.

---

## 5. Threats to validity

Stated because a marker will find them anyway, and finding them stated is a
different conversation from finding them hidden.

- **Three folds, three held-out observations per model.** Every MAPE in §3 is an
  average of three errors. The ranking between families is consistent enough
  across five items to be worth reporting; any individual figure is not precise
  to the decimal place shown.
- **All five agents are now built**, so this run measures the whole system for
  the first time. The 19 Aug figures do not describe the same software and
  should not be presented as a trend against these.
- **Latency rests on two cold runs that disagree.** Single-sector p95 came out
  at 29.0 s and 14.6 s on the same 30 questions. Both breach the 10 s budget, so
  the conclusion is stable, but the magnitude is not — quote it as "breaches,
  p95 15–29 s across two runs", not as a single number. LLM latency is the
  dominant term and it is not under our control.
- **Every latency figure requires a cold `.cache/llm`.** A cached re-run posts a
  65 ms p50 and passes every budget. See the note at the top.
- **One run, one machine, one user.** These were taken against a local
  `make up` stack, not the deployed VMs, so they include no VPC hop.
- **The expected routes are one person's judgement**, written in advance but not
  reviewed by the other two members. Several disagreements in §1 are arguably
  the label being wrong rather than the router.
- **The grounding check is string-based**, not semantic. It catches invented
  figures. It would not catch a correctly-sourced figure attached to a wrong
  claim.
- **Comtrade has no 2018.** Every series here has a hole in it. CAGR endpoints
  spanning 2018 and any fold boundary near it are affected; the pipeline warns
  rather than interpolating.

---

## 6. Where this sits against the plan

The M2 plan names three things that may never be cut: the orchestrator, the
single-LangGraph-graph constraint, and the 30-question evaluation. All three are
done. `tests/orchestrator/test_graph.py` asserts the single-graph constraint
structurally — five nodes, all present, all reachable — rather than by
inspection.

Outstanding and named in [DEFERRED.md](DEFERRED.md): the 50-concurrent-user
load test, WITS tariff ingestion (cut, deviation D9), the coherence rating
session, and the SRS 3.4.7 admin audit log.

### What this run leaves to fix, in priority order

Ranked by what a marker would ask about first, not by effort:

1. **Single-sector latency breaches SRS 3.4.1** (§1). A stated requirement the
   system does not meet, and the queries that breach it also pass the API's own
   25 s timeout. Needs a decision as much as a fix: cache the merge call, cut a
   model hop, or raise the budget in the SRS with a written justification.
2. **Agents must restate computed figures in their evidence** (§1, grounding
   class 1). Five of six ungrounded figures are one bug in `trade_economics`
   and the sector agents, and it is the traceability claim the whole project
   rests on.
3. **The router drops agents on broad cross-sector comparisons** (X04, X06, M06).
   Under-fanning yields a confident answer to half a question.
4. **S06's contradictory price sentence** — verify by hand, then decide whether
   anything can catch a right number on a wrong claim.
5. **The coherence rating session** — longest lead time, needs three people.

---

---

## 7. Policy retrieval — the 15-question set (D10)

Measured on **2026-08-28**, `eval/policy_questions.yaml`, against a Qdrant
collection of **901 chunks from 6 documents** (Sri Lanka, UK, Canada, US, India,
Italy). Reproduce with:

```bash
make eval-policy-baseline   # CEYNEX_POLICY_RETRIEVAL=off — the system before D10
make eval-policy            # with retrieval
```

The 15 questions were committed in `dc70aa3`, **before** `ceynex/retrieval/`
existed, for the reason `questions.yaml` was: questions written after watching
the system answer them describe it instead of testing it.

### Headline

The middle column is the system with retrieval switched off *after* the routing
fix, so the last two columns isolate retrieval and the first two isolate routing.

| Metric | Before routing fix | Routing fixed, retrieval off | Retrieval on |
|---|---|---|---|
| Routing — exact agent-set match | 53.3% | **73.3%** | **73.3%** |
| Routing — recall of expected agents | 0.678 | **0.872** | **0.872** |
| Answers fully grounded | 33.3% | 41.7% | **50.0%** |
| Ungrounded figures | 9 | 8 | **7** |
| Mean evidence per answer | 2.42 | 2.25 | **2.92** |
| **Answers with no evidence at all** | **3** | 2 | **0** |
| Unanswerable correctly refused | 33.3% | 33.3% | 33.3% |

Denominators: 15 for routing, 12 answerable for evidence, 3 for refusal.

Every category stayed inside its SRS 3.4.1 budget: single-sector p95 8,399 ms
against 10,000 ms. Retrieval costs 300–900 ms on the paths that use it and is
skipped entirely for FX shocks, which is most simulation traffic.

### The routing fix was the unlock, and it needed three changes

The first run of this set produced **no movement at all** — 33.3% grounding on
both paths and three answers with no evidence. The cause was not the retriever:
P03, P04 and P05 ("what does India's Foreign Trade Policy say", "what non-tariff
measures does the EU apply", "does the Netherlands identify Sri Lanka as a
priority market") routed to `export_analytics` **alone**, which holds only Sri
Lanka's own trade flows. They scored 0 evidence and 0.15 confidence. Retrieval
lives in `trade_economics`, so a question that never reaches it cannot benefit
from it however good the corpus is.

**Routing them there without the other two changes made the system worse, and
this was measured rather than reasoned about.** `_classify_shock` fell through to
its `fx` default for anything it did not recognise, so "What does India's
Foreign Trade Policy say about imports from Sri Lanka?" was answered with a 5%
rupee depreciation and a figure of **USD −8,240,802** — a confident number about
a currency move nobody mentioned, in reply to a question about a document.

1. **`_classify_shock` gained a `policy` class.** A question is descriptive when
   it names a policy instrument and nothing in it posits a change. Checked
   against all 45 questions in both sets: every simulation stays a simulation.
2. **`_describe_policy` answers from documents and reports no impact figure at
   all.** Where the graph knows part of the answer it still leads with it — "does
   the UK keep preferential access for Sri Lankan tea" is answered from
   `agreement_coverage` (DCTS, GSP+, ISFTA, SAFTA, APTA cover HS 09), with
   documents as corroboration rather than as the source of record.
3. **Both routers send foreign trade-policy questions to `trade_economics`.**

Two defects surfaced only because the fix was measured at each step:

- **The first prompt edit routed correctly and then marked the same questions
  `out_of_scope`.** The merger treats an out-of-scope route's findings as noise,
  so P03 and P04 came back as *"Part of the question names a sector CeyNex does
  not cover"* — about questions it does cover. The scope rule needed an explicit
  carve-out: a destination market's trade policy is in scope even when the
  question names no commodity.
- **`parse_intent` resolved the wrong country.** It matches the *longest* country
  name, and "Sri Lanka" is longer than "India", so the India question anchored on
  LKA and answered about Indian policy with four confident passages from Sri
  Lanka's own export strategy. `_destination()` now excludes the reporter
  outright — Sri Lanka is never one of its own export destinations.

### What retrieval itself contributes

With routing fixed, the retrieval delta is real rather than noise: **+8.3 points
of grounding, +0.67 evidence per answer, and the last two empty answers
eliminated.** Per question, the gains are P03 0→1, P05 0→1, P08 1→3, P15 2→4,
P01/P02/P04/P06/P07 each +1.

Measured directly, outside the orchestrator:

| Query | Top hit | Score |
|---|---|---|
| "Does the UK trade strategy keep preferential access for developing countries?" | UK Trade Strategy — Economic Partnership Agreements | **+5.95** |
| "What are Canada's trade priorities and market access negotiations?" | Canada briefing book — Trade Policy / Market Access | **+6.24** |
| "What does India's foreign trade policy say about imports and exports?" | India FTP 2023 — DGFT scheme administration | **+3.65** |

Country anchoring holds: the India query's second-ranked hit is the Sri Lankan
strategy at **+0.22**, far below the Indian document rather than winning on
general trade vocabulary.

### The refusals are the result worth keeping

Three of the four questions with no answerable content now produce a *precise*
refusal instead of silence or a fabrication:

- **P05 (Netherlands)** — "The knowledge graph's indexed policy documents for NLD
  are: none", citing the Cypher that established it. 60 ms.
- **P03 (India)** — anchors on IND, finds `IND-DGFT-FTP-2023`, and reports that
  none of its passages answer the question. The India page is DGFT scheme
  administration and genuinely says nothing about imports from Sri Lanka.
- **The US tariff question** — the USTR page is a link index ("to read the tariff
  schedule, click here"). Its best chunk scores **−7.91** and the retriever
  returns nothing.

That last one is why `MIN_RERANK_SCORE` exists. Before it, the same question
cited the corpus's ABBREVIATIONS page — which matched only because it contains
the words "United States dollars" — as evidence about US apparel tariffs. A real
citation, a real URL, a real page number, attached to a claim the page does not
support: exactly the failure §5 says `orchestrator/grounding.py` cannot catch.

The floor was left strict after a deliberate check. The **same** UK chunk scores
**+5.36** for "preferential access for developing countries" and **−3.36** for
"preferential access for Sri Lankan tea". The UK strategy does not mention Sri
Lanka, and presenting a passage about Economic Partnership Agreements as an
answer about Sri Lankan tea would be the misattribution above.

### Regression check on the 30-question set

Re-run cold against the same stack. Compared with §1's 2026-08-28 figures:

| Metric | §1 (before D10) | After D10 + routing fix | Degraded path |
|---|---|---|---|
| Routing — exact match | 53.3% | 50.0% | 40.0% |
| Routing — recall | 0.856 | 0.825 | 0.881 |
| Answers fully grounded | 77.8% | **81.5%** | 92.6% |
| Ungrounded figures | 6 | **5** | 2 |
| Mean evidence per answer | 3.67 | **3.85** | 4.52 |
| Answers with no evidence at all | 2 | **1** | 3 |
| Unanswerable correctly refused | 100% (3) | 100% (3) | 100% (3) |
| Crashes | 0 | 0 | 0 |

**Grounding improved and routing moved down by one question.** The routing change
is S03 and X03 drifting under LLM non-determinism, not a systematic effect — the
same set re-routed differently on two runs of unchanged code, which §1 already
warns about for latency and applies equally here. Read 50.0% and 53.3% as the
same number until someone runs it three times.

**One real regression was found and closed.** X09 ("Which sector would be hurt
more by losing access to the United States market?") began routing to
`trade_economics` once the router learned to send access questions there — which
the pre-written label says is correct. `_classify_shock` then read it as an FX
shock, because "access" was in none of its keyword lists, and produced the same
phantom −8,240,802. Market-access loss is now an agreement shock; X09 returns 4
evidence entries and no ungrounded figure.

**All three latency budgets passed on this run**, single-sector p95 at 8,117 ms
against 10,000 ms, where §1 recorded 14,638 ms breaching it. **Do not read that
as a fix.** §1 records p95 swinging between 14.6 s and 29.0 s across two cold
runs of the same questions; one run below the budget does not characterise a tail
that unstable, and nothing in D10 targeted latency.

### What to fix next, in priority order

1. **Replace the five JavaScript-shell URLs with the PDFs they link to.** No code
   change — the manifest header names each one. Germany, the Netherlands and the
   UAE are unrepresented purely because of this.
2. **Replace the USTR landing page with the actual HTS schedule**, and fix the
   three 404s (India MoC, EU DG TRADE, Canada State of Trade). The EU one matters
   most: several questions in this set are EU-focused.
3. **Add the per-country WTO Trade Policy Reviews.** Still the most tariff-dense
   documents available for these markets, and still not collected.
4. **The router still under-fans on broad comparisons** (P07, P08, P09, P15 drop
   `export_analytics`). Same defect §1 records for X04/X06; not made worse by D10.

### Threats specific to this section

- **Six documents, one of them 60% of the corpus.** The Sri Lankan strategy is
  544 of 901 chunks, so any unfiltered retrieval is biased toward it. The country
  filter is what holds that in check, which makes the anchoring test above
  load-bearing rather than decorative.
- **Every retrieved figure is `unverified`**, the same status as
  `trade_agreements.csv`. No rate lifted from a document has been checked against
  an official schedule by a human.
- **The relevance floor is calibrated on one corpus.** `MIN_RERANK_SCORE = 0.0`
  is the cross-encoder's own boundary rather than a tuned constant, but the
  evidence that it separates useful from useless here is 15 questions on 6
  documents.
- **Refusal is unchanged at 33.3% (1 of 3)** and is the weakest number in the
  table. D10 did not target it. The three unanswerable questions here are harder
  than the 30-set's — P06 wants a 2030 tariff and P10 a country with no document —
  and this needs the by-hand check §2 describes before it is quoted anywhere.
- **The retrieval-on/off comparison is one run each.** The routing columns are
  stable across runs because routing does not depend on the switch; the evidence
  columns are not, and a second pair of runs would be worth taking before these
  land in the Testing and Evaluation Document.

---

## 8. The conversational layer, and the noise floor nobody had measured

**Measured 2026-09-10, M2**, against the same local stack §1 used — verified
identical before running: 4,625 `fact_trade` rows spanning 2015–2024, 4,625
`EXPORTS_TO` relationships mirroring them, 901 chunks in `ceynex_policy`. The LLM
cache was cleared before **every** run below, because a warm `.cache/llm` gives
the false 65 ms p50 §1 warns about.

The question this section answers: **did the conversational layer (D12/D13)
change the orchestration path the 30-question set measures?** If it did, that is a
defect, not a feature.

### The committed baseline was two weeks stale, and comparing against it would have lied

`eval_results.json` in this repo was last written 2026-08-28. `main` has since
merged the out-of-scope routing fix. A fresh run of `main` today scores materially
better than the committed file:

| Metric | committed 2026-08-28 | `main` 2026-09-10 |
|---|---:|---:|
| routing exact match | 0.50 of 30 | 0.60 of 30 |
| routing recall | 0.825 | 0.925 |
| answers fully grounded | 0.8148 of 27 | 0.8519 of 27 |
| ungrounded figures | 5 | 4 |
| mean evidence per answer | 3.85 | 4.33 |
| answers with no evidence | 1 | 0 |

Diffing the feature branch against the committed file would have credited the
conversational layer with a routing fix it had nothing to do with. **A comparison
is only meaningful against a run of the code you are comparing to, taken now.**

### The noise floor: ±1 question on routing and grounding, and p95 is not usable at n=1

Two runs of **identical `main` code**, both cold-cache:

| Metric | main run 1 | main run 2 |
|---|---:|---:|
| routing exact match | 0.60 | 0.5667 |
| ungrounded figures | 4 | 5 |
| single-sector p95 | 8,567.8 ms | 5,581.9 ms |

Routing moved by one question and the ungrounded count by one figure with no code
change at all. **Single-sector p95 moved by 35%.** Per-question, run 1 was the
outlier on three questions (X11's route, M01's and M03's ungrounded figures) where
run 2 and the branch agreed with each other.

Two consequences worth carrying into the Testing and Evaluation Document:

- **A single-run difference of one question is not a result.** The LLM path has a
  noise floor of roughly ±3.3 pp on routing exact match and ±1 on ungrounded
  figures. Report a difference only if it survives repetition.
- **Latency p95 from one run of 12 samples should not be quoted at all.** §1
  already flags the tail as unstable; this quantifies it. The 14.6 s and 29.0 s
  figures in §1 and the 8.1 s in §7 are all single runs and should be read as
  draws from a wide distribution, not as measurements.

### S07 is nondeterministic on both branches, and that took 21 runs to establish

The branch's full run differed from `main` on one structurally important
question — S07, *"Which markets buy the most Sri Lankan knitted apparel?"*, where
the route dropped `export_analytics` and the answer came back with **no evidence
at all**. That is exactly the kind of thing this check exists to catch, so it was
run down rather than waved through as noise.

`keyword_route` returns the correct two-agent route for S07, so the failure is not
a fallback — it is `llm_route` itself choosing a single agent. Sampling the
question in isolation:

| | correct | wrong |
|---|---:|---:|
| `main` | 11 | 2 |
| feature branch | 6 | 3 |

Fisher's exact p ≈ 0.36. **`main` fails S07 too, roughly one run in five.** The
two branches are statistically indistinguishable on it. It is a real weakness of
the LLM router on this phrasing and it belongs on the defect list — but it is not
a regression, and it predates this work.

### One genuine defect, found by this check and fixed

`_plan_steps` ran **unconditionally** inside `route_node`, concurrently with
routing. But `trace.emit("thought", ...)` is a no-op without a trace sink, so on
`POST /api/query` and throughout `eval/harness.py` the planner's LLM call was
made, paid for, and its result discarded — an extra call on every query, and extra
concurrent load on the client during the one node whose output decides which
agents run.

Gated on `trace.active()`, the module's own "is anyone listening" helper. Measured
effect on the two questions that had moved, sampled in isolation:

| | S07 correct | X09 clean |
|---|---:|---:|
| before the gate | 2 of 3 | 0 of 3 |
| after the gate | 4 of 4 | 3 of 4 |

and on the full set, the ungrounded-figure count returned from 7 to 5 — inside
`main`'s own observed range of 4–5. `tests/orchestrator/test_planner.py::
test_no_planner_call_is_made_when_nothing_is_listening` guards it in both
directions.

### Verdict

| Metric | `main` (2 runs) | branch, planner gated |
|---|---:|---:|
| routing exact match | 0.60 / 0.5667 | 0.5667 |
| routing recall | 0.925 / 0.925 | 0.8917 |
| answers fully grounded | 0.8519 / 0.8519 | 0.8148 |
| ungrounded figures | 4 / 5 | 5 |
| answers with no evidence | 0 / 0 | 1 *(S07, above)* |
| crashed | 0 / 0 | 0 |
| single-sector p95 | 8,568 / 5,582 ms | 5,353 ms *(within budget)* |

**Degraded mode is byte-identical between `main` and the branch** — routing exact
0.4667, recall 0.9667, grounded 0.9259, 30 of 30 degraded, 0 crashed, on both.
That is the strongest single line here: degraded mode is the fully deterministic
path, so if the orchestration had actually changed, it would show there first and
without ambiguity. It does not.

Every remaining difference on the LLM path sits inside the run-to-run variance
measured above, and the one difference large enough to be worth chasing was
chased and found to be present on `main` as well.

**Not yet re-run:** the 21 GREEN questions in `queries.md`, and `make coherence`.

### After all four phases — measured 2026-09-10

Re-run with the clarification gate, web-search enrichment, the usage ledger,
custom instructions and the confidence breakdown all in place, cold cache:

| Metric | `main` run 1 | `main` run 2 | all phases |
|---|---:|---:|---:|
| routing exact match | 0.60 | 0.5667 | **0.60** |
| routing recall | 0.925 | 0.925 | **0.925** |
| answers fully grounded | 0.8519 | 0.8519 | 0.8148 |
| ungrounded figures | 4 | 5 | 6 |
| mean evidence per answer | 4.33 | 4.33 | **4.33** |
| answers with no evidence | 0 | 0 | **0** |
| crashed | 0 | 0 | **0** |
| single-sector p95 | 8,568 ms | 5,582 ms | 5,867 ms *(within budget)* |

Routing, mean evidence and the no-evidence count land exactly on `main`. Grounding
is one question below it and the ungrounded count one figure above the observed
`main` range of 4-5 — both inside the noise floor measured above, and neither
worth reporting as a result on a single run.

**Degraded mode remains byte-identical**: routing exact 0.4667, recall 0.9667,
grounded 0.9259, 30 of 30 degraded, 0 crashed — the same figures `main` produces.
Since degraded mode is the fully deterministic path, this is the line that says
the orchestration itself is unchanged, and it says it without ambiguity.

**None of this measures the new features**, and it is not meant to. The 30
questions are one-shot queries: they never open a conversation, never trigger the
clarification gate (asserted separately — it is silent on all 30), and never make
an outbound web call. What this run establishes is the thing that mattered most —
that adding all of it did not disturb the path the published numbers describe.
Measuring the conversational features needs a multi-turn harness, which
`DEFERRED.md` records as not built.

**Still not re-run:** the 21 GREEN questions in `queries.md`, and `make coherence`.

## 9. Inline citations — the pre-registered rule, then the measurement

**Rule written 2026-09-11, before any cited run.** `CEYNEX_CITATIONS` shipped off
because enabling it changes the merge prompt every answer is written from, and §8
established that a prompt change measured on a single run is worth nothing. The
protocol that §8 called for now exists — `make eval-repeat` runs the set three
times with the prompt cache cleared before each and reports medians and spread
(`eval/repeat.py`) — so the flag can be decided rather than guessed.

Two conditions, three cold runs each: `make eval-repeat` (off) and
`make eval-repeat-cited` (on). The decision is made on the **medians**, and the
thresholds below encode the noise floor §8 measured (one question on routing and
grounding, one figure on the ungrounded count). All six must hold for the flag to
go on:

| # | Criterion | Why |
|---|---|---|
| 1 | Routing exact match and recall: medians **identical** to the off run | Citations touch the merge prompt only. Any movement here is a defect, not noise. |
| 2 | `answers_fully_grounded` median ≥ off median − 0.037 | One question of 27. A larger drop means the SOURCES block is confusing the model. |
| 3 | `ungrounded_figures_total` median ≤ off median + 1 | The measured floor is ±1. |
| 4 | `citations.marker_valid_rate` median ≥ 0.98 | A `[n]` pointing past the evidence list is an invented citation — the failure the feature exists to prevent. |
| 5 | `citations.figure_sentences_cited_rate` median ≥ 0.80 | Below this the markers are decoration, not a discipline, and not worth a prompt change. |
| 6 | `crashed` = 0 and `answers_with_no_evidence` no higher than the off median | Table stakes. |

Anything else and the flag stays off, with the failing criterion recorded here.
The rule is not revised after the runs; if it turns out to be the wrong rule, that
is a new section with a new rule and a new set of runs.

### Measured 2026-09-11 — the flag stays off

Six cold runs, three each way, on the same stack §8 used (4,625 `fact_trade` rows,
4,625 `EXPORTS_TO` edges, verified before the first run). Every figure is a median
of three with the spread in brackets; the per-run files are in `eval_runs/off/`
and `eval_runs/on/`.

| Metric | off (3 runs) | on (3 runs) | criterion | holds? |
|---|---:|---:|---|---|
| routing exact match | 0.5667 [0.5667–0.60] | 0.5667 [0.5667–0.5667] | identical medians | **yes** |
| routing recall | 0.925 [0.925–0.925] | 0.925 [0.8917–0.925] | identical medians | **yes** |
| answers fully grounded | 0.8519 [0.8148–0.8519] | 0.7778 [0.7778–0.8148] | ≥ 0.8149 | **no** |
| ungrounded figures | 4 [4–5] | 7 [6–7] | ≤ 5 | **no** |
| marker valid rate | — | 1.00 [1.00–1.00] | ≥ 0.98 | **yes** |
| figure sentences cited | — | 0.877 [0.817–0.885] | ≥ 0.80 | **yes** |
| crashed / no evidence | 0 / 0 | 0 / 0 [0–1] | 0 / ≤ off | **yes** |
| single-sector p95 (ms) | 9,219 [6,519–11,568] | 6,154 [6,153–8,895] | — | noise |

**Criteria 2 and 3 fail, so `CEYNEX_CITATIONS` stays off.** Two answers of 27
lost their grounding, not one, and the ungrounded count rose by three, not one.
Everything the feature was meant to do, it did: every `[n]` the model wrote
pointed at a real evidence entry (24 of 27 answers carried markers), and 88% of
the sentences stating a figure cited one.

**What the extra ungrounded figures are — two classes, and only one is the
metric's fault.** Read from the prose around each figure:

- *X09* (`161,815,198`, `30,216,274`, in two of three cited runs): the
  trade-economics impact figures, which the evidence carries **signed** ("USD
  -161,815,198") and the prose states unsigned with the word "decrease".
  `grounding.ungrounded_figures` compares digit strings and keeps the sign, so
  `-161815198` does not ground `161815198`. That is §1's grounding class 1 — a
  figure the agent computed, quoted differently — and the same shape as the four
  ungrounded figures the off runs carry every time (M01, M02, M03, M04).
- *M03* (`1,318,528,338`, one run) and *M05* (`2,872,929,484`, all three cited
  runs against one of three off runs): **totals the model worked out itself** —
  "resulting in a new total of about USD 1,318,528,338", which is the baseline
  minus the impact, and "would bring the total apparel export revenue to around
  USD 2,872,929,484", the baseline plus the shock. No finding and no evidence
  entry states either number. These are exactly what the grounding check exists
  to catch, and asking the model to cite every figure-carrying sentence appears
  to make it *more* inclined to spell such a total out beside the citation.

So the verdict is not an artefact of the metric. Citations cost real grounding
on this set, on a rule written before the runs, and the flag stays off.

**What this does not permit.** The rule was pre-registered and it is not revised
here. A sign-tolerant comparison in `grounding.py` would remove the X09 class —
and it would also change the runtime guard and every published grounding figure
in this document, which is precisely the kind of change that needs its own
pre-registered rule and its own six runs. The derived-total class is a prompt
problem (rule 2 already forbids it; rule 6a seems to pull against it) and would
need a reworded 6a, measured the same way. Both are recorded here and not taken.

**Two things the repeated runs established on the side.** The questions that
disagree with themselves across identical runs are X11 (route) and M05
(grounding) with the flag off, S07 (route, §8's known case), X09 and M03 with it
on — the same handful every time, which is where a larger question set would
earn its keep. And single-sector p95 ranged from 6.2 s to 11.6 s across six runs
of the same code with one question (S01, S10 or S11) over 10 s in two of them,
which is the §8 warning in numbers: no single-run p95 from this set is a
measurement.

## 10. The conversational layer, measured on conversations

**Measured 2026-09-11, M2**, on the same stack as §9, cold cache. §8 established
that the layer left the one-shot path unchanged; nothing before this measured the
layer itself. `eval/conversations.yaml` holds eight conversations, 22 turns,
with each turn's expected behaviour written down before the run — which path the
classifier should take, what a rewrite must carry, whether the gate should ask,
whether a discussion must survive grounding. `eval/chat_harness.py` drives them
through `api/turn_runner.py` exactly as `POST /api/chat/stream` does, in-process,
and scores the frames each turn wrote (`make eval-chat`, `make eval-chat-degraded`;
results in `eval_chat.json` and `eval_chat_degraded.json`).

### Headline

| Metric | LLM (23 turns) | degraded (23 turns) |
|---|---:|---:|
| classifier: follow-up took the expected path | **13 of 14** | 12 of 14 |
| rewrite carried what the reader named | **6 of 6** | 3 of 6 |
| discussion survived grounding | 6 of 7 | 7 of 7 *(trivially — the degraded discussion is the prior answer)* |
| gate asked where expected / silent elsewhere | **1 of 1 / 22 of 22** | 1 of 1 / 22 of 22 |
| turns passing every check | 19 of 23 | 18 of 23 |
| analyses that stated some limit *(informational, SAD §4.1)* | 12 of 16 | 12 of 17 |

The one clarified turn was answered through the resume path with the template's
last option ("both"), the composed query ran the graph, and the gate did not ask
again — the one-round cap holding in a real turn, not a route test.

*The set changed on 2026-09-12 and these figures predate it (re-run: §12).* "both" is no longer
offered (D13, amended), so C05's clarified turn now names its answer, cinnamon.
That is also the case that was broken: a reader's second-named choice was
answered with the first item. The run above has not been repeated on the
changed set here; §12 has the re-run.

### Cost and shape, by mode — the number that replaces "3 frames against 22"

That figure, quoted in D13 and in `IMPLEMENTED_FEATURES.md`, was measured before
`discuss` turns had a trace. With one:

| mode | turns | SSE frames (median) | answer sentences | elapsed p50 | max | model calls | cost per turn |
|---|---:|---:|---:|---:|---:|---:|---:|
| first turn (graph) | 8 | 35.5 | 3 | 6.8 s | 7.9 s | 5.25 | $0.0027 |
| analyse follow-up (graph) | 8 | 32.5 | 3 | 5.8 s | 7.7 s | 5.4 | $0.0027 |
| discuss follow-up (no graph) | 6 | 8.5 | 2.5 | 2.7 s | 3.9 s | 2 | $0.00025 |
| clarify (gate only) | 1 | 5 | 0 | 1.3 s | — | 0 | $0 |

A discussion is **8.5 frames against 33, in 2.7 s against 5.8, at a tenth of the
cost** — two cheap-model calls (classification and the discussion) against five
and a half. The cost argument in D13 holds; the frame count it quoted does not,
and is corrected in both places. Degraded mode answers a first turn in 333 ms
and a discussion in 33 ms, with no model calls at all.

### The four misses, read one at a time

- **C03 turn 1 and turn 2 — the S07 class, again.** *"Which markets buy the most
  Sri Lankan knitted apparel?"* was routed to `apparel_manufacturing` alone,
  dropping `export_analytics`, and on this stack (which has no EDB apparel
  sub-category data) that agent declines, so the turn had no evidence. §8
  measured this router failure at roughly one run in five on `main`; it landed
  on the first turn here and again on the rewritten follow-up. The rewrite itself
  was faithful (the United Kingdom and knitted apparel both carried), though it
  said "volume" where the reader's original said value — the kind of drift the
  `standalone_contains` check cannot see and a human reading the transcript can.
  Not a defect of the conversational layer; the same question one-shot fails
  the same way.
- **C07 turn 2 — the grounding guard withheld a derived figure.** *"How wide is
  the uncertainty band, and what does it mean?"* invites a subtraction, the model
  performed it, and `ungrounded_figures` rejected the reply for a number no
  finding stated. That is the guard doing its job (it is also what §9 found
  citations encourage). It is scored as a miss deliberately: the set expects the
  system to answer the question from the bounds it *was* given, and it did not.
- **C08 turn 2 — a defensible classification counted against it.** *"Give me
  just the agriculture side of that"* was classified `analyse` and re-run as
  *"What are the export figures for Sri Lanka's agriculture sector to the
  European Union in 2024?"* — a correct answer, at the cost of a fan-out the
  previous answer could have supplied. The set says discuss; the model chose to
  re-analyse; both are honest readings, and the miss is recorded as the set's.

### Degraded mode: the limit is the rewrite, not the classifier

With no model, the keyword classifier still took the expected path 12 times in
14, and the template gate asked exactly where the LLM gate did. What it cannot do
is **rewrite**: *"What about the United Kingdom specifically?"* and *"How has that
dependence changed since 2020?"* went to the graph as typed, named nothing CeyNex
covers, and were declined as out of scope. A follow-up that depends on the turn
before it needs the model to make it standalone; SRS 3.4.3's degraded path keeps
the conversation but not that. Recorded as the line to draw, not a defect to fix.

### What this stack cannot show

The local volumes hold the 2015–2024 Comtrade series and the policy corpus, and
lack what the deployed host has: the Tea Board and FAOSTAT series, the EDB
apparel sub-categories and the registered forecast models. Co-routed agents
decline on those, so **per-answer confidence here is not comparable to
`queries.md`** (a knitted-apparel market-share answer that scored 0.70 on the
host scored 0.09 here with the same figures, because two of three agents
reported no data). That is why the harness scores "answered" as prose plus
evidence and records confidence without judging it.

## 11. Concurrency — 50 concurrent users (SRS 3.4.2)

Measured **2026-09-12** with `eval/load_test.py`, the first measurement of this
requirement — `docs/DEFERRED.md` had flagged it untested since the rate limiter
shipped. Unlike §1's harness, which drives the orchestrator in-process, this
sends real HTTP requests at a running server, because SRS 3.4.2 is a claim about
serving capacity: the ASGI event loop, the Postgres/Neo4j connection pools, and
the rate limiter, none of which an in-process call exercises.

```bash
make up
.venv/Scripts/python -m uvicorn ceynex.api.main:app --host 127.0.0.1 --port 8000 &
python -m eval.load_test --users 50 --timeout 45 --json load_results.json

# to match the deployed --workers 2 + Redis topology instead of one process:
docker run -d --name ceynex-redis-loadtest -p 6379:6379 redis:7
REDIS_URL=redis://127.0.0.1:6379/0 .venv/Scripts/python -m uvicorn \
  ceynex.api.main:app --host 127.0.0.1 --port 8000 --workers 2 &
python -m eval.load_test --users 50 --timeout 45 --json load_results_2w.json
```

**First run's setup differed from the deployed image in two ways**: `REDIS_URL`
unset (rate limiter running `InProcessWindow`), and a single `uvicorn` process,
not the deployed `--workers 2`. Each virtual user still got its own rate-limit
identity (a distinct `X-Real-IP` per user — `load_test.py`'s own docstring
explains why), so the *50-distinct-callers* shape of the test held either way,
but that first run measured one process's capacity, not the exact
two-worker-behind-Redis topology running in production. **Re-run same day**
against a standalone `redis:7` container (`REDIS_URL` pointed at it) and
`uvicorn --workers 2` — the actual deployed shape — to check whether either
finding below was a single-process artifact.

### Headline

| | Single process, no Redis | `--workers 2` + Redis (matches deployed) |
|---|---|---|
| Concurrent users | 50 | 50 |
| Succeeded (HTTP 200) | **50 / 50** | **50 / 50** |
| Failed / timed out | 0 | 0 |
| Falsely rate-limited (429) | 0 | 0 |
| Wall clock for all 50 | 13.3 s | 14.6 s |
| Sum of the 50 individual latencies | 420.7 s | 329.2 s |
| Degraded answers | 44 / 50 | 43 / 50 |

Both runs land in the same place: full availability, a large wall-clock-vs-
summed-latency gap either way (confirms genuine concurrent handling regardless
of worker count — the negative check this section exists to run: PR #31/#32,
2026-08-26, fixed two blocking-call bugs that each froze the *entire*
single-threaded event loop for every concurrent caller, not just the one whose
query triggered them; a regression of either would show a wall clock close to
the summed figure, in either topology), and an almost identical degraded-answer
rate. **The two-worker/Redis run does not fix what the single-process run
found** — this was never a single-process artifact.

| Category | Single process — p95 | 2 workers + Redis — p95 | budget (SRS 3.4.1) |
|---|---|---|---|
| single_sector | 13.1 s | 13.1 s | 10 s — **breached in both** |
| cross_sector | 13.1 s | 14.6 s | 20 s — within budget in both |
| simulation | 13.3 s | 7.9 s | 20 s — within budget in both |

**Single-sector's own SRS 3.4.1 budget does not survive 50 concurrent callers,
in either topology.** §1's single-user p95 for this category has headroom
against 10 s; at 50 concurrent users that headroom is gone in both runs. This
is the first evidence that the single-sector budget is a single-user number,
not a serving-capacity one, and the two should not be quoted interchangeably —
and adding a second worker plus the production rate-limit backend did not
change that conclusion.

### A second, real finding: LLM-provider capacity is the actual ceiling under load

43-44 of the 50 answers came back **degraded** in both runs (SRS 3.4.3's
contract: real figures and evidence, no prose) — the server itself never
failed, but the two LLM providers behind it could not serve 50 concurrent
callers, and adding a second worker plus Redis changed that by one answer, not
by forty. The failsafe's own free-tier limit is visible directly in the server
log, in both runs:

```
Rate limit exceeded: free-models-per-min. (X-RateLimit-Limit: 20)
```

20 requests/minute is well below 50 concurrent, so once several callers reached
the failsafe together it was already exhausted for the rest. What is *not*
cleanly established from either run is why the **primary** (OpenAI) call
failed for nearly all of these before falling through to that failsafe at
all — both server logs, one capturing ~50 coroutines in a single process and
the other split across two, show plenty of failsafe-side warnings but
essentially zero from the primary path's own `except` blocks, which is itself
suspicious given the code guarantees one on every failed attempt
(`ceynex/llm/client.py`) — and no `daily spend cap ... reached` warning
appears in either log, which rules out the R5 spend cap as the cause. Getting
the same near-total silence on the primary path in a topology with half the
concurrency per process (2 workers, ~25 requests each vs. 50 in one) argues
against this being a log-interleaving artifact of one process handling all 50
at once, and toward a real capacity ceiling on the primary call itself
(nothing in `docs/DEFERRED.md`'s Operational section suggests the deployed key
has been checked against OpenAI's own rate limit tier) — but neither run's log
is clean enough to say that with certainty, and re-measuring with
per-request-tagged logging (or reading `provider_status()` directly through
the admin LLM-status route mid-run, rather than grepping console output) is
worth doing before this is treated as settled. Filed here rather than silently
left out, per this document's own rule about a headline number hiding what
produced it (§1's "Read the denominators" note).

**Net for SRS 3.4.2**: the system stays available and answers all 50 concurrent
users with real figures and evidence — nobody gets an error or a hang, and that
holds under both the local single-process setup and the `--workers 2` + Redis
topology that matches what's deployed. What degrades under load is answer
*prose*, gracefully, exactly as SRS 3.4.3 specifies, and single-sector latency,
which breaches its single-user budget — and neither finding is a topology
artifact, since both runs land in the same place. Whether the LLM-provider
ceiling found here is a deployed-key tier limit or something narrower is still
open; re-running this against the deployed VM itself (with the cost and
availability implications that implies) is the remaining step before this is
called measured against production rather than against a production-shaped
local stack.

### The sustained, signed-in run: the rule, written before it ran (2026-09-12)

The burst above answers availability for one moment of 50 anonymous callers.
SRS 3.4.2 names *authenticated* users, and a moment is not load. So there is a
second measurement, and its rule is fixed here before any of it runs.

**Shape.** 50 users, each signed in with a real account and token (`--signed-in`).
Each one asks, waits for the answer, and asks again no sooner than 2.5 s after
its last question (`--mode sustained`). They run against uvicorn `--workers 2`
with Redis, the deployed topology, over the local stack. That offers up to 1,200
questions a minute, with every user under its 30 a minute. Both endpoints are
run: `/api/query`, and `/api/chat/stream`, where the time that counts is the time
to its `done` frame. Each run's baseline is `--mode sequential`: one user, each
question once, same session, same endpoint.

**Two runs.**

- **(a) Degraded, no model key, 180 s.** This measures CeyNex's own capacity.
- **(b) With the model, the prompt cache off, 3 questions per user (150).** This
  largely measures the provider's tier. A provider's 429 reaches the reader as a
  degraded answer, which is reported, not scored.

**The rule** (`eval/load_test.py::verdict`), applied to each run:

1. No failures: no 5xx, timeouts, connection errors, or turns ending `failed`. No
   429s: the pacing keeps every user under its allowance, so a 429 is a limiter
   or identity bug.
2. Each category's p95 within its SRS 3.4.1 budget: 10 s single-sector, 20 s
   cross-sector and simulation. That is the literal reading of "no material
   increase in the response times specified".
3. Reported, not scored: p95 under load over the same session's single-user
   p95. Above 1.5× it reads as a material increase, even inside budget.
4. A category whose single-user p95 already breaks its budget is reported as
   breached at one user, not blamed on concurrency.

### Measured 2026-09-12 — run (a), degraded: both endpoints pass

**Setup.**

- The local stack, with uvicorn `--workers 2` and Redis (`redis:7.2`), on a
  32-core machine that also ran the load generator.
- The prompt cache off, through a copy of `config/` with `cache.enabled: false`,
  so no answer was served from an earlier paid run. With the cache on, 4 of 30
  baseline answers came back with cached prose despite there being no key.

Every request came from a real signed-in account. Runs are in `eval_runs/load/`;
the sustained runs keep their summary and every error, not all 3,600 rows.

**The first sustained run failed rule 1, and the cause was the harness.** It met
11 × 429, all on user 0. That account was the one the baseline had just used for
30 queries in 4 s, so user 0 began the run with its window already full. The
rule says a 429 means the limiter, or whose allowance a request was counted
against, is wrong. This time it was the harness's accounting: two runs shared one
account. Accounts are now named by mode (`eval/load_test.py`), and the run was
repeated with the rule unchanged. Both runs are recorded.

| endpoint | requests | succeeded | 429 | per minute | p95 single / cross / simulation | × one user | first frame p95 |
|---|---:|---:|---:|---:|---|---|---:|
| `/api/query` | 3,600 | 3,600 | 0 | 1,064 | 3.5 / 3.5 / 6.7 s | 65 / 67 / 5.7 | — |
| `/api/chat/stream` | 3,600 | 3,600 | 0 | 837 | 4.5 / 4.8 / 6.8 s | 15 / 16 / 4.6 | 0.9 s (17 ms at one user) |

**Both pass the rule.** There were no failures and no 429s at up to about 1,060
questions a minute, and every category's p95 stayed inside its SRS 3.4.1 budget.
Reading 3 calls the increase material. p95 grows from tens of milliseconds to
several seconds, which is what two workers cost under this concurrency with no
model in the path. That is CeyNex's own capacity. It leaves the budget room, but
the room is the model's to spend: §1 measures the single-user p95 *with* the
model at 6.2–11.6 s, before any concurrency. Run (b) is the one that says how
the two add up.

### Measured 2026-09-12 — run (b), with the model: single-sector breaks its budget, and the provider is why

Same setup as (a), with the model key present (the prompt cache still off), 50
signed-in users asking 3 questions each. No failover key was set, so a failed
call went straight to a degraded answer.

| endpoint | requests | succeeded | 429 from CeyNex | degraded (one user → load) | p95 single / cross / simulation | × one user | first frame p95 |
|---|---:|---:|---:|---|---|---|---:|
| `/api/query` | 150 | 150 | 0 | 5 of 30 → 14 of 150 | **11.6** / 14.9 / 14.0 s | 1.3 / 2.1 / 2.7 | — |
| `/api/chat/stream` | 150 | 150 | 0 | 5 of 30 → 21 of 150 | **13.4** / 15.9 / 14.8 s | 2.2 / 2.0 / 2.3 | 79 ms |

**Both fail rule 2, on single-sector only.** The breach is not there at one user:
the same session's single-user p95 was 9.2 s on `/api/query` and 6.1 s on the
stream. So it comes with concurrency. Cross-sector and simulation kept their
20 s budgets, and nothing failed or met a 429 from CeyNex.

**The cause, from the API's own log.** OpenAI answered 429, "Rate limit reached
for gpt-4o in organization …", on 208 first attempts across the two runs. 176
calls were still refused after their retry and degraded. That also answers what
this section left open above: why the primary call fell through under load. It
is the account's rate limit on `gpt-4o`, the model `config/llm.yaml` gives the
merge role. The stream's first frame stayed under 0.1 s throughout, so a reader
sees the trace start at once even when the answer is late.

**What SRS 3.4.2 therefore gets.** CeyNex's own serving path holds 50
authenticated users inside every budget, with no failures (run a). The system
with this model account does not hold single-sector's 10 s. The ceiling is the
provider's tier, not the server. SRS 3.6.5 treats the model as a purchased
component, and 3.4.2 allows capacity to grow "through standard scaling": a higher
tier, a failover key (unset here), or the merge role on the cheaper model. Each
of those is a change to measure the same way, under this same rule, before it
is claimed.

## 12. The owner's calls of 2026-09-12, measured

**Measured 2026-09-12, M2**, on the stack §9 used, before either change was
merged. The two calls `DEFERRED.md` left open after the live pass were decided:
don't let a distrusted route stick in the prompt cache, and stop offering "both"
on the clarification card (D13, amended). What each was expected to show was
written into CeyNex-AI/ceynex-core#80 before these runs.

### The router's prompt cache: the first rule was too broad, and was narrowed

The first version distrusted any route that dropped an agent `keyword_route`
selected. A cold run cannot see a cache change, so the check was a pair: a cold
run (`--repeat 1 --cold`, then a look at which router responses the cache kept),
then a warm run over what it kept (`eval_runs/router-cache/`).

| | cold | warm |
|---|---:|---:|
| routing exact match / recall | 0.60 / 0.925 | 0.60 / 0.925 |
| router responses kept in the cache | 19 of 30 | — |
| routes that narrowed the keyword route, and were not kept | 11 | — |
| …of those, already the *expected* route | 5 | — |
| …whose route changed when asked again | — | **0** |
| median time, replayed questions | — | 59 ms |
| median time, the 11 re-routed | — | 1,511 ms (max 3,240) |
| answers with no evidence | 0 | 0 |

The rule did what it said. No suspect route was kept, by direct inspection of
the cache. But it was the wrong rule. Most narrowings are the model being right,
and at temperature 0 re-asking returned the same route every time. So it cost
about 1.5 s on every repeat of a third of the set, and bought nothing this run
could see. **The owner narrowed it.** A route is distrusted only if its response
fell back (unparseable, or naming no real agent) or if **its answer came back with
no evidence**. That is what S07 did, and `run_query` is where it is known. On
these two runs the narrowed rule evicts nothing: neither had a fallback or an
evidence-free answer. The S07 miss did not occur this time, as it does in roughly
four runs of five (§8). When it does, its route no longer outlives its answer.

### The clarification card: a choice is now the item analysed

The multi-turn set was re-run (`eval_runs/chat-2026-09-12/`). C05's clarified
turn now names its answer, cinnamon, because "both" is no longer an option. That
is also the case that was broken: before the `parse_intent` fix in the same PR, a
reader who chose cinnamon was given the tea analysis. The run answered C05 with
Sri Lanka's USD 214,425,881 of cinnamon exports, with the model and in degraded
mode alike.

| | LLM (23 turns) | degraded (23 turns) |
|---|---:|---:|
| turns passing every check | **21** (19 in §10) | 18 (18 in §10) |

The two LLM misses are §10's own, read the same way: C07's derived band width
withheld by the grounding guard, and C08's defensible `analyse`. §10's two S07
misses on C03 did not recur, which is that question's usual nondeterminism, not
a fix. The degraded misses are §10's rewrite limit.

## 13. Direction-aware grounding — the pre-registered rule

**Rule written 2026-09-12, before any run with `CEYNEX_GROUNDING=direction`.**

### Why the check needs changing

§9 counted *"decrease by approximately USD 161,815,198"* as an ungrounded
figure. The trade-economics agent writes every impact signed ("USD
-161,815,198"). The model restates it unsigned, beside the word "decrease", and
`grounding.ungrounded_figures` keeps the sign. So the prose counts as
ungrounded. M01–M04 carried one such figure in every off run.

**On today's code that class costs prose, not grounding.** Since §9 ran, main's
#58 (`70513bb`) arrived with the merge. It grounds each agent's own explanation
before the merge sees it. That closed a real hole: an unchecked explanation
used to launder its figures into the corpus the merge guard trusts. It also
moved this class out of the metric and into the guard. The strict guard now
does two things:

- It discards the trade-economics explanation ("states a figure not in its own
  findings").
- It then discards the merge prose.

The reader gets the deterministic composition, marked degraded. The cold run
§12 took on this code (`eval_runs/router-cache/cold.json`, plus its console log,
which is not committed; the harness did not record discards until now) shows it:

- **4 answers degraded** (M01–M04). There were none in any of §9's six runs.
- **7 answers served as the deterministic composition.** 5 of them (X09, M01,
  M02, M03, M04) are this class. The other 2 cite years the evidence does not
  state.
- The strict ungrounded count **fell from §9's 4 to 2**, because the prose
  that carried those figures was never served.

`tests/agents/test_common.py` pins the mechanism, at no cost:

- the same explanation is discarded under `strict` and kept under `direction`;
- a rise stated for a fall is discarded under both.

The plan approved on 2026-09-12 judged this change on
`ungrounded_figures_total` (median ≤ baseline − 1). The guard now hides the
class from that metric, so that criterion would have failed a change that
works. It is replaced below, before any run, by what the change is for: prose
restored. The plan's routing criterion is replaced too, for the reason given
in criterion 1.

### The change

It sits behind `CEYNEX_GROUNDING=direction` and is off by default.

- An unsigned figure in the prose is grounded by the same figure carried
  negative, **but only when the figure's own sentence says the value fell**
  (`grounding.FELL`: decrease, decline, fall, drop, lower, reduce, loss and the
  like).
- A signed figure still matches only its own sign.
- A figure with no fall word in its sentence is judged exactly as before.

The check reads one sentence at a time, so the whole-prose guard and the
streaming `SentenceGate` still reach the same verdict. The 450 seeded property
cases in `tests/orchestrator/test_answer_stream.py` pass unchanged, and 100 more
exercise the direction rule.

Two properties are inherited from strict:

- **The prefix rule.** Strict accepts a prose figure when a corpus figure
  starts with its integer part. The same generosity now reaches negative
  figures: "USD 161.8 million" is accepted for "USD -161,815,198", just as it
  is for "161,815,198".
- **No check on a positive figure's direction.** "Fell by 11,355,453" is
  grounded by "USD +11,355,453" under both rules, as it always was.

### What each run records

Every run file carries all of the following:

- **`answers_fully_grounded` and `ungrounded_figures_total`** stay strict
  whichever way the flag is set, so the series above stays continuous.
- **The `…_direction_aware` versions** sit beside them.
- **`figures_accepted_by_direction_rule`** lists what only the new rule
  accepts. Each entry carries its sentence and the evidence entry that
  carries it negative.
- **`guards.answers_served_deterministic`** counts answers whose composed
  prose the merge guard discarded. The merge runs once per question, so each
  discard is one answer. Each `prose_discarded` entry names the figures it
  rejected.
- **`guards.explanations_discarded`** counts agent explanations discarded,
  one per agent, so it can exceed the number of answers.
- **`provider_gave_up`** counts calls on which the primary provider gave up.

### The rule

Two conditions, three cold runs each, all with `CEYNEX_CITATIONS=off`:

- `CEYNEX_GROUNDING=strict` (the strict runs)
- `CEYNEX_GROUNDING=direction` (the direction runs)

Same stack, same day, and the stack is verified before the first run as in §9.
Medians decide. All seven criteria must hold for `direction` to become the
default:

| # | Criterion | Why |
|---|---|---|
| 1 | Routing exact match and recall: the direction runs' medians are at most one question (≤ 0.034) from the strict runs' | The flag is read only after routing, so routing cannot move. This checks that the two sets of runs are comparable. It is not "identical", as in §9. Routing exact match was 0.60 in 3 of the 10 cold runs in §8 and §9, and 0.5667 in the other 7. With that spread, two medians of three would differ about one time in three by chance alone. |
| 2 | The direction runs' `guards.answers_served_deterministic` median ≤ the strict runs' median − 1 | The change exists to stop discarding correct prose. A change to a guard that restores nothing is not worth making. |
| 3 | **Every** figure the direction rule accepts, in all six runs, is read against its sentence and its evidence entry. Each must state that the same quantity fell, by the same amount up to the stated rounding. If even one does not, the change fails outright. That includes a figure stated as a rise, as a level, as another quantity's change, or as a different amount | Two things let a figure through that should not pass. One is sentence scope: a fall word elsewhere in the sentence vouches for the figure. The other is the prefix rule: a rounded form shares only the leading digits. Only reading catches either. The list is computed against evidence, as the metric is. The guard checks a wider corpus: the question, summaries, figures and assumptions. A figure it accepts from that corpus which evidence does not ground shows up in both ungrounded lists instead, and criterion 4 counts it. |
| 4 | The direction runs' `ungrounded_figures_total_direction_aware` median ≤ the strict runs' `ungrounded_figures_total` median + 1 | Prose the guard now keeps may bring in no figure except the ones the rule accepts. The +1 is §8's floor. |
| 5 | The direction runs' `answers_fully_grounded_direction_aware` median ≥ the strict runs' `answers_fully_grounded` median − 0.037 | One question of 27, as in §9's criterion 2. |
| 6 | `crashed` = 0 in every run, and the direction runs' `answers_with_no_evidence` median ≤ the strict runs' | Table stakes. |
| 7 | `make eval-degraded` under each setting: every answer is byte-identical | The deterministic path writes no prose to check, so the flag must not reach it. This check is free and has no noise. |

**Reported beside the criteria, not criteria themselves:**

- `degraded_answers` and `guards.explanations_discarded`, the explanation
  half of the same effect.
- The strict metric on the direction runs. It will rise: the prose that comes
  back states its impacts unsigned, which is exactly what the change accepts.
- Latency.

**Void runs.** A run with `provider_gave_up` > 0 is void. On some call the free
failsafe answered, or nothing did, and that says nothing about CeyNex. A first
attempt that fails followed by a retry that answers does not count.
`degraded_answers` is not a void condition, because under `strict` it is part
of the effect being measured.

A void run is renamed `void-run-N.json` and kept. Its replacement is written as
the next free `run-N.json` in the same directory. The summary is recomputed
over the three counted runs, and both are reported.

**The expected result, stated now.** It comes from the §12 cold run. That was a
single run taken for another purpose, so it is a prior, not a baseline; the
strict runs supply the baseline. The strict runs should serve about five
answers deterministic for this class and two for other reasons. The direction
runs should serve about two. Criterion 2 should therefore clear by about five,
not one. A pass by exactly one would mean the class is smaller than one run
suggested.

Anything else and the default stays `strict`, with the failing criterion
recorded here. The rule is not revised after the runs.

## 14. Rule 6a reworded — §9's rule, run again

**Rule written 2026-09-12, before any run with the reworded rule 6a.** §9 kept
`CEYNEX_CITATIONS` off because criteria 2 and 3 failed. The extra ungrounded
figures fell into two classes:

- One was the metric's: the signs §13 takes up.
- The other was the prompt's: totals the model worked out itself and wrote
  beside a citation. Examples are M03's *"a new total of about USD
  1,318,528,338"* and M05's *"USD 2,872,929,484"*. Rule 2 already forbade such
  figures, and rule 6a ("every figure a SOURCE states is cited") seemed to pull
  against it.

Rule 6a now ends:

> Cite a figure only as a SOURCE states it: never add, subtract or combine
> figures into a total, a difference or a new level, even beside a citation. If
> a sentence would need a figure no SOURCE states, leave that figure out.

Rule 6a exists only in the cited prompt, so it cannot move an off run.

**#58 hides the derived-total class from §9's metrics, just as it hides §13's
class.** A total like M03's appears in no finding. On today's code the merge
guard discards the prose that states it and serves the deterministic
composition, so the ungrounded metric never sees the figure. §9's six criteria
alone would therefore pass a prompt that made the model write *more* such
totals, as long as the guard kept throwing them away. So this rule adds
criterion 7, on what the guard discards. That makes the rule harder to pass.

Criterion 1 moves the other way, as in §13 and for the same reason. It is
reported both ways, identical medians and within one question, so a reader can
apply §9's rule verbatim.

The runs: three cold runs with `CEYNEX_CITATIONS=on` (the cited runs) against
three off runs, both under whichever grounding §13 adopts.

- **If §13 adopts `direction`,** its direction runs serve as the off runs.
  Criteria 2 and 3 then read the direction-aware metrics
  (`answers_fully_grounded_direction_aware`,
  `ungrounded_figures_total_direction_aware`).
- **If it does not,** its strict runs serve as the off runs, and the criteria
  read the strict metrics, exactly as §9 did.

Both definitions are reported either way, and §13's void rule applies.

| # | Criterion |
|---|---|
| 1 | Routing exact match and recall: the medians are at most one question (≤ 0.034) from the off runs'. §9's "identical" is reported beside it |
| 2 | `answers_fully_grounded` median ≥ off median − 0.037 |
| 3 | `ungrounded_figures_total` median ≤ off median + 1 |
| 4 | `citations.marker_valid_rate` median ≥ 0.98 |
| 5 | `citations.figure_sentences_cited_rate` median ≥ 0.80 |
| 6 | `crashed` = 0, and `answers_with_no_evidence` no higher than the off median |
| 7 | `guards.answers_served_deterministic` median ≤ off median + 1 |

Criteria 5 and 7 are not independent. A deterministic answer carries no
markers, so every discard also lowers the cited-sentence rate. If both fail on
the same answers, that is one failure counted twice, and the write-up says so.

If all seven hold, `CEYNEX_CITATIONS` defaults to on. That is a prompt change,
so it is redeployed. Otherwise the flag stays off, with the failing criterion
recorded here.

Reported beside the criteria, and not one of them: the number of derived totals
per run, whether they reached the prose or were rejected in a discard. A derived
total is a figure that is the sum or difference of two stated ones, and it is
the class the rewording is for.
