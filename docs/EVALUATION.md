# Evaluation — M2, Core Systems and Orchestration

> ## ⚠ These numbers are superseded. Do not quote them in the report.
>
> Every figure below was measured on **2026-08-19**. Three things have changed
> since, and each one independently invalidates a different part of it:
>
> 1. **Both sector agents were stubs then.** §5 says so outright and warns that
>    "coherence cannot be fairly rated until those agents land". M1's
>    `agriculture_commodity` and M3's `apparel_manufacturing` both landed on
>    2026-08-26. Every routing, grounding and coverage figure describes a system
>    two agents smaller than the current one.
> 2. **No LLM key was configured then; one is now** (OpenAI, plus an OpenRouter
>    free-tier failsafe added 2026-08-26). §1 predicts single-sector latency
>    moving from 1.36 s into the 3–8 s range once prose generation is on. Every
>    latency figure here is a measurement of the degraded path.
> 3. **The LLM router was therefore never exercised.** §1's three "genuine
>    routing misses" (X09, M05, M06) are all attributed to keyword-only routing,
>    and `llm_route` now actually runs.
>
> A fourth change affects grounding specifically: answers are now validated
> against retrieved facts at runtime (SRS 3.1.3,
> `ceynex/orchestrator/grounding.py`), so a composed answer carrying an
> unsupported figure is discarded rather than served. The 92.6% grounding rate
> below was measured with nothing enforcing that.
>
> **Re-run before writing the Testing and Evaluation Document (activity 084,
> due 20 Sept):**
>
> ```bash
> make eval            # 30 questions, LLM live
> make eval-degraded   # the same set, SRS 3.4.3 path
> make coherence       # blind rating sheets — now unblocked, see §4
> ```
>
> Keep this file's structure and its §5 threats-to-validity discipline; replace
> the measurements.

Measured on 2026-08-19 against the deployed stack: PostgreSQL 18 and Neo4j 5.26
on the database VM, 4,625 `fact_trade` rows of live UN Comtrade data covering
2015–2024 for HS 0902, 0906, 4001, 61 and 62.

Reproduce with:

```bash
python -m eval.harness --json results.json          # the 30-question set
python -m eval.harness --degraded --json degraded.json
python -m eval.backtest --sector agriculture --item cinnamon
```

**Read the denominators.** Several rates here rest on 3 observations. They are
reported with their `of` counts throughout because "100% correct" out of three
is a weaker claim than the percentage implies, and rounding that away would be
the most misleading thing in this document.

---

## 1. Orchestrator: the 30-question set

`eval/questions.yaml` holds 30 questions with their expected agent routes — 12
single-sector, 12 cross-sector, 6 simulation, of which 3 are unanswerable and 1
is partially answerable. **The file was committed before the harness was ever
run** (commit `1b4d831`, preceding the harness at `f209d02`), because questions
written after watching the system answer them describe the system rather than
test it.

### Headline

| Metric | Result | Denominator |
|---|---|---|
| Questions completed without crashing | **100%** | 30 |
| Routing — exact agent-set match | **40%** | 30 |
| Routing — recall of expected agents | **0.88** | 30 |
| Routing — never returned an empty route | **100%** | 30 |
| Answers with every figure traceable to evidence | **92.6%** | 27 answerable |
| Ungrounded figures across the whole run | **2** | — |
| Mean evidence entries per answer | **3.67** | 27 |
| Answers with no evidence at all | **0** | 27 |
| Unanswerable questions correctly refused | **100%** | 3 |
| Answerable questions returning no content | **0%** | 27 |

### Latency (SRS 3.4.1)

| Query type | p50 | p95 | Budget | Within budget |
|---|---:|---:|---:|---|
| Single-sector | 1,360 ms | 2,662 ms | 10,000 ms | yes |
| Cross-sector | 1,319 ms | 3,938 ms | 20,000 ms | yes |
| Simulation | 1,090 ms | 1,285 ms | 20,000 ms | yes |

Every query finished inside its budget with 5–15× headroom. This is a
single-user measurement; the 50-concurrent-user requirement (SRS 3.4.2) has not
been load-tested and is recorded as outstanding in [DEFERRED.md](DEFERRED.md).

The headroom has an unglamorous explanation: **no LLM key is configured, so the
system runs the SRS 3.4.3 degraded path** and never pays for a model call. These
are honest numbers for the system as it currently runs, and they are not the
numbers it will post once prose generation is switched on. Expect single-sector
to land in the 3–8 s range then, still inside budget, and re-measure rather than
assuming.

### Routing: why exact match is 40% and why that is not the whole story

Exact match counts a route as correct only if the agent set matches the
pre-written label exactly. Recall — did the router include every agent that was
needed — is **0.88**. The gap between the two is almost entirely the router
adding one *more* agent than the label listed, not missing one.

Fifteen of the eighteen non-matches are of this shape:

```
S07  expected [export_analytics]  ->  [apparel_manufacturing, export_analytics]
S08  expected [export_analytics]  ->  [apparel_manufacturing, export_analytics]
```

"Which markets buy the most Sri Lankan knitted apparel?" is labelled as pure
export analytics; the router also sends it to the apparel agent. That is
defensible behaviour and arguably better than the label. **The labels were not
edited to match** — doing so after seeing the results is precisely the failure
the pre-commit was meant to prevent. The honest reading is that exact match
penalises defensible over-fanning as harshly as a genuine miss, which is why
both numbers are reported.

Three genuine routing misses remain, and they are real defects, not labelling
disagreements:

| Id | Question | Missed |
|---|---|---|
| X09 | "Which sector would be hurt more by losing access to the United States market?" | `trade_economics` — reads as a loss scenario but uses none of the simulation vocabulary |
| M05 | "If global demand for knitted apparel fell 15%…" | `trade_economics` — a demand shock, but the keyword list only knows FX, tariff and agreement shocks |
| M06 | "Which sector should Sri Lanka prioritise… and by how much will that raise GDP?" | routed to `forecast` alone on "next decade" |

All three are limitations of keyword routing on queries that describe a shock
without naming one. The LLM router (`llm_route`) exists and falls back to
keywords, but with no API key configured the keyword router *is* the system, so
these are the numbers that matter today.

### Evidence grounding

Every figure appearing in a merged answer is checked against the claims and
Cypher of the `Evidence` entries attached to it. **92.6% of answers are fully
grounded, with 2 ungrounded figures in the entire run.** Both are the shock
magnitude quoted back from the question itself ("a 10% tariff"), which appears
in the assumptions rather than in evidence. That is arguably correct behaviour
and is left as reported rather than special-cased away.

The check is deliberately crude and over-reports: it compares digit strings, so
a figure rounded differently in prose than in evidence is flagged. For a metric
whose job is catching hallucinated numbers, a false alarm costs a manual check
and a miss costs the claim.

### Degraded mode (SRS 3.4.3)

The same 30 questions with the LLM forced unavailable: **0 crashes, 0 answers
without evidence, all 30 flagged `degraded=True`.** Grounding drops to 81.5%
because the deterministic composer restates forecast figures in prose that the
LLM-composed version phrases differently; no answer loses its evidence.

---

## 2. What the evaluation changed

The harness found four defects on its first run. All four are fixed, and the
before/after is the clearest evidence that the evaluation did work rather than
just describing a system that already passed.

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

### Cinnamon benchmark limitation

The Liyanage/Silva/Marasinghe purchasing-price panel is unavailable. Therefore
the cinnamon result above uses FAOSTAT's annual USD producer-price fallback and
is **not a reproduction of the published benchmark**. Any comparison in the
report must quote the paper's reported MAPE with this target, frequency, and
source difference stated beside it; it must not imply the same train/test split
or data were used.

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

**This requires three human raters and has not been run.** The scoring reports
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
- **Two of five agents are stubs.** M1's `agriculture_commodity` and M3's
  `apparel_manufacturing` return "not implemented yet". Routing to them is
  scored correct, content is scored unanswered — deliberately separate metrics,
  because a working router pointing at an unbuilt agent is a different situation
  from a broken router. Cross-sector answers are consequently thinner than they
  will be, and coherence cannot be fairly rated until those agents land.
  **Both landed 2026-08-26**, which is the single biggest reason this run needs
  repeating — see the banner at the top.
- **Latency was measured with no LLM configured.** See §1.
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
load test, WITS tariff ingestion (cut, deviation D9), and the coherence rating
session.
