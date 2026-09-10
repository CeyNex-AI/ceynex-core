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
