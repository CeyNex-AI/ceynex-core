# Evaluation — M2, Core Systems and Orchestration

Measured on **2026-08-28** against PostgreSQL 18 and Neo4j 5.26 holding 4,625
`fact_trade` rows and 4,625 `EXPORTS_TO` edges of live UN Comtrade data covering
2015–2024 for HS 0902, 0906, 4001, 61 and 62.

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

## 7. Policy retrieval — the 15-question set (D10)

Measured on **2026-08-28**, `eval/policy_questions.yaml`, against a Qdrant
collection of **901 chunks from 6 documents**. Reproduce with:

```bash
make eval-policy-baseline   # CEYNEX_POLICY_RETRIEVAL=off — the system before D10
make eval-policy            # with retrieval
```

The 15 questions were committed in `dc70aa3`, **before** `ceynex/retrieval/`
existed, for the reason `questions.yaml` was: questions written after watching
the system answer them describe it instead of testing it.

### Headline

| Metric | Retrieval off | Retrieval on | Denominator |
|---|---|---|---|
| Routing — exact agent-set match | 53.3% | **53.3%** | 15 |
| Answers fully grounded | 33.3% | **33.3%** | 12 answerable |
| Ungrounded figures | 9 | **10** | — |
| Mean evidence per answer | 2.42 | **2.67** | 12 |
| Answers with no evidence at all | 3 | **3** | 12 |
| Unanswerable correctly refused | 33.3% | **33.3%** | 3 |

**Read this as a negative result on the end-to-end metrics.** Retrieval changed
the evidence count on 3 of 15 questions and moved nothing else. Reporting it as
a win would require quoting the mean-evidence column and hiding the rest.

Latency is the one clean result: every category stays inside its SRS 3.4.1
budget, and single-sector — the budget the 30-question set breaches — came in at
p95 6,993 ms against 10,000 ms. Retrieval costs 300–900 ms on the paths that use
it and is skipped entirely for FX shocks.

### Why the numbers did not move, in order of how much each matters

**1. The router never sends a policy question to the agent that can retrieve.**
P03, P04 and P05 — "what does India's Foreign Trade Policy say", "what non-tariff
measures does the EU apply", "does the Netherlands identify Sri Lanka as a
priority market" — all route to `export_analytics` **alone**, score 0 evidence
and 0.15 confidence in *both* runs. Retrieval lives in `trade_economics`
(the D10 scope decision), so a question that never reaches `trade_economics`
cannot benefit from it however good the corpus is. This is the binding
constraint and it is a routing problem, not a retrieval problem.

**2. The corpus does not cover the countries the reachable questions ask about.**
Of the 15 manifest rows, 11 downloaded and **6 extract into usable text**:
Sri Lanka, the UK, Canada, the US, India and Italy. Germany, the Netherlands,
the UAE, China and France are absent — three URLs 404, five served JavaScript
shells of 93–1,351 characters that `extract.py` rejects rather than indexing.
The simulation questions that *do* reach `trade_economics` are US- and
EU-focused, and the EU document is one of the 404s.

**3. The one US document is a link index, not a schedule.** The USTR
"Presidential Tariff Actions" page is a list of press-release titles and "to
read the tariff schedule, click here". Asked what tariff applies to Sri Lankan
knitwear, its best chunk scores **−7.91** and the retriever returns nothing.

That last one is the result worth keeping. **The retriever declining to answer
is correct behaviour, and it is the behaviour that most needed testing.** Before
the relevance floor existed, this same question cited the corpus's
ABBREVIATIONS page — which matched only because it contains the words "United
States dollars" — as evidence about US apparel tariffs. A real citation, a real
URL, a real page number, attached to a claim the page does not support. That is
precisely the failure §5 says `orchestrator/grounding.py` cannot catch.

### The retriever itself works

Measured directly, outside the orchestrator, on the documents that did extract:

| Query | Top hit | Score |
|---|---|---|
| "Does the UK trade strategy keep preferential access for developing countries?" | UK Trade Strategy — Economic Partnership Agreements | **+5.95** |
| "What does the UK DCTS do for tariffs?" | UK Trade Strategy — cumulation groups, DCTS | **+5.55** |
| "What are Canada's trade priorities and market access negotiations?" | Canada briefing book — Trade Policy / Market Access | **+6.24** |
| "What does India's foreign trade policy say about imports and exports?" | India FTP 2023 — DGFT scheme administration | **+3.65** |

Country anchoring holds under test: the India query's second-ranked hit is the
Sri Lankan strategy at **+0.22**, correctly far below the Indian document rather
than winning on general trade vocabulary.

So the machinery is sound and the corpus is the bottleneck. **The honest summary
is that D10 built a retrieval path that demonstrably works, and an evaluation
that demonstrably does not yet exercise it.**

### What to fix, in priority order

1. **Route policy questions to an agent that retrieves.** Either add
   `trade_economics` to the route when a question names a foreign trade policy,
   or move retrieval somewhere `export_analytics` can reach it. Nothing else on
   this list matters until this is done — it is what makes P03/P04/P05
   answerable at all.
2. **Replace the five JavaScript-shell URLs with the PDFs they link to.** No
   code change; the manifest header names each one.
3. **Replace the USTR landing page with the actual HTS schedule**, and fix the
   three 404s (India MoC, EU DG TRADE, Canada State of Trade).
4. **Add the per-country WTO Trade Policy Reviews.** Still the most tariff-dense
   documents available for these markets, and still not collected.

### Threats specific to this section

- **Six documents, one of them 60% of the corpus.** The Sri Lankan strategy is
  544 of 901 chunks, so any unfiltered retrieval is biased toward it. The
  country filter is what holds that in check, which makes the anchoring test
  above load-bearing rather than decorative.
- **Every retrieved figure is `unverified`.** Same status as
  `trade_agreements.csv`. No rate lifted from a document has been checked
  against an official schedule by a human.
- **The relevance floor is calibrated on one corpus.** `MIN_RERANK_SCORE = 0.0`
  is the cross-encoder's own boundary rather than a tuned constant, but the
  evidence that 0.0 separates useful from useless here is 15 questions on 6
  documents.
- **The three "unanswerable" questions are refused at 33.3% on both paths.**
  That is unchanged by D10 and is the same measurement problem §2 describes:
  worth re-checking by hand before it is quoted.

### Regression check on the 30-question set

D10 touches `trade_economics`, `kg/queries.py` and `agents/common.py`, so the
original set was re-run cold against the same stack. Compared with §1's
2026-08-28 figures:

| Metric | §1 (before D10) | After D10 |
|---|---|---|
| Routing — exact match | 53.3% | 53.3% |
| Routing — recall | 0.856 | 0.856 |
| Answers fully grounded | 77.8% | **81.5%** |
| Ungrounded figures | 6 | **5** |
| Mean evidence per answer | 3.67 | **3.93** |
| Answers with no evidence at all | 2 | **1** |
| Unanswerable correctly refused | 100% (3) | 100% (3) |
| Crashes | 0 | 0 |

**No regression, and the grounding numbers improved slightly.** The gain is not
from retrieval — it is the evidence fix retrieval work uncovered. Adding the
relevance floor removed a junk policy citation from an agreement simulation,
which dropped that answer to one evidence entry and exposed that
`trade_economics` had been citing the baseline query but never the coverage
query that supplied the MFN rate. The rate was a figure in the answer traceable
to nothing — EVALUATION.md §1 grounding class 1, in the agent that section names.
Citing the coverage query fixes it for every agreement question, D10 or not.

**Single-sector p95 still breaches SRS 3.4.1**: 11,888 ms against 10,000 ms.
That is unchanged in kind from §1's finding. It came in below §1's 14,638 ms,
but §1 already records p95 swinging between 14.6 s and 29.0 s across two cold
runs of the same questions, so this is inside that spread and is **not** evidence
of an improvement.
