# Bug report — "shipping costs" query returns a misleading out-of-scope refusal

**Component:** `ceynex-core` — orchestrator router / merger
**Reported:** 2026-09-03
**Severity:** Medium (wrong-looking answer on a reasonable trade question; user-visible on the live demo VM)
**Area owner:** M2 (core systems)
**Status: FIXED in code (Fix A + B + C), pending live verification.**

> **What shipped.** Fix A — `llm_route()` now populates `decision.notes` for both
> out-of-scope shapes, from module-level constants (`SCOPE_SENTENCE`,
> `NO_TOPIC_NOTE`, `MIXED_SCOPE_NOTE`) shared with `keyword_route()` so the two
> cannot drift again. Fix B — `merger.py`'s empty-note fallback no longer asserts
> a sector was named. Fix C — freight is named in `OUT_OF_SCOPE_WORDS`, so the
> refusal states the real reason instead of "names nothing CeyNex covers".
>
> **The §5/D scope decision was taken: freight is out of bounds for now.** Not
> because it is uninteresting but because CeyNex holds no freight data and a
> narrative answer with an empty evidence panel is not what this system is for.
> Integrating UNCTAD's Trade-and-Transport Dataset (`US.TransportCosts`; free
> bulk + API, Origin/Destination M49 × HS-2017 4-digit × mode × year, ad-valorem
> and per-unit transport costs) is scoped as its own effort. Two constraints make
> it a separate piece of work: its coverage ends in 2021 while `fact_trade` now
> runs to 2025, so it answers a structural question rather than "what are rates
> doing now"; and `fact_trade` is frozen with no transport-cost measure, so it
> has to be staged-only — the same treatment production data already gets, and
> for the same reason.
>
> Fix E (frontend) is M3's and is not addressed here.

---

## 1. Summary

Asking **"How are shipping costs affecting Sri Lankan exporters?"** returns:

> **Very low confidence · 15%**
> "This question could not be answered from the data currently loaded. **Part of the question names a sector CeyNex does not cover.**"
> Evidence: *No supporting evidence for this answer.*

Two things are wrong:

1. **The refusal text is factually incorrect.** The question names **no** excluded sector
   (gems, tourism, fisheries, …). "Shipping" is not a sector. The sentence shown is a
   generic hard-coded fallback string, not a real explanation of scope.
2. **The refusal is triggered by a routing gap**, not a deliberate scope decision.
   Neither router recognises "shipping / freight / logistics" as a topic, so the
   query falls through to the "no topic recognised" branch and is flagged
   `out_of_scope`. The system never attempts an answer, even though
   *Freight & shipping* is a surfaced GLOBAL trending topic on the same page and
   GDELT found related news.

There is also a **latent bug** behind #1 that affects **every** LLM-router
out-of-scope verdict, not just this query (see §4.2).

---

## 2. Reproduction

- **Environment:** live query VM (`https://35.200.228.142/query`), current `main`.
- **Query:** `How are shipping costs affecting Sri Lankan exporters?`
- **Observed:** confidence `0.15`, `refused: true`, `evidence_count: 0`,
  answer = `"This question could not be answered from the data currently loaded. Part of the question names a sector CeyNex does not cover."`
- **Expected:** either
  - (a) an honest scope message that correctly states what CeyNex covers and does
    **not** claim an excluded sector was named, **or**
  - (b) a genuine attempt at a qualitative answer (freight as a cost lever on
    tea / apparel competitiveness) with the related news attached.

Reproduces offline too — `keyword_route()` alone flags this query out-of-scope
(see §3, step 1), so it is not dependent on the LLM router's output.

---

## 3. Root-cause trace

File paths are relative to `ceynex-core/`.

### Step 1 — the router flags the query `out_of_scope`

`ceynex/orchestrator/router.py`

`keyword_route()` scans the query against six keyword groups:

| Group | Constant | Matches in this query? |
|---|---|---|
| Agriculture | `AGRICULTURE_WORDS` (`router.py:33`) | no |
| Apparel | `APPAREL_WORDS` (`router.py:37`) | no |
| Simulation / policy | `SIMULATION_WORDS` (`router.py:41`), `POLICY_WORDS` (`router.py:56`) | no ("affecting" ≠ "impact of") |
| Forecast | `FORECAST_WORDS` (`router.py:62`) | no |
| Analytics | `ANALYTICS_WORDS` (`router.py:66`) | no |
| Excluded sectors | `OUT_OF_SCOPE_WORDS` (`router.py:75`) | no |

With every group empty:

```python
# router.py:132-140
no_topic_recognized = not (
    hits_agriculture or hits_apparel or wants_simulation
    or wants_forecast or wants_analytics or named_out_of_scope
)                                   # -> True
out_of_scope = bool(named_out_of_scope) or no_topic_recognized   # -> True
```

So `keyword_route()` returns `out_of_scope=True`, `no_topic_recognized=True`.
The LLM router (`llm_route()`) independently also returned `out_of_scope: true`
for this query (its prompt only whitelists commodities, garments, and
destination-market policy — `ROUTER_SYSTEM`, `router.py:251-284`).

### Step 2 — the graph turns the decision into an `errors` entry

`ceynex/orchestrator/graph.py:116-127`

```python
if decision.out_of_scope:
    errors = [f"out_of_scope: {decision.notes[0] if decision.notes else ''}"]
    if decision.no_topic_recognized:
        errors.append("out_of_scope_no_topic: true")
    patch["errors"] = errors
```

- `keyword_route()` **does** populate `decision.notes` — `router.py:220-224`
  appends *"the question does not name anything CeyNex covers. Scope is
  agriculture (tea, cinnamon, rubber, coconut) and apparel (HS 61/62) exports —
  SRS 2.4"*.
- `llm_route()` **never populates `decision.notes`** — `router.py:332-340`
  constructs the `RouteDecision` with no `notes=` argument, so it defaults to `[]`.

When the LLM router runs (the default — `build_graph(..., use_llm_router=True)`,
`graph.py:110`), `decision.notes` is empty, so:

```python
errors == ["out_of_scope: ", "out_of_scope_no_topic: true"]
#                        ^ empty note
```

### Step 3 — the merger substitutes a hard-coded string

`ceynex/orchestrator/merger.py`

```python
OUT_OF_SCOPE_PREFIX = "out_of_scope: "          # merger.py:488

def _out_of_scope_gaps(state):                  # merger.py:514-529
    gaps = []
    for error in state.get("errors", []) or []:
        text = str(error)
        if text.startswith(OUT_OF_SCOPE_PREFIX):
            note = text[len(OUT_OF_SCOPE_PREFIX):].strip()     # -> ""
            gaps.append(note or "part of the question names a sector CeyNex does not cover")
    return gaps
```

With an empty `note`, the `or` fallback fires and the gap becomes the literal
string **"part of the question names a sector CeyNex does not cover"** — which is
what the user sees, capitalised, in `_nothing_succeeded()` (`merger.py:728-736`):

```
"This question could not be answered from the data currently loaded. "
"Part of the question names a sector CeyNex does not cover."
```

### Step 4 — no evidence, confidence floored

`no_topic_recognized` suppression (`merger.py:497-511`, `no_topic_recognized()`)
discards the routed agent's output as noise, so `merged_evidence == []` and the
answer is marked `refused` at the `0.15` floor.

---

## 4. What is wrong, precisely

### 4.1 Content scope gap

`grep -ri "freight|shipping|logistic|container" ceynex-core/ceynex` returns
**nothing** functional. CeyNex holds:

- the export knowledge graph (values / volumes / markets / districts / prices for
  tea, cinnamon, rubber, coconut and HS 61/62 apparel), and
- the D10 destination-market trade-policy document corpus.

It has **no freight-rate / shipping-cost / logistics data**. So a fully grounded,
figure-backed answer to this question is genuinely not possible today — but the
current behaviour communicates that badly.

### 4.2 Latent bug: `llm_route()` never sets `decision.notes`

This is the real defect and it is **not** specific to shipping. **Any** query the
LLM router marks `out_of_scope` — mixed ("tea vs fisheries") or no-topic
("who is Euler") — produces `errors = ["out_of_scope: "]` with an empty note, so
the merger always falls back to the generic *"part of the question names a sector
CeyNex does not cover"* line. That line is:

- **wrong for no-topic queries** (they name no sector at all), and
- **wrong for mixed queries** where the excluded thing is not a "sector"
  (e.g. "shipping costs", "the weather").

`keyword_route()` gets this right; `llm_route()` regressed it by omitting the
parity code.

### 4.3 Routing gap: "logistics / freight" is unclassified

Freight cost is a legitimate export-competitiveness question and overlaps
`trade_economics`' remit (cost levers on export revenue). Today it matches no
keyword group and no LLM whitelist entry, so it is silently pushed to
"no topic recognised" rather than being either answered or explicitly declined.

---

## 5. Recommended fixes

Ordered by effort. **Fix A and B are low-risk and should ship together.**

### Fix A — give `llm_route()` the same `notes` parity as `keyword_route()`

`ceynex/orchestrator/router.py`, in `llm_route()` before the final `return`
(around `router.py:330`):

```python
notes: list[str] = []
if out_of_scope:
    if no_topic_recognized:
        notes.append(
            "the question does not name anything CeyNex covers. Scope is agriculture "
            "(tea, cinnamon, rubber, coconut) and apparel (HS 61/62) exports — SRS 2.4"
        )
    else:
        notes.append(
            "part of the question is outside what CeyNex covers. Scope is agriculture "
            "(tea, cinnamon, rubber, coconut) and apparel (HS 61/62) — SRS 2.4"
        )

return RouteDecision(
    ...
    notes=notes,
)
```

Effect: the shipping query now returns *"…This question could not be answered
from the data currently loaded. The question does not name anything CeyNex
covers. Scope is agriculture (tea, cinnamon, rubber, coconut) and apparel
(HS 61/62) exports — SRS 2.4."* — accurate, and it stops reading like a bug.
Also fixes every other LLM-router out-of-scope answer.

**Consider** factoring the two scope sentences into module-level constants shared
by both routers so they cannot drift apart again.

### Fix B — soften the merger's fallback string

`ceynex/orchestrator/merger.py:528` — the `or` fallback should not assert a
"sector" was named, since by the time we are here the note is missing:

```python
gaps.append(note or "part of the question is outside what CeyNex covers")
```

Belt-and-braces for any future path that reaches the merger with an empty note.

### Fix C — name freight/shipping explicitly as excluded (small)

`ceynex/orchestrator/router.py:75`, extend `OUT_OF_SCOPE_WORDS`:

```python
OUT_OF_SCOPE_WORDS = (
    "gem", "sapphire", "tourism", "tourist", "remittance", "fisheries", "fish ",
    "cement", "petroleum", "software export", "it export", "bpo",
    "freight", "shipping cost", "shipping costs", "ocean freight", "sea freight",
    "container rate", "container rates", "logistics cost", "logistics costs",
)
```

Effect: the query routes through the `named_out_of_scope` branch
(`router.py:209-219`) and the message becomes the specific *"the question is
about freight/shipping, which CeyNex does not cover. Scope is agriculture … —
SRS 2.4"*. Do this **only if** the team's decision (see Fix D) is that freight is
out of bounds.

> ⚠️ Word-boundary check: `"fish "` already relies on the trailing space and the
> ` {query} ` padding in `keyword_route()` (`router.py:109`). Add `"shipping"`
> bare only if you are comfortable it will not collide with e.g. "drop-shipping
> apparel" style phrasing — the two-word `"shipping cost(s)"` entries above are
> safer.

### Fix D — (scope decision, needs M2 + product) treat freight as an in-scope cost lever

If CeyNex should *attempt* a qualitative answer instead of declining:

- add a `LOGISTICS_WORDS` group in `router.py` (`"freight"`, `"shipping cost"`,
  `"ocean/sea freight"`, `"container rate"`, `"port congestion"`, `"logistics"`);
- on a hit, route to `trade_economics` (relevance ~0.8) **plus** both sector
  agents (~0.6), and **do not** set `out_of_scope` / `no_topic_recognized`;
- `trade_economics` answers narratively (freight rates compress exporter margins;
  here is what tea / apparel export values did over the period) and the GDELT
  related-news panel supplies context.

Trade-off: with no freight time series, confidence stays modest and the evidence
panel stays thin — but the user gets analysis instead of a dead end. Requires
`trade_economics` to actually have cost-pass-through logic; confirm with M2
before committing.

### Fix E — (frontend, M3) de-emphasise scope refusals in the UI

When `refused` / very-low-confidence **and** the gap is a scope note, render the
scope statement + the existing "Try asking" suggestions as the primary content
rather than a red error-style card. No core change needed; complements A–D.

---

## 6. Suggested PR breakdown (ceynex-core = 1 PR per feature, stacked)

1. **PR 1 — router `notes` parity + shared scope constants** (Fix A + B).
   Tests: extend `tests/orchestrator/test_router.py` with an `llm_route`
   out-of-scope case asserting `decision.notes` is non-empty for both the
   no-topic and mixed shapes; extend `tests/orchestrator/test_merger.py`
   (near `test_merger.py:393`) to assert the surfaced gap for an empty-note
   `out_of_scope:` error no longer says "names a sector".
2. **PR 2 — freight/shipping classification** (Fix C *or* Fix D, per §5/D
   decision). Add router unit tests for the shipping query; add an eval-set
   question and update `docs/EVALUATION.md`.
3. **Frontend** (Fix E) — separate change in `ceynex-web`.

---

## 7. Acceptance criteria

- [ ] `"How are shipping costs affecting Sri Lankan exporters?"` no longer
      returns the string *"Part of the question names a sector CeyNex does not
      cover."*
- [ ] The returned message correctly states CeyNex's scope and does not claim an
      excluded **sector** was named when none was.
- [ ] `llm_route()` populates `decision.notes` for both out-of-scope shapes;
      covered by a unit test.
- [ ] Team decision recorded (Fix C vs Fix D) for how freight/logistics queries
      are handled going forward.
- [ ] Verified with a real query against the live VM, not just a merged PR.

---

## 8. Appendix — key code references

| What | Location |
|---|---|
| Keyword groups incl. `OUT_OF_SCOPE_WORDS` | `ceynex/orchestrator/router.py:33-78` |
| `keyword_route()` out-of-scope logic | `ceynex/orchestrator/router.py:124-140` |
| `keyword_route()` populates `notes` | `ceynex/orchestrator/router.py:209-224` |
| `llm_route()` builds `RouteDecision` **without** `notes` | `ceynex/orchestrator/router.py:332-340` |
| LLM router system prompt / scope rules | `ceynex/orchestrator/router.py:251-284` |
| Router mode selection (`use_llm_router=True` default) | `ceynex/orchestrator/graph.py:110`, `graph.py:188-197` |
| Decision → `errors` (empty-note bug origin) | `ceynex/orchestrator/graph.py:116-127` |
| `OUT_OF_SCOPE_PREFIX` / `NO_TOPIC_MARKER` | `ceynex/orchestrator/merger.py:488-489` |
| `_out_of_scope_gaps()` fallback string | `ceynex/orchestrator/merger.py:514-529` |
| `no_topic_recognized()` evidence suppression | `ceynex/orchestrator/merger.py:497-511` |
| `_nothing_succeeded()` final-answer assembly | `ceynex/orchestrator/merger.py:728-736` |
| Prior related eval finding (P03/P04 out-of-scope regression) | `docs/EVALUATION.md:534-540` |
