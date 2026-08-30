# Policy retrieval — plain guide

What it is, how to run it, and where it stops. Full reasoning is in
[ARCHITECTURE_DELTA.md](ARCHITECTURE_DELTA.md) (D10) and
[EVALUATION.md](EVALUATION.md) §7.

## What it does

CeyNex used to know only Sri Lanka's own trade numbers. It could not say what
the US, the UK or India actually *does* — their tariffs, agreements, or rules.

Policy retrieval fixes that. We downloaded trade-policy documents from Sri
Lanka's biggest export markets, cut them into passages, and stored them in a
search engine called **Qdrant**. When someone asks a policy question, the
`trade_economics` agent looks up the relevant passages and quotes them, with a
link back to the source page.

## How a question flows

```
"What tariff would the US apply to Sri Lankan knitwear?"
   |
   1. Neo4j: which documents are about the US and HS 61?   -> a shortlist
   2. Qdrant: which passages in that shortlist answer this? -> top 5
   3. Rerank: are they actually relevant?                   -> drop weak ones
   4. Answer quotes the surviving passages, with URL + page
```

**Step 1 is the important one.** Trade-policy documents all sound alike, so
searching them all at once returns the right topic from the wrong country. We
ask the knowledge graph *first* which documents are even eligible, then search
only inside those.

## Two kinds of question

| Question | What happens |
|---|---|
| "What if Sri Lanka loses GSP+?" | A **simulation**. Produces a number, and uses a tariff rate from a document if one states it clearly. |
| "What does India's trade policy say?" | A **description**. Quotes the documents. **Produces no number at all** — nothing was changed, so there is nothing to calculate. |

The second case exists because without it the agent treated every unrecognised
question as a currency shock and invented a figure for it.

## Running it

Rebuild the document corpus (offline, occasional):

```bash
cd trade-data-pipeline
PY=../ceynex-core/.venv/bin/python
$PY fetch.py --update-manifest   # download
$PY extract.py --use-ocr         # PDFs/HTML -> text
$PY chunk.py && $PY enrich.py    # split into passages, tag them
$PY index.py                     # load into Qdrant
```

Then point the graph at it:

```bash
cd ceynex-core
make up          # now starts Qdrant too
make kg-load     # creates the PolicyDocument nodes
```

Ask something:

```bash
python -m ceynex.orchestrator.demo "Does the UK trade strategy keep preferential access for Sri Lankan tea?"
```

## Turning it off

Set `CEYNEX_POLICY_RETRIEVAL=off` in `.env`, or leave `QDRANT_URL` unset. The
system then behaves exactly as it did before this feature existed — it does not
break, it just stops quoting documents. `make eval-policy-baseline` uses this to
measure the before-and-after.

## What is in the corpus today

6 documents, 901 passages: **Sri Lanka, UK, Canada, US, India, Italy**.

Missing: **Germany, Netherlands, UAE, China, France**. Five of the URLs return a
page shell with no text in it, and three return 404. The fix is to point those
rows at the PDF instead of the web page — no code change. The list and reasons
are in the header of `ceynex/data/reference/policy_documents.csv`.

## Things to know

- **Nothing here is verified.** Every document is marked `unverified`, like the
  trade-agreement tables. A rate quoted from a document has not been checked
  against an official schedule by a human.
- **It says "I don't know" often, on purpose.** If no passage clearly answers the
  question, it returns nothing rather than quoting the closest match. An
  abbreviations page cited as a tariff source looks exactly like a real answer.
- **It only adds to `trade_economics`.** The other four agents are unchanged.
- **The graph is still the source of truth** for which agreements cover which
  goods. Documents corroborate it; they do not overrule it.
