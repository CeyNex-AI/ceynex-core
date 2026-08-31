# Contract proposal — `:PolicyDocument` in `schema.cypher`

**Status: NOT APPLIED.** This is a proposed change to `ceynex-contracts`, which is
frozen and requires M1's and M3's approval. Nothing in this repo has edited
`ceynex/contracts/schema/schema.cypher`. Circulate this, get two approvals, then
open the PR against `ceynex-contracts`.

Raised by M2, 2026-08-28, for deviation D10 (policy-document retrieval).

---

## What is being added

One node label and three relationship types.

`schema.cypher` declares constraints and indexes, not edge types, so only the two
statements below actually go in the file. The three edges are listed anyway
because the SAD's rule is "six node types, four relationship types; do not
improvise" — the edge count is part of what is being approved even though it
appears nowhere in the file.

```cypher
// --- Policy documents -----------------------------------------------------
// SRS 3.1.9, deviation D10.  The node is a POINTER, not the text: doc_id,
// provenance, and which Qdrant collection holds the document's chunks.  The
// text lives only in Qdrant.  The graph's job is to answer "which documents
// could possibly be relevant" in Cypher, before any vector search runs — an
// unanchored similarity search over trade-policy text returns the right topic
// from the wrong country, because these documents all read alike.
CREATE CONSTRAINT policy_document_id IF NOT EXISTS
  FOR (p:PolicyDocument) REQUIRE p.doc_id IS UNIQUE;

// policy_documents_for() filters on the issuing country before anything else.
CREATE INDEX policy_document_iso3 IF NOT EXISTS
  FOR (p:PolicyDocument) ON (p.iso3);
```

New relationship types:

| Edge | Meaning | Created by |
|---|---|---|
| `(:PolicyDocument)-[:ISSUED_BY]->(:Country)` | who published it | `MATCH` on both — never `MERGE`s a Country |
| `(:PolicyDocument)-[:APPLIES_TO]->(:HSCode)` | goods the document is scoped to | `MERGE`s the HSCode, as the coverage loader already does |
| `(:PolicyDocument)-[:DESCRIBES]->(:TradeAgreement)` | the agreement it sets out | `MATCH` on both |

Node properties: `doc_id`, `title`, `publisher`, `url`, `doc_type`, `iso3` (list),
`language`, `published`, `retrieved_at`, `sha256`, `verified`,
`qdrant_collection`, `indexed`, `chunk_count`.

## Why it needs a contract change at all

The alternative considered and rejected was hanging policy provenance off the
existing `TradeAgreement` nodes as extra properties, which needs no approval.
It does not work:

- Most of the corpus **describes no agreement**. The UK Trade Strategy, the UAE
  Export Development Policy and Canada's trade briefing book are trade policy
  without being about a named agreement, and there is no `TradeAgreement` node to
  attach them to.
- A document is issued by a **country**, and `TradeAgreement` has no country
  edge — only a `partners` string property.
- `chunk_count` and `qdrant_collection` are properties of a *document*, and
  putting them on an agreement node would mean an agreement covered by two
  documents can hold only one of the two.

## Why the text is not in the graph

Neo4j Community would store 544 chunks of text fine. It is kept out because the
two stores are being asked different questions: Neo4j answers "which documents
are about the United States and HS 61" in Cypher, and Qdrant answers "which
passage of those documents is about this question" by similarity. Putting the
text in both means two copies that can disagree about what the corpus contains.

## Migration and blast radius

- **Purely additive.** No existing label, property, constraint or index changes.
- **`IF NOT EXISTS`**, like every other statement in the file, so re-applying is
  a no-op and the three of us can keep loading into one graph.
- **Nothing existing reads these labels.** No current query, loader or agent
  matches `:PolicyDocument`; without the change they simply do not exist and
  `policy_documents_for()` returns no rows, which the Trade Economics agent
  already treats as "no policy evidence available".
- **The loader works without the constraint**, so this PR is not blocking. It
  merges on `doc_id` regardless; the constraint upgrades a convention into a
  guarantee and makes a duplicate `doc_id` fail at write time rather than
  silently producing two nodes.

## What M1 and M3 should check

1. That `ISSUED_BY` never `MERGE`s a `Country`. `dim_country` and the trade-flow
   loaders own country creation, and a bare Country node created here would have
   no `m49` and would collide with the `country_m49` uniqueness constraint on the
   next real load. `ceynex/kg/loaders/policy_documents.py::MERGE_ISSUED_BY` uses
   `MATCH` on both sides for exactly this reason.
2. That `APPLIES_TO` `MERGE`-ing an `HSCode` is acceptable — it matches what
   `MERGE_COVERAGE` in the trade-agreement loader already does.
3. Whether `DESCRIBES` is the right edge name, or whether it should be
   `DOCUMENTS` / `SETS_OUT`. No strong opinion here; it is the one naming choice
   in the proposal that is arbitrary.
