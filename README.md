# ceynex-core

Multi-agent decision intelligence platform for Sri Lanka's national export
economy — Group 07, Project P16, CS3501, University of Moratuwa.

This repo holds the agent layer (LangGraph graph over five domain agents:
`export_analytics`, `agriculture_commodity`, `apparel_manufacturing`,
`trade_economics`, `forecast`), the knowledge layer (Neo4j via
`ceynex/kg/client.py`), the data layer (Postgres/Parquet via
`ceynex/data/writer.py`), retrieval, LLM integration, and the orchestrator.
Contract types (`ceynex.contracts`) live in the sibling `ceynex-contracts`
repo and are installed as a dependency — never edited here.

Governing documents are in `docs/SRS/`, `docs/SAD/`, and
`CeyNex_plan_updated/`; `make docs` converts them to greppable text.

## Setup

```bash
make install      # ceynex-contracts from ../ceynex-contracts, then this package
make up            # docker compose up, waits for postgres + neo4j healthy
make db-init        # apply schema.sql + seed dim_country / dim_hs
make kg-load         # schema.cypher + TradeAgreement nodes
```

## Common commands

```bash
make test          # pytest (all)
make test-unit      # pytest -m "not integration" — no docker needed
make lint            # ruff check
make ingest           # python -m ceynex.data.pipeline --sources all
make eval              # the 30-question orchestrator evaluation
```

See `CLAUDE.md` for the full layer rules, agent node contract, and eval setup.
