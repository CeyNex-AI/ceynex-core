.PHONY: up down logs test test-unit lint fmt install ingest kg-load db-init backtest docs clean

# Where the frozen contracts come from. Sibling checkout during the sprint;
# override to pin a git ref once the repo is pushed:
#   make install CONTRACTS_SPEC="ceynex-contracts @ git+ssh://git@github.com/CeyNex-AI/ceynex-contracts.git"
CONTRACTS_SPEC ?= -e ../ceynex-contracts

install:
	python -m pip install $(CONTRACTS_SPEC)
	python -m pip install -e ".[dev]"
	pre-commit install

up:
	docker compose up -d
	@echo "waiting for services to report healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' ceynex-postgres)" = "healthy" ] && \
	       [ "$$(docker inspect -f '{{.State.Health.Status}}' ceynex-neo4j)" = "healthy" ]; do \
	  sleep 2; done
	@echo "postgres + neo4j healthy. adminer on http://localhost:$${ADMINER_PORT:-8080}"

down:
	docker compose down

logs:
	docker compose logs -f

test:
	pytest

test-unit:
	pytest -m "not integration"

lint:
	ruff check .

fmt:
	ruff check --fix . && ruff format .

# Applies ceynex-contracts' schema.sql and seeds dim_country / dim_hs. Idempotent.
# Runs against whatever POSTGRES_* in .env points at, so it serves both the local
# stack and the deployed database VM.
db-init:
	python -m ceynex.data.bootstrap

ingest:
	python -m ceynex.data.pipeline --sources all

kg-load:
	python -m ceynex.kg.load --schema --agreements

# usage: make backtest SECTOR=agriculture ITEM=cinnamon
backtest:
	python -m eval.backtest --sector $(SECTOR) --item $(ITEM)

# Governing documents -> plain text under docs/_text/ so they are greppable.
# Reads the .docx whenever one sits beside a .pdf.
docs:
	python tools/docs_to_text.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
