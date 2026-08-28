.PHONY: up down logs test test-unit lint fmt install ingest kg-load db-init backtest eval eval-degraded eval-policy eval-policy-baseline coherence docs clean

# Where the frozen contracts come from. Sibling checkout during the sprint;
# override to pin a git ref once the repo is pushed:
#   make install CONTRACTS_SPEC="ceynex-contracts @ git+ssh://git@github.com/CeyNex-AI/ceynex-contracts.git"
CONTRACTS_SPEC ?= -e ../ceynex-contracts

# The checkout's own interpreter, not whatever `python` resolves to on PATH.
# A bare `python` outside an activated venv is the system one, which here is
# 3.10 — and ceynex-contracts requires >=3.11, so `make install` failed with a
# Python-version error while .venv sat there on 3.12. Every target that runs
# project code goes through this.
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

install:
	$(PYTHON) -m pip install $(CONTRACTS_SPEC)
	$(PYTHON) -m pip install -e ".[dev,docs]"
	$(PYTHON) -m pre_commit install

up:
	docker compose up -d
	@echo "waiting for services to report healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' ceynex-postgres)" = "healthy" ] && \
	       [ "$$(docker inspect -f '{{.State.Health.Status}}' ceynex-neo4j)" = "healthy" ] && \
	       [ "$$(docker inspect -f '{{.State.Health.Status}}' ceynex-qdrant)" = "healthy" ]; do \
	  sleep 2; done
	@echo "postgres + neo4j + qdrant healthy. adminer on http://localhost:$${ADMINER_PORT:-8080}"

down:
	docker compose down

logs:
	docker compose logs -f

test:
	$(PYTHON) -m pytest

test-unit:
	$(PYTHON) -m pytest -m "not integration"

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

# Applies ceynex-contracts' schema.sql and seeds dim_country / dim_hs. Idempotent.
# Runs against whatever POSTGRES_* in .env points at, so it serves both the local
# stack and the deployed database VM.
db-init:
	$(PYTHON) -m ceynex.data.bootstrap

ingest:
	$(PYTHON) -m ceynex.data.pipeline --sources all

kg-load:
	$(PYTHON) -m ceynex.kg.load --schema --agreements --apparel --flows

# usage: make backtest SECTOR=agriculture ITEM=cinnamon
backtest:
	$(PYTHON) -m eval.backtest --sector $(SECTOR) --item $(ITEM)

# The 30-question orchestrator evaluation. `eval-degraded` runs the same set with
# the LLM forced unavailable (SRS 3.4.3); both must complete without crashing.
eval:
	$(PYTHON) -m eval.harness --json eval_results.json

eval-degraded:
	$(PYTHON) -m eval.harness --degraded --json eval_degraded.json

# The 15-question policy-retrieval set (eval/policy_questions.yaml), separate
# from the 30 so that baseline stays comparable. Run both of these: the delta
# between them is the whole accuracy claim for policy retrieval.
eval-policy-baseline:
	CEYNEX_POLICY_RETRIEVAL=off $(PYTHON) -m eval.harness \
	  --questions eval/policy_questions.yaml --json eval_policy_baseline.json

eval-policy:
	$(PYTHON) -m eval.harness \
	  --questions eval/policy_questions.yaml --json eval_policy.json

# Blind merge-coherence sheets for three human raters. Needs `make eval` first.
coherence:
	$(PYTHON) -m eval.coherence sheet --results eval_results.json --out coherence_sheet.csv

# Governing documents -> plain text under docs/_text/ so they are greppable.
# Reads the .docx whenever one sits beside a .pdf.
docs:
	$(PYTHON) tools/docs_to_text.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
