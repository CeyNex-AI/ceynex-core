.PHONY: up down logs test test-unit lint fmt install ingest kg-load news-refresh db-init backtest eval eval-degraded eval-repeat eval-repeat-cited eval-chat eval-chat-degraded eval-policy eval-policy-baseline coherence docs clean

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

# The policy retriever's models come from the Hugging Face Hub on first use
# (into /tmp/fastembed_cache, which does not survive a reboot). Measured
# 2026-09-11: the Hub's xet transfer path stalled indefinitely on this network
# with 0-byte blobs and idle sockets, which hung `make eval` for 17 minutes
# before its first model call; the plain HTTP path downloaded the same files at
# ~320 KB/s. Off unless the environment says otherwise.
export HF_HUB_DISABLE_XET ?= 1

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
	$(PYTHON) -m ceynex.kg.load --schema --agreements --apparel --flows --policy

# One news refresh, by hand (D11). The API runs this hourly on its own, but a
# cold ceynex_news makes the demo's degraded path look broken -- run this once
# before showing anything. `--dry-run` prints the GDELT URLs without calling.
news-refresh:
	$(PYTHON) -m ceynex.news.refresh --once --ensure-collection

# usage: make backtest SECTOR=agriculture ITEM=cinnamon
backtest:
	$(PYTHON) -m eval.backtest --sector $(SECTOR) --item $(ITEM)

# The 30-question orchestrator evaluation. `eval-degraded` runs the same set with
# the LLM forced unavailable (SRS 3.4.3); both must complete without crashing.
eval:
	$(PYTHON) -m eval.harness --json eval_results.json

eval-degraded:
	$(PYTHON) -m eval.harness --degraded --json eval_degraded.json

# The repeated-run protocol (docs/EVALUATION.md §8-§9): three cold runs, medians
# and spread, and the questions that disagreed with themselves. `-cited` is the
# same with inline citations on, which is how the flag is decided rather than
# guessed. Each run clears the prompt cache first, so every call is paid for.
REPEAT ?= 3
eval-repeat:
	$(PYTHON) -m eval.harness --repeat $(REPEAT) --cold --json-dir eval_runs/off

eval-repeat-cited:
	CEYNEX_CITATIONS=on $(PYTHON) -m eval.harness --repeat $(REPEAT) --cold --json-dir eval_runs/on

# The multi-turn set (eval/conversations.yaml): follow-ups, the classifier, the
# clarification gate and the streamed answer, driven through the turn runner
# in-process against the real store. Needs the docker stack.
eval-chat:
	$(PYTHON) -m eval.chat_harness --cold --json eval_chat.json

eval-chat-degraded:
	$(PYTHON) -m eval.chat_harness --degraded --json eval_chat_degraded.json

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
