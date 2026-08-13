.PHONY: up down logs test test-unit lint fmt install ingest kg-load backtest clean

install:
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

ingest:
	python -m ceynex.data.pipeline --sources all

kg-load:
	python -m ceynex.kg.load --schema --agreements

# usage: make backtest SECTOR=agriculture ITEM=cinnamon
backtest:
	python -m eval.backtest --sector $(SECTOR) --item $(ITEM)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
