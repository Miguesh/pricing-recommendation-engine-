UV ?= uv
PYTHON_VERSION ?= 3.12

.PHONY: help install lock lint format format-check test test-cov build check run serve \
	demo-data train compose-config compose-build compose-up compose-down

help:
	@echo "install          Sync the locked development environment"
	@echo "lint             Run Ruff security/style checks and strict MyPy"
	@echo "test-cov         Run tests with the enforced coverage threshold"
	@echo "build            Build wheel and source distribution"
	@echo "check            Run the local release-quality gate"
	@echo "run              Start the reload-enabled local API"
	@echo "serve            Start the production-style local API"
	@echo "compose-up       Build and start the local integration platform"

install:
	$(UV) sync --locked --extra dev --python $(PYTHON_VERSION)

lock:
	$(UV) lock --python $(PYTHON_VERSION)

lint:
	$(UV) run ruff check .
	$(UV) run mypy src
	$(UV) pip check

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

format-check:
	$(UV) run ruff format --check .

test:
	$(UV) run pytest

test-cov:
	$(UV) run pytest --cov=pricing_engine --cov-report=term-missing --cov-report=xml

build:
	$(UV) build --no-sources

check: lint format-check test-cov build compose-config

run:
	$(UV) run uvicorn pricing_engine.interfaces.api.app:create_app --factory --reload --port 8000

serve:
	$(UV) run uvicorn pricing_engine.interfaces.api.app:create_app --factory --host 0.0.0.0 \
		--port 8000 --workers 1 --limit-concurrency 100 --timeout-keep-alive 5 \
		--timeout-graceful-shutdown 30 --no-access-log

demo-data:
	$(UV) run pricing-engine generate-demo-data --output data/demo/observations.parquet

train:
	$(UV) run pricing-engine train --input data/demo/observations.parquet \
		--output artifacts/local-model

compose-config:
	docker compose config --quiet

compose-build:
	docker compose build api mlflow

compose-up:
	docker compose --env-file .env up --build --detach

compose-down:
	docker compose down --remove-orphans
