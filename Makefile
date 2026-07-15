.PHONY: install lint format test test-cov run demo-data train compose-up compose-down

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

lint:
	ruff check .
	mypy src

format:
	ruff format .

test:
	pytest

test-cov:
	pytest --cov=pricing_engine --cov-report=term-missing --cov-report=xml

run:
	uvicorn pricing_engine.interfaces.api.app:create_app --factory --reload

demo-data:
	pricing-engine generate-demo-data --output data/demo/observations.parquet

train:
	pricing-engine train --input data/demo/observations.parquet

compose-up:
	docker compose --env-file .env up --build

compose-down:
	docker compose down --remove-orphans
