SHELL := /bin/bash
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

.DEFAULT_GOAL := help
.PHONY: help up down ps logs install fmt generate features dbt-test train load-online serve loadtest airflow-init airflow demo

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

# ---- Phase 0: infrastructure --------------------------------------------------
up: ## Start Kafka, Redis, MLflow and wait until healthy
	@mkdir -p mlartifacts data
	docker compose up -d --build --wait
	@docker compose ps

down: ## Stop the stack (volumes are kept)
	docker compose down

ps: ## Show service status
	docker compose ps

logs: ## Tail service logs
	docker compose logs -f --tail=100

install: ## Create .venv and install deps with uv
	uv sync

fmt: ## Format and lint-fix with ruff
	uv run ruff format .
	uv run ruff check --fix .

# ---- Placeholders for later phases -------------------------------------------
# Override on the command line, e.g. `make generate NUM_EVENTS=50000` or `make generate GEN_ARGS=--to-kafka`.
NUM_EVENTS ?= 2100000
GEN_ARGS ?=

generate: ## (Phase 1) Synthetic events -> data/bronze parquet (add GEN_ARGS=--to-kafka to publish)
	uv run python -m signallake.generate.producer --num-events $(NUM_EVENTS) --to-parquet $(GEN_ARGS)

FEATURES_ARGS ?=

features: ## (Phase 2) Spark bronze -> silver -> gold (34 features). Override with FEATURES_ARGS="--driver-memory 6g"
	uv run python -m signallake.features.build_features $(FEATURES_ARGS)

consume-once: ## (Phase 2) Bounded demo: drain Kafka `events` into data/bronze_stream_demo
	uv run python -m signallake.streaming.consume --once --output-dir data/bronze_stream_demo

dbt-test: ## (Phase 3) dbt quality gates
	@echo "not implemented yet: Phase 3"; exit 1

train: ## (Phase 4) Train + log to MLflow
	@echo "not implemented yet: Phase 4"; exit 1

load-online: ## (Phase 5) Load features into Redis
	@echo "not implemented yet: Phase 5"; exit 1

serve: ## (Phase 5) Run the FastAPI service
	@echo "not implemented yet: Phase 5"; exit 1

loadtest: ## (Phase 5) Locust load test
	@echo "not implemented yet: Phase 5"; exit 1

airflow-init: ## (Phase 6) Set up the Airflow venv
	@echo "not implemented yet: Phase 6"; exit 1

airflow: ## (Phase 6) Run Airflow standalone
	@echo "not implemented yet: Phase 6"; exit 1

demo: ## (Phase 7) One-command end-to-end demo
	@echo "not implemented yet: Phase 7"; exit 1
