SHELL := /bin/bash
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

.DEFAULT_GOAL := help
.PHONY: help up down ps logs install fmt generate features consume-once dbt-test dbt-bad-data-demo train load-online serve loadtest airflow-init airflow demo

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

dbt-test: ## (Phase 3) dbt quality gates on the silver layer (dbt-duckdb)
	uv run dbt build --project-dir dbt/signallake_dbt --profiles-dir dbt/signallake_dbt

dbt-bad-data-demo: ## (Phase 3) Inject bad rows into a scratch copy of silver, show the gates fail, clean up
	bash scripts/dbt_bad_data_demo.sh

TRAIN_ARGS ?=

train: ## (Phase 4) Train IsolationForest + 3 HGBC feature-tier runs, log/register to MLflow
	uv run python -m signallake.train.train $(TRAIN_ARGS)

load-online: ## (Phase 5) Load latest per-customer features into Redis
	uv run python -m signallake.serve.load_online_store

SERVE_PORT ?= 8000
# Multiple worker PROCESSES, not threads: /score is CPU-bound (sklearn predict_proba), and a
# single process serializes that work on Python's GIL -- under concurrent load, p95 blows up
# from queueing even though any one request is fast. Workers give each request its own GIL.
SERVE_WORKERS ?= 4

serve: ## (Phase 5) Run the FastAPI online-scoring service
	# Each worker already gets its own process/core; without this, numpy/scikit-learn's BLAS
	# and OpenMP thread pools each spin up ~nproc threads PER WORKER (measured: ~120 threads in
	# a single worker process here), so N workers oversubscribe the machine by ~N*nproc -- that
	# alone took p95 from ~10ms to >1s under concurrent load. One math thread per worker process
	# is correct here since every request is a single tiny row, not a batch that benefits from
	# intra-request parallelism.
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
		uv run uvicorn signallake.serve.app:app --host 0.0.0.0 --port $(SERVE_PORT) --workers $(SERVE_WORKERS)

loadtest: ## (Phase 5) Headless Locust load test against a running `make serve`
	uv run python -m signallake.serve.run_loadtest

# Airflow gets its OWN venv (.venv-airflow), never the uv project's .venv -- its dependency
# tree (FastAPI/Starlette/pydantic pins, etc.) can conflict with ours, and DAG files that shell
# out via BashOperator never need signallake or pyspark importable in-process anyway.
AIRFLOW_VERSION ?= 3.3.2
AIRFLOW_PYTHON ?= 3.12
# LocalExecutor needs real concurrent DB connections, which airflow standalone's own default
# (SQLite) can't safely provide -- hence the airflow-postgres service in docker-compose.yml.
# `airflow standalone` spawns its scheduler/api-server/triggerer/dag-processor as child
# subprocesses that invoke the bare `airflow` command -- PATH must have the venv's bin/ first,
# not just this recipe's own shell, or those children fail with "No such file: 'airflow'".
AIRFLOW_ENV = PATH=$(CURDIR)/.venv-airflow/bin:$$PATH \
	AIRFLOW_HOME=$(CURDIR)/airflow \
	AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost:5433/airflow \
	AIRFLOW__CORE__EXECUTOR=LocalExecutor \
	AIRFLOW__CORE__LOAD_EXAMPLES=false \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/airflow/dags

airflow-init: ## (Phase 6) Set up the dedicated Airflow venv + Postgres metadata DB
	uv venv .venv-airflow --python $(AIRFLOW_PYTHON)
	uv pip install --python .venv-airflow/bin/python \
		"apache-airflow==$(AIRFLOW_VERSION)" apache-airflow-providers-postgres \
		--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-$(AIRFLOW_VERSION)/constraints-$(AIRFLOW_PYTHON).txt"
	mkdir -p airflow/dags airflow/logs
	$(AIRFLOW_ENV) .venv-airflow/bin/airflow db migrate

airflow: ## (Phase 6) Run Airflow standalone (LocalExecutor, Postgres backend) -- watch for the printed admin login
	$(AIRFLOW_ENV) .venv-airflow/bin/airflow standalone

demo: ## (Phase 7) One-command end-to-end demo (200k events, isolated from your main dataset)
	bash scripts/demo.sh
