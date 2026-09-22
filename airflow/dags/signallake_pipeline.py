"""Phase 6: the SignalLake pipeline DAG -- ingest -> validate -> build_features -> train -> publish.

Runs entirely as BashOperator tasks invoking `uv run python -m signallake.<module>` (or `dbt`)
in the MAIN PROJECT'S venv (`.venv`), never importing `signallake` or `pyspark` directly into
this file -- this file is parsed by the Airflow SCHEDULER, which runs in the SEPARATE Airflow
venv (`.venv-airflow`) that intentionally does not have those (heavy, Spark-pulling)
dependencies installed. Keeping the two venvs cleanly separated is why every task shells out.

Partition-awareness: this DAG is scheduled `@daily`, so every run has a distinct
`execution_date` (Airflow's `{{ ds }}`), threaded into each task as SIGNALLAKE_EXECUTION_DATE
for observability in the task logs. Said plainly: `build_features` and `train` currently
rebuild from the FULL historical dataset each run (that's what those modules do today, and is
correct for training -- a model should see all history). Partition-SCOPED incremental
processing, where only new/late-arriving partitions get reprocessed, is what
`recompute_benchmark.py` specifically measures and demonstrates; this DAG is the orchestration
skeleton those savings would plug into.
"""

import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT_ROOT = Path(os.environ["AIRFLOW_HOME"]).parent

default_args = {
    "owner": "signallake",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="signallake_pipeline",
    description="ingest -> validate (dbt) -> build_features -> train -> publish (Redis)",
    default_args=default_args,
    schedule="@daily",
    start_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    catchup=False,
    tags=["signallake", "phase6"],
) as dag:
    env = {"SIGNALLAKE_EXECUTION_DATE": "{{ ds }}"}

    ingest = BashOperator(
        task_id="ingest",
        bash_command="echo partition={{ ds }}; uv run python -m signallake.streaming.consume "
        "--once --output-dir data/bronze_stream_demo",
        cwd=str(PROJECT_ROOT),
        env=env,
        append_env=True,
        doc_md="Bounded Kafka -> bronze_stream_demo drain (Phase 2 Part A). Completes "
        "immediately if the topic has nothing new; demonstrates the real-time ingestion "
        "step this DAG orchestrates.",
    )

    validate = BashOperator(
        task_id="validate",
        bash_command="uv run dbt build --project-dir dbt/signallake_dbt "
        "--profiles-dir dbt/signallake_dbt",
        cwd=str(PROJECT_ROOT),
        env=env,
        append_env=True,
        doc_md="dbt quality gates on the silver layer (Phase 3). Fails the DAG run if any "
        "gate fails -- see dbt/signallake_dbt/models/staging/_stg_silver.yml.",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command="uv run python -m signallake.features.build_features",
        cwd=str(PROJECT_ROOT),
        env=env,
        append_env=True,
        doc_md="Spark bronze -> silver -> gold (34 features), Phase 2 Part B. Rebuilds the "
        "full dataset -- see module docstring for why.",
    )

    train = BashOperator(
        task_id="train",
        bash_command="uv run python -m signallake.train.train",
        cwd=str(PROJECT_ROOT),
        env=env,
        append_env=True,
        doc_md="Trains IsolationForest baseline + 3 HGBC feature-tier runs, registers the "
        "best in MLflow (Phase 4).",
    )

    publish = BashOperator(
        task_id="publish",
        bash_command="uv run python -m signallake.serve.load_online_store",
        cwd=str(PROJECT_ROOT),
        env=env,
        append_env=True,
        doc_md="Pushes the latest per-customer gold feature vector into Redis for online "
        "serving (Phase 5).",
    )

    ingest >> validate >> build_features >> train >> publish
