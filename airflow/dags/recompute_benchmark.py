"""Phase 6: incremental-vs-full recompute benchmark DAG.

Three BashOperator tasks, chained sequentially to avoid two concurrent Spark jobs fighting
over this machine's cores (see signallake.serve -- Phase 5 hit exactly that kind of
contention): full_recompute >> incremental_recompute >> report_reduction. Each shells out to
`signallake.orchestration.recompute_benchmark` in the MAIN project's venv, same reasoning as
signallake_pipeline.py (this file itself only imports `airflow.*`, since it's parsed by the
Airflow scheduler running in the separate `.venv-airflow`).

`report_reduction` reads both scenarios' result JSON, prints full vs incremental work units +
wall-clock + the real reduction percentage, and appends that to reports/METRICS.md -- see
`signallake.orchestration.recompute_benchmark`'s module docstring for the work-unit definition
and exactly how the incremental partition set is chosen.
"""

import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT_ROOT = Path(os.environ["AIRFLOW_HOME"]).parent
ARTIFACTS_DIR = "reports/_benchmark_artifacts"

default_args = {
    "owner": "signallake",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="recompute_benchmark",
    description="Measure recompute cost: full rebuild vs. incremental (watermark) rebuild",
    default_args=default_args,
    schedule=None,  # run on demand (`airflow dags trigger recompute_benchmark`), not scheduled
    start_date=pendulum.datetime(2026, 8, 1, tz="UTC"),
    catchup=False,
    tags=["signallake", "phase6", "benchmark"],
) as dag:
    full_recompute = BashOperator(
        task_id="full_recompute",
        bash_command=(
            "uv run python -m signallake.orchestration.recompute_benchmark run "
            f"--mode full --output {ARTIFACTS_DIR}/full.json"
        ),
        cwd=str(PROJECT_ROOT),
        doc_md="Rebuilds gold features from ALL bronze partitions; records work units "
        "(bronze rows touched) and wall-clock.",
    )

    incremental_recompute = BashOperator(
        task_id="incremental_recompute",
        bash_command=(
            "uv run python -m signallake.orchestration.recompute_benchmark run "
            f"--mode incremental --output {ARTIFACTS_DIR}/incremental.json"
        ),
        cwd=str(PROJECT_ROOT),
        doc_md="Rebuilds gold features from only the newest day + a simulated late-arriving "
        "slice of older days (~1/3 of partitions). Same work, same measurements as "
        "full_recompute, over a smaller partition set.",
    )

    report_reduction = BashOperator(
        task_id="report_reduction",
        bash_command=(
            "uv run python -m signallake.orchestration.recompute_benchmark report "
            f"--full {ARTIFACTS_DIR}/full.json --incremental {ARTIFACTS_DIR}/incremental.json"
        ),
        cwd=str(PROJECT_ROOT),
        doc_md="Prints full vs incremental work units + wall-clock + the real reduction %, "
        "and appends it to reports/METRICS.md.",
    )

    full_recompute >> incremental_recompute >> report_reduction
