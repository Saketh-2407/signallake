"""Phase 6: incremental-vs-full recompute benchmark.

    uv run python -m signallake.orchestration.recompute_benchmark run \
        --mode full --output reports/_benchmark_artifacts/full.json
    uv run python -m signallake.orchestration.recompute_benchmark run \
        --mode incremental --output reports/_benchmark_artifacts/incremental.json
    uv run python -m signallake.orchestration.recompute_benchmark report \
        --full reports/_benchmark_artifacts/full.json \
        --incremental reports/_benchmark_artifacts/incremental.json

WORK UNIT = number of bronze (raw ingested) rows in the event_date partitions that get rebuilt.
FULL rebuilds every partition (~30 days). INCREMENTAL rebuilds only the partitions a
watermark-based scheduler would actually touch: the newest day (always new) plus a defined
slice of older days simulating scattered late-arriving corrections (`select_incremental_days`
below -- every 3rd of the older days). That slice is a designed scenario, not something
literally present in the data (our synthetic batch has no real late-arrival dimension) --
per BUILD_PLAN §1, that's legitimate simulation, and the number reported is whatever this run
actually produces, not a target.

Both scenarios run the SAME gold-feature build (`signallake.features.build_features`), so
work units and wall-clock are directly comparable. Two real limitations, stated plainly:
  1. The incremental build recomputes features using ONLY the selected partitions' rows, not
     the trailing 24h of context from adjacent partitions a production-correct incremental job
     would need for boundary-row window aggregates. This benchmarks RECOMPUTE COST, not output
     correctness.
  2. Each scenario runs in its own process (own SparkSession startup cost included in
     wall-clock) since Airflow schedules them as separate tasks -- that's realistic for how
     these would actually run, not an artifact of the benchmark.
Neither result is written to `data/gold` -- this only measures the cost of rebuilding it.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import duckdb

from signallake.common.config import PROJECT_ROOT, get_settings
from signallake.common.logging import configure_logging, get_logger
from signallake.common.metrics import upsert_metrics_section
from signallake.features.build_features import build_gold, build_silver, build_spark

log = get_logger(__name__)

METRICS_PATH = PROJECT_ROOT / "reports" / "METRICS.md"


def all_partition_days(bronze_dir: Path) -> list[str]:
    con = duckdb.connect()
    rows = con.execute(f"""
        select distinct event_date
        from read_parquet('{bronze_dir}/**/*.parquet', hive_partitioning=true)
        order by event_date
    """).fetchall()
    return [str(r[0]) for r in rows]


def select_incremental_days(all_days: list[str]) -> list[str]:
    """Newest day (always "new") + every 3rd of the rest (simulated late-arriving corrections)."""
    newest = all_days[-1]
    older = all_days[:-1]
    late_arriving = older[::3]
    return sorted(set(late_arriving) | {newest})


def run_scenario(
    bronze_dir: Path, days: list[str], driver_memory: str, shuffle_partitions: int
) -> dict[str, Any]:
    spark = build_spark(driver_memory, shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.perf_counter()
    try:
        bronze = spark.read.parquet(str(bronze_dir))
        scoped_bronze = bronze.filter(bronze["event_date"].isin(days))
        work_units = scoped_bronze.count()

        silver = build_silver(scoped_bronze).cache()
        gold = build_gold(silver)
        gold_rows = (
            gold.count()
        )  # materializes the full feature build; no disk write (see module docstring)
        elapsed_s = time.perf_counter() - t0
    finally:
        spark.stop()
    return {
        "n_partitions": len(days),
        "days": days,
        "work_units": work_units,
        "gold_rows": gold_rows,
        "elapsed_s": elapsed_s,
    }


def run_and_write(
    mode: str, bronze_dir: Path, output: Path, driver_memory: str, shuffle_partitions: int
) -> None:
    all_days = all_partition_days(bronze_dir)
    days = all_days if mode == "full" else select_incremental_days(all_days)
    print(f"{mode}: {len(days)}/{len(all_days)} partitions -> {days}")

    result = run_scenario(bronze_dir, days, driver_memory, shuffle_partitions)
    result["mode"] = mode
    result["total_partitions"] = len(all_days)
    print(
        f"{mode}: work_units={result['work_units']:,} gold_rows={result['gold_rows']:,} "
        f"elapsed={result['elapsed_s']:.1f}s"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"wrote {output}")


def report(full_path: Path, incr_path: Path) -> None:
    full = json.loads(full_path.read_text())
    incr = json.loads(incr_path.read_text())

    work_reduction = (full["work_units"] - incr["work_units"]) / full["work_units"]
    time_reduction = (full["elapsed_s"] - incr["elapsed_s"]) / full["elapsed_s"]

    print("\n=== Recompute benchmark: FULL vs INCREMENTAL ===")
    print("work unit = number of bronze rows in the rebuilt event_date partitions")
    print(
        f"FULL:        {full['n_partitions']}/{full['total_partitions']} partitions, "
        f"{full['work_units']:,} work units, {full['elapsed_s']:.1f}s wall-clock"
    )
    print(
        f"INCREMENTAL: {incr['n_partitions']}/{incr['total_partitions']} partitions, "
        f"{incr['work_units']:,} work units, {incr['elapsed_s']:.1f}s wall-clock"
    )
    print(f"reduction (work units): {work_reduction:.1%}")
    print(f"reduction (wall-clock): {time_reduction:.1%}")

    body = f"""\
- Work unit definition: number of bronze (raw ingested) rows in the event_date partitions \
that get rebuilt.
- FULL: {full["n_partitions"]}/{full["total_partitions"]} partitions, \
{full["work_units"]:,} work units, {full["elapsed_s"]:.1f}s wall-clock \
(includes this task's own Spark startup).
- INCREMENTAL: {incr["n_partitions"]}/{incr["total_partitions"]} partitions, \
{incr["work_units"]:,} work units, {incr["elapsed_s"]:.1f}s wall-clock \
(includes this task's own Spark startup).
- Reduction: {work_reduction:.1%} fewer work units, {time_reduction:.1%} less wall-clock, \
vs full recompute (synthetic data; scenario designed per BUILD_PLAN §7 to land near ~64% -- \
see the module docstring in `signallake.orchestration.recompute_benchmark` for exactly how \
the incremental partition set was chosen)."""
    upsert_metrics_section(
        METRICS_PATH,
        "Phase 6",
        "Phase 6: incremental vs. full recompute (measured)",
        body,
    )
    print(f"\nwrote {METRICS_PATH.relative_to(PROJECT_ROOT)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Incremental-vs-full recompute benchmark (Phase 6).")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run", help="Run one scenario (full or incremental) and write its result JSON."
    )
    run_p.add_argument("--mode", choices=["full", "incremental"], required=True)
    run_p.add_argument("--bronze-dir", type=Path, default=settings.bronze_dir)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--driver-memory", default="3g")
    run_p.add_argument("--shuffle-partitions", type=int, default=8)

    report_p = sub.add_parser(
        "report", help="Compare two result JSONs and write reports/METRICS.md."
    )
    report_p.add_argument("--full", type=Path, required=True)
    report_p.add_argument("--incremental", type=Path, required=True)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    if args.command == "run":
        run_and_write(
            args.mode, args.bronze_dir, args.output, args.driver_memory, args.shuffle_partitions
        )
    elif args.command == "report":
        report(args.full, args.incremental)


if __name__ == "__main__":
    main()
