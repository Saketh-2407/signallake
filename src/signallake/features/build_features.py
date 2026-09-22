"""Phase 2, Part B: bronze -> silver (cleaned, deduped) -> gold (silver + 34 features + label).

    uv run python -m signallake.features.build_features

Reads `data/bronze`, writes `data/silver` and `data/gold`, both partitioned by `event_date`.
"""

import argparse
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from signallake.common.config import get_settings
from signallake.common.logging import configure_logging, get_logger
from signallake.common.schemas import EVENT_FIELDS
from signallake.features.columns import FEATURE_NAMES, build_gold_features

log = get_logger(__name__)


def build_spark(driver_memory: str, shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.appName("signallake-build-features")
        .master("local[*]")
        .config(
            "spark.sql.session.timeZone", "UTC"
        )  # hour_of_day / is_night / is_weekend need this
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def build_silver(bronze: DataFrame) -> DataFrame:
    """Clean + type + dedupe bronze: drop rows missing required fields, one row per event_id."""
    required = ["event_id", "customer_id", "timestamp", "amount", "status", "label"]
    cleaned = bronze
    for col in required:
        cleaned = cleaned.filter(F.col(col).isNotNull())
    cleaned = cleaned.filter(F.col("amount") >= 0)
    # Re-derive event_date from the canonical timestamp (don't just trust the partition column
    # bronze was read from -- that's the point of a "cleaned, typed" layer).
    cleaned = cleaned.withColumn("event_date", F.to_date("timestamp"))
    return cleaned.dropDuplicates(["event_id"])


def build_gold(silver: DataFrame) -> DataFrame:
    """silver + all 34 features, feature columns cast to double so training sees one dtype."""
    gold = build_gold_features(silver)
    for name in FEATURE_NAMES:
        gold = gold.withColumn(name, F.col(name).cast("double"))
    # "amount" is both a raw event field and one of the 34 catalog features (same column) --
    # keep it once, via EVENT_FIELDS, rather than selecting it twice.
    extra_features = [name for name in FEATURE_NAMES if name not in EVENT_FIELDS]
    return gold.drop("ts_epoch").select(*EVENT_FIELDS, "event_date", *extra_features)


def assert_no_nulls(gold: DataFrame, columns: list[str]) -> None:
    """Fail loudly (naming the offending columns) rather than silently shipping null features."""
    counts = gold.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns]).first()
    bad = {c: n for c, n in zip(columns, counts, strict=True) if n}
    if bad:
        raise RuntimeError(f"null values in gold feature columns: {bad}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(
        description="Build the silver and gold (34-feature) layers from bronze.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bronze-dir", type=Path, default=settings.bronze_dir)
    p.add_argument("--silver-dir", type=Path, default=settings.silver_dir)
    p.add_argument("--gold-dir", type=Path, default=settings.gold_dir)
    p.add_argument("--driver-memory", default="3g")
    p.add_argument("--shuffle-partitions", type=int, default=8)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    spark = build_spark(args.driver_memory, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    t0 = time.perf_counter()
    try:
        bronze = spark.read.parquet(str(args.bronze_dir))
        bronze_rows = bronze.count()
        print(f"bronze: {bronze_rows:,} rows  [{time.perf_counter() - t0:.1f}s]")

        t = time.perf_counter()
        silver = build_silver(bronze).cache()
        silver_rows = silver.count()  # materializes the cache; every later use reads it back
        dropped = bronze_rows - silver_rows
        print(
            f"silver: {silver_rows:,} rows ({dropped:,} dropped)  [{time.perf_counter() - t:.1f}s]"
        )
        silver.write.mode("overwrite").partitionBy("event_date").parquet(str(args.silver_dir))

        t = time.perf_counter()
        gold = build_gold(silver).cache()
        gold_rows = gold.count()
        assert_no_nulls(gold, FEATURE_NAMES)
        gold.write.mode("overwrite").partitionBy("event_date").parquet(str(args.gold_dir))
        gold_elapsed = time.perf_counter() - t
        n_feat = len(FEATURE_NAMES)
        print(
            f"gold:   {gold_rows:,} rows, {n_feat} feature columns, no nulls  [{gold_elapsed:.1f}s]"
        )

        print("\n=== Gold schema ===")
        gold.select(*FEATURE_NAMES, "label").printSchema()
        print(f"\ngold build wall-clock (34 features, null-checked, written): {gold_elapsed:.1f}s")
        print(f"total wall-clock (bronze read + silver + gold): {time.perf_counter() - t0:.1f}s")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
