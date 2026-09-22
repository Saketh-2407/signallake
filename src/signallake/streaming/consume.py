"""Spark Structured Streaming: Kafka topic `events` -> bronze parquet, append, checkpointed.

    # Bounded demo: drain whatever is on the topic right now, then exit (safe to re-run).
    uv run python -m signallake.streaming.consume --once --output-dir data/bronze_stream_demo

    # Continuous: keep polling Kafka until Ctrl-C.
    uv run python -m signallake.streaming.consume --output-dir data/bronze

Requires network access on first run: PySpark fetches the Kafka source connector
(`spark-sql-kafka-0-10`, matched to the installed PySpark version) from Maven Central via
`spark.jars.packages` and caches it under `~/.ivy2`.

The output schema matches `signallake.generate.producer.EVENT_SCHEMA` exactly, so this can
append into the same `data/bronze` tree the generator writes -- but `ParquetSink` in producer.py
deletes existing `event_date=*` partitions on every run, so re-running the generator after this
would discard whatever this job landed. Point `--output-dir` elsewhere for a repeatable demo.
"""

import argparse
import time
from pathlib import Path

import pyspark
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from signallake.common.config import get_settings
from signallake.common.logging import configure_logging, get_logger
from signallake.common.schemas import EVENT_FIELDS

log = get_logger(__name__)

# Mirrors the JSON payload KafkaSink writes in producer.py: `timestamp` is an ISO-8601 string
# ("2026-08-01T09:44:35.047Z"), parsed to a real timestamp below rather than trusted to from_json.
_KAFKA_VALUE_SCHEMA = StructType(
    [
        StructField("event_id", StringType()),
        StructField("customer_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("amount", DoubleType()),
        StructField("currency", StringType()),
        StructField("merchant_category", StringType()),
        StructField("location", StringType()),
        StructField("device_id", StringType()),
        StructField("is_new_device", BooleanType()),
        StructField("status", StringType()),
        StructField("label", IntegerType()),
        StructField("anomaly_type", StringType()),
    ]
)
assert [f.name for f in _KAFKA_VALUE_SCHEMA] == EVENT_FIELDS


def build_spark(app_name: str = "signallake-consume") -> SparkSession:
    kafka_connector = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{pyspark.__version__}"
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")  # keep event-time fields un-shifted
        .config("spark.jars.packages", kafka_connector)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def read_events(spark: SparkSession, broker: str, topic: str, starting_offsets: str) -> DataFrame:
    """Kafka topic -> a DataFrame with the canonical Event schema (see `common.schemas.Event`)."""
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", broker)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )
    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), _KAFKA_VALUE_SCHEMA).alias("e")
    ).select("e.*")
    # ISO-8601 with a literal "Z" offset; Java's DateTimeFormatter reads "Z" via the offset ("X")
    # pattern, so this covers both ms-precision and whole-second timestamps.
    return parsed.withColumn(
        "timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd'T'HH:mm:ss[.SSS]X")
    ).withColumn("event_date", F.to_date("timestamp"))


def start_query(
    events: DataFrame, output_dir: Path, checkpoint_dir: Path, once: bool, trigger_seconds: int
) -> StreamingQuery:
    writer = (
        events.writeStream.format("parquet")
        .option("path", str(output_dir))
        .option("checkpointLocation", str(checkpoint_dir))
        .partitionBy("event_date")
        .outputMode("append")
    )
    writer = (
        writer.trigger(availableNow=True)
        if once
        else writer.trigger(processingTime=f"{trigger_seconds} seconds")
    )
    return writer.start()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(
        description="Stream Kafka `events` into bronze parquet, partitioned by event_date.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--broker", default=settings.kafka_broker)
    p.add_argument("--topic", default=settings.kafka_topic)
    p.add_argument("--output-dir", type=Path, default=settings.bronze_dir)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=settings.data_dir / "_checkpoints" / "streaming_bronze",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="drain what's currently on the topic, then exit (Trigger.AvailableNow)",
    )
    p.add_argument(
        "--trigger-seconds", type=int, default=10, help="micro-batch interval when not --once"
    )
    p.add_argument("--starting-offsets", choices=["earliest", "latest"], default="earliest")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        events = read_events(spark, args.broker, args.topic, args.starting_offsets)
        log.info(
            "streaming_query_starting",
            topic=args.topic,
            output_dir=str(args.output_dir),
            mode="once" if args.once else f"continuous ({args.trigger_seconds}s)",
        )
        t0 = time.perf_counter()
        query = start_query(
            events, args.output_dir, args.checkpoint_dir, args.once, args.trigger_seconds
        )
        try:
            query.awaitTermination()
        except Exception as e:
            # A topic that doesn't exist YET (nobody has produced to it) is a real, expected
            # state for a fresh/cold pipeline -- `make generate` alone never touches Kafka, only
            # `--to-kafka` does. For the bounded demo, that's "0 new events", not a failure: a
            # scheduled ingestion task running before the topic's first producer shouldn't crash
            # the DAG. Continuous mode still raises -- a long-running consumer losing its topic
            # is a real problem worth surfacing loudly.
            if args.once and "UnknownTopicOrPartitionException" in str(e):
                elapsed = time.perf_counter() - t0
                print(
                    f"topic '{args.topic}' does not exist yet (nothing has produced to it) -- "
                    f"0 events ingested  [{elapsed:.1f}s]"
                )
                return
            raise
        elapsed = time.perf_counter() - t0
        progress = query.lastProgress or {}
        rows = int(progress.get("numInputRows", 0))
        dest = args.output_dir
        print(f"query stopped after {elapsed:.1f}s; last batch landed {rows:,} rows -> {dest}")
        if args.once:
            print("mode: --once (Trigger.AvailableNow) -- exited after draining the topic")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
