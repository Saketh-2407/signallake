"""Synthetic event producer: simulate, then write bronze parquet and/or publish to Kafka.

    uv run python -m signallake.generate.producer --num-events 2100000 --to-parquet
    uv run python -m signallake.generate.producer --num-events 50000 --to-kafka --no-parquet

Events are written day by day in timestamp order, so Kafka sees them in event-time order and
bronze gets one `event_date=YYYY-MM-DD/` partition per simulated day.
"""

import argparse
import json
import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from confluent_kafka import Producer
from tqdm import tqdm

from signallake.common.config import get_settings
from signallake.common.logging import configure_logging, get_logger
from signallake.common.schemas import EVENT_FIELDS
from signallake.generate.simulate import (
    ANOMALY_NAMES,
    REFERENCE_NOISE,
    EventTable,
    GeneratorConfig,
    generate_events,
)

log = get_logger(__name__)

PARTITION_PREFIX = "event_date="
MANIFEST = "_generation.json"  # leading underscore: Spark/pyarrow skip it when reading bronze


# Explicit so every partition has an identical schema (e.g. a day with no anomalies must still
# have a string `anomaly_type`). Timestamps are UTC/millisecond: Spark 3.5 rejects nanoseconds and
# would read tz-naive timestamps as TIMESTAMP_NTZ.
EVENT_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("customer_id", pa.string()),
        ("timestamp", pa.timestamp("ms", "UTC")),
        ("amount", pa.float64()),
        ("currency", pa.string()),
        ("merchant_category", pa.string()),
        ("location", pa.string()),
        ("device_id", pa.string()),
        ("is_new_device", pa.bool_()),
        ("status", pa.string()),
        ("label", pa.int32()),
        ("anomaly_type", pa.string()),
    ]
)
assert EVENT_SCHEMA.names == EVENT_FIELDS, "EVENT_SCHEMA drifted from common.schemas.Event"


def to_arrow(columns: dict[str, np.ndarray]) -> pa.Table:
    """Canonical Event columns (as returned by `EventTable.decode`) -> Arrow table."""
    arrays = []
    for field in EVENT_SCHEMA:
        if field.name == "timestamp":
            arrays.append(pa.array(columns["timestamp_ms"], pa.int64()).cast(field.type))
        else:
            arrays.append(pa.array(columns[field.name], field.type))
    return pa.Table.from_arrays(arrays, schema=EVENT_SCHEMA)


class ParquetSink:
    """Writes one file per event_date under `root/event_date=YYYY-MM-DD/`."""

    def __init__(self, root: Path, start: date) -> None:
        self.root, self.start = root, start
        self.root.mkdir(parents=True, exist_ok=True)
        stale = [p for p in self.root.glob(f"{PARTITION_PREFIX}*") if p.is_dir()]
        for partition in stale:
            shutil.rmtree(partition)
        if stale:
            log.info("replaced_existing_partitions", root=str(self.root), partitions=len(stale))

    def write_day(self, day: int, table: pa.Table) -> None:
        partition = self.root / f"{PARTITION_PREFIX}{self.start + timedelta(days=day)}"
        partition.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, partition / "part-00000.parquet", compression="snappy")


class KafkaSink:
    """Batched JSON producer keyed by customer_id (keeps a customer's events ordered)."""

    def __init__(self, broker: str, topic: str) -> None:
        self.topic = topic
        self.delivered = self.failed = 0
        self.producer = Producer(
            {
                "bootstrap.servers": broker,
                "linger.ms": 50,
                "batch.num.messages": 20_000,
                # No compression: the broker is on localhost, and it keeps every consumer
                # (kafka-python included) able to read the topic without extra codec libraries.
                "queue.buffering.max.messages": 500_000,
            }
        )

    def _on_delivery(self, err, _msg) -> None:
        if err is None:
            self.delivered += 1
        else:
            self.failed += 1

    def write(self, table: pa.Table) -> None:
        # pandas renders tz-aware timestamps as ISO-8601 ("2026-08-01T00:00:00.000Z").
        payloads = table.to_pandas().to_json(
            orient="records", lines=True, date_format="iso", date_unit="ms"
        )
        keys = table.column("customer_id").to_pylist()
        for key, payload in zip(keys, payloads.splitlines(), strict=True):
            while True:
                try:
                    self.producer.produce(
                        self.topic, payload, key=key, on_delivery=self._on_delivery
                    )
                    break
                except BufferError:  # local queue full: let librdkafka drain it
                    self.producer.poll(0.2)
            self.producer.poll(0)

    def close(self) -> None:
        remaining = self.producer.flush(120)
        if remaining or self.failed:
            raise RuntimeError(f"kafka: {remaining} undelivered, {self.failed} failed")


def write_all(
    table: EventTable, parquet: ParquetSink | None, kafka: KafkaSink | None
) -> dict[str, float]:
    """Stream the table to the sinks one event_date at a time; returns seconds spent per sink."""
    spent = {"parquet": 0.0, "kafka": 0.0}
    bounds = table.day_bounds()
    with tqdm(total=len(table), unit="ev", unit_scale=True, desc="writing") as bar:
        for day in range(len(bounds) - 1):
            lo, hi = int(bounds[day]), int(bounds[day + 1])
            if hi == lo:
                continue
            arrow = to_arrow(table.decode(lo, hi))
            if parquet:
                t = time.perf_counter()
                parquet.write_day(day, arrow)
                spent["parquet"] += time.perf_counter() - t
            if kafka:
                t = time.perf_counter()
                kafka.write(arrow)
                spent["kafka"] += time.perf_counter() - t
            bar.update(hi - lo)
    if kafka:
        t = time.perf_counter()
        kafka.close()
        spent["kafka"] += time.perf_counter() - t
    return spent


def verify_bronze(root: Path, expected_rows: int) -> tuple[int, int, str, str]:
    """Re-open bronze as a hive-partitioned dataset: (rows, partitions, first date, last date)."""
    dataset = ds.dataset(root, format="parquet", partitioning="hive", exclude_invalid_files=True)
    rows = dataset.count_rows()
    dates = sorted(p.name.removeprefix(PARTITION_PREFIX) for p in root.glob(f"{PARTITION_PREFIX}*"))
    if rows != expected_rows:
        raise RuntimeError(f"bronze has {rows:,} rows, expected {expected_rows:,}")
    return rows, len(dates), dates[0], dates[-1]


def _pct(x: float) -> str:
    return f"{100 * x:5.2f}%"


def print_report(table: EventTable) -> None:
    """Label balance and normal-vs-anomalous distributions, computed from the in-memory table."""
    label, amount = table.label, table.amount
    normal, anom = label == 0, label == 1
    n = len(table)
    print("\n=== Label balance ===")
    print(f"events        {n:>12,}")
    print(f"anomalies     {int(anom.sum()):>12,}  ({_pct(anom.mean())})")
    print("by injected type (after label noise; 'noised' = flipped labels, no type):")
    for code, name in enumerate(ANOMALY_NAMES):
        if code:
            print(f"  {name:<22}{int((table.anomaly == code).sum()):>10,}")
    print(f"  {'noised (no type)':<22}{int((anom & (table.anomaly == 0)).sum()):>10,}")

    print("\n=== Amount: normal vs anomalous (USD) ===")
    print(f"{'':<10}{'mean':>10}{'median':>10}{'p90':>10}{'p99':>10}")
    for name, mask in (("normal", normal), ("anomaly", anom)):
        a = amount[mask]
        p90, p99 = np.percentile(a, [90, 99])
        print(f"{name:<10}{a.mean():>10.1f}{np.median(a):>10.1f}{p90:>10.1f}{p99:>10.1f}")
    n_p90 = np.percentile(amount[normal], 90)
    overlap = (amount[anom] <= n_p90).mean()
    print(f"anomalies at or below the normal p90 amount: {_pct(overlap)}  (overlap)")
    print(
        "anomaly median / normal median: "
        f"{np.median(amount[anom]) / np.median(amount[normal]):.2f}x  (skew)"
    )

    print("\n=== Other signals ===")
    print(f"{'':<10}{'failed':>10}{'new_device':>12}{'night 0-5h':>12}")
    hour = (table.timestamp_ms // 3_600_000) % 24
    for name, mask in (("normal", normal), ("anomaly", anom)):
        print(
            f"{name:<10}{_pct(table.failed[mask].mean()):>10}"
            f"{_pct(table.is_new_device[mask].mean()):>12}{_pct((hour[mask] < 6).mean()):>12}"
        )
    per_customer = np.bincount(table.customer, minlength=table.config.num_customers)
    print(
        f"\ncustomers: {table.config.num_customers:,}; events/customer "
        f"min {per_customer.min():,} / median {int(np.median(per_customer)):,} "
        f"/ max {per_customer.max():,}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    d = GeneratorConfig()
    p = argparse.ArgumentParser(
        description="Generate synthetic transaction events with injected anomalies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-events", type=int, default=d.num_events)
    p.add_argument("--num-customers", type=int, default=d.num_customers)
    p.add_argument("--anomaly-rate", type=float, default=d.anomaly_rate)
    p.add_argument(
        "--signal-strength",
        type=float,
        default=d.signal_strength,
        help="scales burst size / amount multiplier / geo-jump odds; higher = easier to detect",
    )
    p.add_argument(
        "--noise",
        type=float,
        default=d.noise,
        help=f"amount spread and hard-negative rate (calibrated at {REFERENCE_NOISE}); "
        "higher = more normal/anomalous overlap",
    )
    p.add_argument("--subtle-fraction", type=float, default=d.subtle_fraction)
    p.add_argument("--label-noise", type=float, default=d.label_noise)
    p.add_argument("--num-days", type=int, default=d.num_days)
    p.add_argument("--start-date", type=date.fromisoformat, default=d.start_date)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--to-kafka", dest="to_kafka", action="store_true", help="publish to Kafka")
    p.add_argument("--no-kafka", dest="to_kafka", action="store_false", help="(default)")
    p.add_argument("--to-parquet", dest="to_parquet", action="store_true", help="(default)")
    p.add_argument("--no-parquet", dest="to_parquet", action="store_false")
    p.set_defaults(to_kafka=False, to_parquet=True)
    p.add_argument("--kafka-broker", default=settings.kafka_broker)
    p.add_argument("--topic", default=settings.kafka_topic)
    p.add_argument("--output-dir", type=Path, default=settings.bronze_dir)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    if not (args.to_kafka or args.to_parquet):
        raise SystemExit("nothing to do: pass --to-kafka and/or --to-parquet")

    cfg = GeneratorConfig(
        num_events=args.num_events,
        num_customers=args.num_customers,
        anomaly_rate=args.anomaly_rate,
        signal_strength=args.signal_strength,
        noise=args.noise,
        subtle_fraction=args.subtle_fraction,
        label_noise=args.label_noise,
        num_days=args.num_days,
        start_date=args.start_date,
        seed=args.seed,
    )
    t0 = time.perf_counter()
    table = generate_events(cfg)
    t_sim = time.perf_counter() - t0
    print(f"simulated {len(table):,} events in {t_sim:.1f}s")

    parquet = ParquetSink(args.output_dir, cfg.start_date) if args.to_parquet else None
    kafka = KafkaSink(args.kafka_broker, args.topic) if args.to_kafka else None
    spent = write_all(table, parquet, kafka)
    print_report(table)

    print("\n=== Output ===")
    if parquet:
        rows, parts, first, last = verify_bronze(args.output_dir, len(table))
        print(f"bronze parquet: {rows:,} rows in {parts} event_date partitions ({first}..{last})")
        print(f"                {args.output_dir}  [write {spent['parquet']:.1f}s]")
        manifest = {
            "config": {**cfg.__dict__, "start_date": str(cfg.start_date)},
            "rows": rows,
            "anomalies": int(table.label.sum()),
            "first_event_date": first,
            "last_event_date": last,
        }
        (args.output_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    if kafka:
        rate = kafka.delivered / spent["kafka"] if spent["kafka"] else 0
        print(
            f"kafka: {kafka.delivered:,} messages delivered to '{args.topic}' "
            f"@ {args.kafka_broker} [{spent['kafka']:.1f}s, {rate:,.0f} msg/s]"
        )
    print(f"total wall-clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
