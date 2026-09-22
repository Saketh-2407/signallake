"""Phase 5, part 1: push the latest per-customer feature vector from `data/gold` into Redis.

    uv run python -m signallake.serve.load_online_store

Each customer's most recent gold row (by `timestamp`) is written to key `feat:{customer_id}`
as a JSON object of the 34 feature values -- the "online store" `/score` reads from at request
time (see `signallake.serve.app`).
"""

import argparse
import json
import time
from pathlib import Path

import duckdb
import redis

from signallake.common.config import get_settings
from signallake.common.logging import configure_logging, get_logger
from signallake.features.columns import FEATURE_NAMES

log = get_logger(__name__)

KEY_PREFIX = "feat:"
PIPELINE_BATCH_SIZE = 2000


def latest_per_customer(gold_dir: Path) -> list[dict]:
    con = duckdb.connect()
    cols = ", ".join(FEATURE_NAMES)
    query = f"""
        select customer_id, {cols}
        from (
            select *, row_number() over (partition by customer_id order by timestamp desc) as rn
            from read_parquet('{gold_dir}/**/*.parquet', hive_partitioning=true)
        )
        where rn = 1
    """
    return con.execute(query).df().to_dict(orient="records")


def load_into_redis(rows: list[dict], redis_url: str) -> int:
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    client.ping()
    written = 0
    pipe = client.pipeline(transaction=False)
    for i, row in enumerate(rows, start=1):
        customer_id = row.pop("customer_id")
        pipe.set(f"{KEY_PREFIX}{customer_id}", json.dumps(row))
        written += 1
        if i % PIPELINE_BATCH_SIZE == 0:
            pipe.execute()
            pipe = client.pipeline(transaction=False)
    pipe.execute()
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(
        description="Push the latest per-customer gold feature vector into Redis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gold-dir", type=Path, default=settings.gold_dir)
    p.add_argument("--redis-url", default=settings.redis_url)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()
    t0 = time.perf_counter()
    rows = latest_per_customer(args.gold_dir)
    print(f"latest-per-customer: {len(rows):,} customers  [{time.perf_counter() - t0:.1f}s]")

    n_features = len(FEATURE_NAMES)
    print(f"each value: {n_features} feature fields (JSON), key = '{KEY_PREFIX}<customer_id>'")

    t = time.perf_counter()
    written = load_into_redis(rows, args.redis_url)
    print(
        f"redis: {written:,} keys written under '{KEY_PREFIX}*'  [{time.perf_counter() - t:.1f}s]"
    )
    print(f"total wall-clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
