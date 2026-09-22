"""Phase 5, part 3: headless Locust load test against a running `make serve`.

    uv run python -m signallake.serve.run_loadtest

Runs `loadtest/locustfile.py` headless for a bounded duration, reports p50/p95/p99 latency
and RPS with the run config that produced them (never a hardcoded number -- see
BUILD_PLAN §1), and writes that same summary into `reports/METRICS.md`.
"""

import argparse
import concurrent.futures as cf
import csv
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import redis
import requests

from signallake.common.config import PROJECT_ROOT, get_settings
from signallake.serve.load_online_store import KEY_PREFIX

LOADTEST_FILE = PROJECT_ROOT / "loadtest" / "locustfile.py"
SCRATCH_DIR = PROJECT_ROOT / "reports" / "_loadtest_artifacts"
METRICS_PATH = PROJECT_ROOT / "reports" / "METRICS.md"
P95_TARGET_MS = 120
WARMUP_REQUESTS = 200
WARMUP_CONCURRENCY = 16


def check_server_healthy(host: str) -> None:
    try:
        resp = requests.get(f"{host}/health", timeout=3)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(
            f"cannot reach {host}/health ({e}). Start the service first: `make serve` "
            f"(and `make load-online` if you haven't loaded Redis yet)."
        ) from e


def warm_up(host: str) -> None:
    """Send a burst of real /score requests before the timed run.

    Without this, the first hit to each uvicorn worker process pays one-time costs (numpy/BLAS
    thread-pool init, sklearn JIT-ish caching) that have nothing to do with steady-state latency
    but land inside the timed window and badly skew p95/p99 -- measured ~10x on this machine.
    Every worker needs to be hit at least once, so this uses real concurrency, not a single loop.
    """
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    keys = client.keys(f"{KEY_PREFIX}*")
    if not keys:
        raise SystemExit("no online features in Redis -- run `make load-online` first")
    customer_ids = [k[len(KEY_PREFIX) :] for k in keys]

    def _one(i: int) -> None:
        customer_id = customer_ids[i % len(customer_ids)]
        requests.post(
            f"{host}/score", json={"customer_id": customer_id, "amount": 42.5}, timeout=10
        )

    print(f"warming up ({WARMUP_REQUESTS} requests, concurrency={WARMUP_CONCURRENCY})...")
    with cf.ThreadPoolExecutor(max_workers=WARMUP_CONCURRENCY) as ex:
        list(ex.map(_one, range(WARMUP_REQUESTS)))


def run_locust(host: str, users: int, spawn_rate: float, run_time: str) -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_prefix = SCRATCH_DIR / "run"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOADTEST_FILE),
        "--headless",
        "--host",
        host,
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        run_time,
        "--csv",
        str(csv_prefix),
        "--only-summary",
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return Path(f"{csv_prefix}_stats.csv")


def parse_aggregated_stats(stats_csv: Path) -> dict[str, float]:
    with stats_csv.open() as f:
        rows = list(csv.DictReader(f))
    agg = next(r for r in rows if r["Name"] == "Aggregated")
    return {
        "request_count": int(agg["Request Count"]),
        "failure_count": int(agg["Failure Count"]),
        "rps": float(agg["Requests/s"]),
        "p50": float(agg["50%"]),
        "p95": float(agg["95%"]),
        "p99": float(agg["99%"]),
    }


def write_metrics_section(stats: dict[str, float], args: argparse.Namespace) -> None:
    machine = f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores"
    failure_pct = (
        100 * stats["failure_count"] / stats["request_count"] if stats["request_count"] else 0
    )
    verdict = "PASS" if stats["p95"] < P95_TARGET_MS else "FAIL"
    section = f"""## Phase 5: online serving latency (measured, in load test)

- Run config: {args.users} users, spawn rate {args.spawn_rate}/s, `-t {args.run_time}`, \
target `{args.host}`, machine: {machine}
- Requests: {stats["request_count"]:,} total, {stats["failure_count"]:,} failed \
({failure_pct:.2f}%)
- Throughput: {stats["rps"]:.1f} req/s
- Latency: p50 = {stats["p50"]:.1f} ms, p95 = {stats["p95"]:.1f} ms, p99 = {stats["p99"]:.1f} ms
- p95 target: < {P95_TARGET_MS} ms locally, in load tests (BUILD_PLAN §2) -- **{verdict}**
- Note: Kafka, Redis, MLflow and the load generator itself all ran on this same {machine} box \
during the test, alongside the IDE/dev session -- p95 is a real number for a shared local dev \
machine, not a dedicated benchmarking one.
"""
    existing = (
        METRICS_PATH.read_text()
        if METRICS_PATH.exists()
        else "# SignalLake -- Measured Metrics\n\n"
    )
    pattern = re.compile(r"## Phase 5:.*?(?=\n## |\Z)", re.DOTALL)
    new_content = (
        pattern.sub(section, existing)
        if pattern.search(existing)
        else existing.rstrip() + "\n\n" + section
    )
    METRICS_PATH.write_text(new_content)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless Locust load test of POST /score, writes reports/METRICS.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--users", type=int, default=80)
    p.add_argument("--spawn-rate", type=float, default=20.0)
    p.add_argument("--run-time", default="120s")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    check_server_healthy(args.host)
    warm_up(args.host)

    t0 = time.perf_counter()
    stats_csv = run_locust(args.host, args.users, args.spawn_rate, args.run_time)
    stats = parse_aggregated_stats(stats_csv)

    print(f"\n=== Load test results ({time.perf_counter() - t0:.1f}s wall-clock) ===")
    print(f"run config: {args.users} users, spawn_rate={args.spawn_rate}/s, -t {args.run_time}")
    print(f"requests: {stats['request_count']:,} total, {stats['failure_count']:,} failed")
    print(f"throughput: {stats['rps']:.1f} req/s")
    print(f"latency: p50={stats['p50']:.1f}ms p95={stats['p95']:.1f}ms p99={stats['p99']:.1f}ms")
    verdict = "PASS" if stats["p95"] < P95_TARGET_MS else "FAIL"
    print(f"p95 < {P95_TARGET_MS}ms target: {verdict}")

    write_metrics_section(stats, args)
    print(f"\nwrote {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
