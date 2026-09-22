# SignalLake -- Measured Metrics

Every number below is what a real run of this pipeline actually produced, not a target hit by
adjusting the number itself -- see `SignalLake_BUILD_PLAN.md` §1. Where a number depends on how
it was measured (a load test's concurrency, a benchmark's scenario), that's stated alongside it.

## Summary

| Metric | Value | Qualifier |
|---|---|---|
| Events processed | 2,100,000 | **synthetic data** -- Phase 1 generator, 30 days, 2,000 customers, 3% injected anomaly rate |
| Reusable features | 34 | Spark windowed aggregates over customer history, 4 trailing windows (5m/10m/1h/24h) |
| Anomaly detection F1 | 0.880 (precision 0.915, recall 0.848) | supervised HistGradientBoostingClassifier, all 34 features, measured on a **held-out TEST split of synthetic data** (time-based split, no future event in training) -- an F1 score, not an accuracy percentage |
| Online scoring p95 latency | 240 ms | **in load tests only** -- 80 concurrent users, 120s, shared local dev machine (not a per-request guarantee, and not a dedicated benchmarking box -- see Phase 5 detail below) |
| Recompute reduction | 63.1% fewer work units | **vs. a full-recompute baseline** -- incremental (watermark-based) rebuild touching 11 of 30 day-partitions, on synthetic data with a designed late-arrival scenario (see Phase 6 detail below) |

## Phase 5: online serving latency (measured, in load test)

- Run config: 80 users, spawn rate 20.0/s, `-t 120s`, target `http://localhost:8000`, machine: Linux x86_64, 8 cores
- Requests: 50,223 total, 0 failed (0.00%)
- Throughput: 422.2 req/s
- Latency: p50 = 130.0 ms, p95 = 240.0 ms, p99 = 300.0 ms
- p95 target: < 120 ms locally, in load tests (BUILD_PLAN §2) -- **FAIL**
- Note: Kafka, Redis, MLflow and the load generator itself all ran on this same Linux x86_64, 8 cores box during the test, alongside the IDE/dev session -- p95 is a real number for a shared local dev machine, not a dedicated benchmarking one.

## Phase 6: incremental vs. full recompute (measured)

- Work unit definition: number of bronze (raw ingested) rows in the event_date partitions that get rebuilt.
- FULL: 30/30 partitions, 2,100,000 work units, 78.1s wall-clock (includes this task's own Spark startup).
- INCREMENTAL: 11/30 partitions, 773,946 work units, 23.5s wall-clock (includes this task's own Spark startup).
- Reduction: 63.1% fewer work units, 69.9% less wall-clock, vs full recompute (synthetic data; scenario designed per BUILD_PLAN §7 to land near ~64% -- see the module docstring in `signallake.orchestration.recompute_benchmark` for exactly how the incremental partition set was chosen).
