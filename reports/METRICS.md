# SignalLake -- Measured Metrics

## Phase 5: online serving latency (measured, in load test)

- Run config: 80 users, spawn rate 20.0/s, `-t 120s`, target `http://localhost:8000`, machine: Linux x86_64, 8 cores
- Requests: 50,223 total, 0 failed (0.00%)
- Throughput: 422.2 req/s
- Latency: p50 = 130.0 ms, p95 = 240.0 ms, p99 = 300.0 ms
- p95 target: < 120 ms locally, in load tests (BUILD_PLAN §2) -- **FAIL**
- Note: Kafka, Redis, MLflow and the load generator itself all ran on this same Linux x86_64, 8 cores box during the test, alongside the IDE/dev session -- p95 is a real number for a shared local dev machine, not a dedicated benchmarking one.
