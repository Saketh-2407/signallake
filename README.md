# SignalLake

A local, real-time data + ML platform: synthetic events → Kafka → Spark → dbt quality gates →
features → MLflow-tracked anomaly models → Redis/FastAPI online serving, orchestrated by Airflow.
All synthetic data, all local (WSL2).

> Status: **Phase 1** (synthetic generator). Full docs land in Phase 7.

## Prerequisites

- WSL2 Ubuntu, Docker (with WSL integration), Java 17 (Temurin), Python 3.11+, [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
cp .env.example .env
make install   # uv sync
make up        # Kafka (:9092), Redis (:6379), MLflow (:5000), waits until healthy
```

MLflow UI: http://localhost:5000

## Generate data (Phase 1)

```bash
make generate NUM_EVENTS=50000                  # quick smoke run -> data/bronze (parquet, by event_date)
make generate                                   # full 2.1M events
make generate GEN_ARGS="--to-kafka --no-parquet"  # publish to the Kafka topic `events` instead
```

Difficulty is tunable and never perfect: `--signal-strength` (higher = easier) and `--noise`
(higher = more normal/anomalous overlap). See `src/signallake/generate/simulate.py`.

## Layout

See `SignalLake_BUILD_PLAN.md` for the architecture and phase plan.
