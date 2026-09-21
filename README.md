# SignalLake

A local, real-time data + ML platform: synthetic events → Kafka → Spark → dbt quality gates →
features → MLflow-tracked anomaly models → Redis/FastAPI online serving, orchestrated by Airflow.
All synthetic data, all local (WSL2).

> Status: **Phase 0** (scaffold + infrastructure). Full docs land in Phase 7.

## Prerequisites

- WSL2 Ubuntu, Docker (with WSL integration), Java 17 (Temurin), Python 3.11+, [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
cp .env.example .env
make install   # uv sync
make up        # Kafka (:9092), Redis (:6379), MLflow (:5000), waits until healthy
```

MLflow UI: http://localhost:5000

## Layout

See `SignalLake_BUILD_PLAN.md` for the architecture and phase plan.
