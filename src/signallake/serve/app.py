"""Phase 5, part 2: online feature serving + scoring.

    uv run uvicorn signallake.serve.app:app --host 0.0.0.0 --port 8000

At startup, loads the latest registered `signallake-anomaly-detector` model + its chosen
decision threshold from MLflow, once, into memory. Every request after that is pure
in-memory work: one Redis GET (the customer's last-batch feature vector, written by
`make load-online`) plus one `pipeline.predict_proba` call -- no disk I/O, no MLflow calls,
on the hot path.

`POST /score` blends that stored vector with the CURRENT event: `amount`, the temporal
fields (`hour_of_day`, `is_night`, `is_weekend`) and `new_device_flag` are refreshed from
the request, since those are true "as of right now". The remaining ~30 windowed/derived
features (rolling counts, means, z-scores, ...) are served as of the last `make load-online`
run, not recomputed live -- doing that would mean re-running the Spark feature pipeline
per request, which defeats the point of an online store. That staleness window is exactly
what `make load-online` (typically run on a schedule in Airflow, Phase 6) exists to bound.
"""

import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import numpy as np
import redis
from fastapi import FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from signallake.common.config import get_settings
from signallake.common.logging import configure_logging, get_logger
from signallake.common.mlflow_names import REGISTERED_MODEL_NAME
from signallake.features.columns import features_for_tier
from signallake.serve.load_online_store import KEY_PREFIX

log = get_logger(__name__)


class FeaturesResponse(BaseModel):
    customer_id: str
    features: dict[str, float]


class ScoreRequest(BaseModel):
    customer_id: str
    amount: float = Field(ge=0)
    timestamp: datetime | None = None
    is_new_device: bool = False


class ScoreResponse(BaseModel):
    customer_id: str
    score: float
    is_anomaly: bool
    threshold: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    feature_set: str
    redis: str


def _load_latest_model(client: MlflowClient) -> tuple[object, float, list[str], str]:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"no registered versions of '{REGISTERED_MODEL_NAME}' -- run `make train` first"
        )
    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"
    # skops_trusted_types was set at log time (Phase 4) and is persisted in the model's flavor
    # config, so load_model re-applies it automatically -- nothing to pass here.
    pipeline = mlflow.sklearn.load_model(model_uri)
    threshold = float(latest.tags["decision_threshold"])
    feature_order = features_for_tier(latest.tags["feature_set"])
    return pipeline, threshold, feature_order, latest.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model, threshold, feature_order, version = _load_latest_model(MlflowClient())
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()

    app.state.model = model
    app.state.threshold = threshold
    app.state.feature_order = feature_order
    app.state.model_version = version
    app.state.redis = redis_client
    log.info(
        "serve.startup",
        model_version=version,
        n_features=len(feature_order),
        threshold=threshold,
    )
    yield
    redis_client.close()


app = FastAPI(title="SignalLake Online Scoring", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    try:
        state.redis.ping()
        redis_status = "ok"
    except redis.RedisError:
        redis_status = "error"
    return HealthResponse(
        status="ok" if redis_status == "ok" else "degraded",
        model_name=REGISTERED_MODEL_NAME,
        model_version=str(state.model_version),
        feature_set=f"{len(state.feature_order)} features",
        redis=redis_status,
    )


@app.get("/features/{customer_id}", response_model=FeaturesResponse)
def get_features(customer_id: str, request: Request) -> FeaturesResponse:
    raw = request.app.state.redis.get(f"{KEY_PREFIX}{customer_id}")
    if raw is None:
        raise HTTPException(
            404, f"no online features for customer_id={customer_id!r} -- run `make load-online`"
        )
    return FeaturesResponse(customer_id=customer_id, features=json.loads(raw))


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    t0 = time.perf_counter()
    state = request.app.state

    raw = state.redis.get(f"{KEY_PREFIX}{req.customer_id}")
    if raw is None:
        raise HTTPException(
            404,
            f"no online features for customer_id={req.customer_id!r} -- run `make load-online`",
        )
    feats: dict[str, float] = json.loads(raw)

    ts = req.timestamp or datetime.now(UTC)
    feats["amount"] = req.amount
    feats["hour_of_day"] = float(ts.hour)
    feats["is_night"] = 1.0 if ts.hour < 6 else 0.0
    feats["is_weekend"] = 1.0 if ts.weekday() >= 5 else 0.0
    feats["new_device_flag"] = float(req.is_new_device)

    # A raw ndarray (not a DataFrame) skips pandas' per-call construction/indexing overhead --
    # the pipeline only needs column ORDER at inference time, which state.feature_order fixes.
    x = np.array([[feats[name] for name in state.feature_order]], dtype=np.float64)
    proba = float(state.model.predict_proba(x)[0, 1])

    return ScoreResponse(
        customer_id=req.customer_id,
        score=proba,
        is_anomaly=proba >= state.threshold,
        threshold=state.threshold,
        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
    )
