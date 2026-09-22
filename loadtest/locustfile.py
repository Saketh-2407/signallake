"""Phase 5 load test: POST /score against a running `make serve`, random existing customer_ids.

Interactive (web UI):    uv run locust -f loadtest/locustfile.py --host http://localhost:8000
Headless, with a written METRICS.md summary: `make loadtest` (see `signallake.serve.run_loadtest`)

Customer ids are read straight from Redis (the same `feat:*` keys `/score` itself reads), so
this never drifts from whatever `make load-online` actually loaded.
"""

import random

import redis
from locust import HttpUser, between, events, task

from signallake.common.config import get_settings
from signallake.serve.load_online_store import KEY_PREFIX

_customer_ids: list[str] = []


@events.test_start.add_listener
def _load_customer_ids(environment, **kwargs) -> None:
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    keys = client.keys(f"{KEY_PREFIX}*")
    if not keys:
        raise RuntimeError("no online features in Redis -- run `make load-online` first")
    _customer_ids.extend(k[len(KEY_PREFIX) :] for k in keys)


class ScoringUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def score(self) -> None:
        customer_id = random.choice(_customer_ids)
        amount = round(random.uniform(1.0, 500.0), 2)
        self.client.post("/score", json={"customer_id": customer_id, "amount": amount})
