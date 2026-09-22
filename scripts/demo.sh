#!/usr/bin/env bash
# Phase 7: one-command, end-to-end reproduction of the whole pipeline on a smaller event count.
#
#   make demo
#
# Runs: up -> generate -> features -> dbt-test -> train -> load-online -> serve (background)
# -> a couple of sample /score calls -> summary, then stops the server it started.
#
# Uses an ISOLATED data/demo/ tree (bronze/silver/gold), not the main data/ directory, so this
# never clobbers a larger dataset you already built by running the phases individually --
# generate/produce always deletes existing event_date partitions in its output dir on every run
# (see signallake/generate/producer.py), which would otherwise be destructive here.
#
# Training and the online model registry ARE shared with the rest of the project on purpose:
# MLflow's registry is versioned (this just adds a new version of signallake-anomaly-detector),
# and `make serve` always serves whatever version is latest -- exactly what a one-command demo
# should do.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DEMO_EVENTS="${DEMO_EVENTS:-200000}"
DEMO_DIR="data/demo"
BRONZE="$DEMO_DIR/bronze"
SILVER="$DEMO_DIR/silver"
GOLD="$DEMO_DIR/gold"
SERVE_HOST="http://localhost:8000"

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "==> stopping the demo's serve process (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

step() { echo; echo "==> [$1/8] $2"; }

step 1 "make up (Kafka, Redis, MLflow)"
make up

step 2 "generate ($DEMO_EVENTS events -> $BRONZE)"
uv run python -m signallake.generate.producer \
    --num-events "$DEMO_EVENTS" --to-parquet --output-dir "$BRONZE"

step 3 "features ($BRONZE -> $SILVER, $GOLD)"
uv run python -m signallake.features.build_features \
    --bronze-dir "$BRONZE" --silver-dir "$SILVER" --gold-dir "$GOLD"

step 4 "dbt-test (quality gates on $SILVER)"
SIGNALLAKE_DATA_DIR="$DEMO_DIR" SIGNALLAKE_DUCKDB_PATH="$DEMO_DIR/dbt_demo.duckdb" \
    uv run dbt build --project-dir dbt/signallake_dbt --profiles-dir dbt/signallake_dbt

step 5 "train (IsolationForest + 3 HGBC runs -> MLflow, registers a new model version)"
uv run python -m signallake.train.train --gold-dir "$GOLD"

step 6 "load-online (latest per-customer features -> Redis)"
uv run python -m signallake.serve.load_online_store --gold-dir "$GOLD"

step 7 "serve (background)"
SERVE_WORKERS=1 make serve > "$DEMO_DIR/serve.log" 2>&1 &
SERVER_PID=$!
n=0
until curl -s -o /dev/null -m 1 "$SERVE_HOST/health" || [ "$n" -ge 60 ]; do
    sleep 1
    n=$((n + 1))
done
if ! curl -s -o /dev/null -m 1 "$SERVE_HOST/health"; then
    echo "serve did not become healthy -- see $DEMO_DIR/serve.log" >&2
    exit 1
fi
echo "healthy: $(curl -s "$SERVE_HOST/health")"

step 8 "sample /score calls"
CUSTOMER_ID=$(uv run python -c "
import redis
from signallake.common.config import get_settings
r = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)
keys = r.keys('feat:*')
print(keys[0].split(':', 1)[1] if keys else '')
")
if [ -z "$CUSTOMER_ID" ]; then
    echo "no customers loaded in Redis -- load-online step must have failed silently" >&2
    exit 1
fi
echo "-- normal-looking transaction ($CUSTOMER_ID, \$42.50):"
curl -s -X POST "$SERVE_HOST/score" -H "Content-Type: application/json" \
    -d "{\"customer_id\": \"$CUSTOMER_ID\", \"amount\": 42.50}"
echo
echo "-- large, new-device transaction ($CUSTOMER_ID, \$50,000):"
curl -s -X POST "$SERVE_HOST/score" -H "Content-Type: application/json" \
    -d "{\"customer_id\": \"$CUSTOMER_ID\", \"amount\": 50000, \"is_new_device\": true}"
echo

echo
echo "================================================================"
echo " SignalLake demo complete"
echo "================================================================"
echo " events generated:     $DEMO_EVENTS  (data/demo/, isolated from your main dataset)"
echo " dbt quality gates:    passed (see output above)"
echo " model:                trained + registered a new MLflow version of"
echo "                       signallake-anomaly-detector"
echo " online store:         loaded into Redis"
echo " serving:               $SERVE_HOST  (stopped when this script exits)"
echo
echo " MLflow UI:   http://localhost:5000"
echo " Re-run just serving:  make serve"
echo " Full numbers (2.1M-event runs): reports/METRICS.md"
echo "================================================================"
