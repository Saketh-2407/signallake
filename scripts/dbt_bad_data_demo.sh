#!/usr/bin/env bash
# Phase 3 demo: prove the dbt quality gates actually catch bad data.
#
# Copies a handful of real silver partitions into a scratch directory, corrupts
# a few rows there (null id, duplicate id, negative amount on a success row,
# an invalid status, a future-dated event), runs `dbt test` against the scratch
# copy and shows it fail, then runs `dbt test` again against the real (clean)
# data and shows it pass. The real data/ directory is never touched.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SCRATCH_DIR="data/_dbt_bad_demo"
DUCKDB_PATH="${SCRATCH_DIR}/signallake_baddemo.duckdb"

cleanup() { rm -rf "$SCRATCH_DIR"; }
trap cleanup EXIT

rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

echo "==> Building a corrupted copy of silver under ${SCRATCH_DIR}/silver ..."
uv run python - "$SCRATCH_DIR" <<'PY'
import sys
import duckdb
import pandas as pd

scratch_dir = sys.argv[1]

con = duckdb.connect()
con.execute(f"""
    copy (
        select * from read_parquet('data/silver/**/*.parquet', hive_partitioning=true)
        order by event_date
        limit 20000
    ) to '{scratch_dir}/silver' (format parquet, partition_by event_date, overwrite_or_ignore true)
""")

first_partition = con.execute(f"""
    select event_date from read_parquet('{scratch_dir}/silver/**/*.parquet', hive_partitioning=true)
    group by event_date order by event_date limit 1
""").fetchone()[0]

part_dir = f"{scratch_dir}/silver/event_date={first_partition}"
df = con.execute(f"select * from read_parquet('{part_dir}/*.parquet')").fetchdf()

dup_row = df.iloc[[0]].copy()
dup_row["amount"] = -50.0
dup_row["status"] = "success"  # negative amount on a success row: gate violation

future_row = df.iloc[[1]].copy()
future_row["timestamp"] = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=5)

df.loc[df.index[2], "event_id"] = None          # not_null violation
df.loc[df.index[3], "status"] = "pending"        # accepted_values violation

corrupted = pd.concat([df, dup_row, future_row], ignore_index=True)  # dup_row's event_id duplicates df.iloc[0] -> uniqueness violation

import glob
import os

for f in glob.glob(f"{part_dir}/*.parquet"):
    os.remove(f)

con.register("corrupted_df", corrupted)
con.execute(f"copy corrupted_df to '{part_dir}/corrupted.parquet' (format parquet)")

print(f"Corrupted partition: event_date={first_partition}")
print(f"Injected: 1 null event_id, 1 duplicate event_id, 1 negative amount on a "
      f"success row, 1 invalid status ('pending'), 1 future-dated event.")
PY

echo
echo "==> Running dbt build against the CORRUPTED scratch copy (expect test failures) ..."
set +e
SIGNALLAKE_DATA_DIR="$SCRATCH_DIR" SIGNALLAKE_DUCKDB_PATH="$DUCKDB_PATH" \
    uv run dbt build --project-dir dbt/signallake_dbt --profiles-dir dbt/signallake_dbt
BAD_EXIT=$?
set -e
echo "==> dbt build exit code on corrupted data: ${BAD_EXIT} (non-zero = gate caught it)"

echo
echo "==> Running dbt build against the REAL, clean data (expect all PASS) ..."
uv run dbt build --project-dir dbt/signallake_dbt --profiles-dir dbt/signallake_dbt

echo
echo "==> Done. Scratch data at ${SCRATCH_DIR} will be removed; data/ is untouched."
if [ "$BAD_EXIT" -eq 0 ]; then
    echo "WARNING: corrupted-data run did not fail as expected." >&2
    exit 1
fi
