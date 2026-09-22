# Talking about SignalLake

This is a reference for describing SignalLake in an interview and for defending its five
headline numbers under follow-up questions. The rule behind every number here is the same one
the build followed: **report what the pipeline actually produced, never a target** (see
`SignalLake_BUILD_PLAN.md` §1). That's also the strongest thing to say about this project if
asked "why should I trust these numbers" -- they're reproducible by running `make demo` or the
phase-by-phase commands in the README, not hand-picked.

## STAR

**Situation.** Anomaly/fraud-style transaction detection is a common real-world ML problem, but
most portfolio projects either use a toy CSV in a notebook or skip the parts that make it
*production* work: a real ingestion path, data-quality gates, feature-store serving, and load
testing. I wanted a project that exercised the full path a transaction-anomaly system would
actually need, without needing a company's real data or cloud budget to do it.

**Task.** Build a local, end-to-end anomaly detection platform: synthetic transaction events with
injected, tunable-difficulty anomalies, flowing through real-time ingestion, a proper feature
store, a tracked and registered model, and an online scoring API -- all orchestrated, all
measured honestly, all running on a single WSL2 machine.

**Action.** Built it in seven phases, each with its own acceptance criteria before moving on:
1. A synthetic event generator (2,000 customers, 30 days, 2.1M events) with four anomaly types
   (velocity spikes, amount spikes, new-device+geo-jumps, failed-attempt bursts) and *tunable*
   separability (`--signal-strength`, `--noise`) so the detection problem isn't trivial.
2. Kafka ingestion via Spark Structured Streaming into a bronze layer, then a batch Spark job
   computing 34 windowed features (5m/10m/1h/24h trailing aggregates) into a gold layer.
3. dbt quality gates (12 tests: not-null, uniqueness, accepted-values, custom range checks) on
   the silver layer, demonstrated actually catching injected bad data.
4. Three supervised HistGradientBoostingClassifier runs (12/22/34 features) plus an
   IsolationForest baseline, tracked and compared in MLflow, best model registered with lineage
   tags (data range, schema hash, git commit).
5. A FastAPI + Redis online-serving layer, load-tested with Locust.
6. Airflow orchestration of the whole batch pipeline, plus a separate benchmark DAG comparing
   full vs. incremental recompute cost.
7. This documentation, a one-command demo (`make demo`), and the consolidated metrics file.

**Result.** A working, reproducible platform: 0.880 F1 on the registered anomaly detector (a
natural 0.797 -> 0.847 -> 0.880 progression across the three feature-tier runs, not tuned to hit
a number), a documented 240ms p95 under load (see the honest defense below -- it missed the
<120ms target, and that's part of the story), and a 63.1% reduction in recompute work units from
the incremental-vs-full benchmark. Along the way, three real infrastructure bugs got found and
fixed with root-cause diagnosis: a Docker bind-mount permission issue, severe BLAS thread
oversubscription under multi-process serving (the single biggest latency fix, ~4x), and an
Airflow task templating collision.

## Defending each metric

### 1. "2.1M events"

**Say:** "2.1 million *synthetic* transaction events, generated locally with injected anomalies
at a 3% rate."

**Don't say:** "2.1 million transactions" without "synthetic" -- it invites the assumption this
is real financial data, which it never was.

**If pushed on realism:** the generator gives each of 2,000 customers a stable baseline profile
(typical amount range, home location, usual devices) and samples normal events around it, so the
data has realistic per-customer structure, not just IID noise -- but it's still synthetic, and
that's stated everywhere the number appears.

### 2. "0.880 F1" (not "88% accuracy")

**Say:** "0.88 F1 -- precision 0.915, recall 0.848 -- on a supervised HistGradientBoostingClassifier
over all 34 features, measured on a held-out time-based test split."

**Don't say:** "88% accuracy." Accuracy on a ~3%-anomaly-rate dataset is a meaningless metric (a
model that predicts "never anomaly" scores ~97% accuracy) -- F1 is the right metric *because*
the classes are imbalanced, and conflating the two undermines the whole result.

**If pushed on methodology:** the split is by day (train on the earliest ~21 days, validate on
the next ~4, test on the last ~5), not a random row split, so no future event ever leaks into
training. The decision threshold is chosen on the validation set to maximize F1, then applied
once to the untouched test set -- the reported number is what that threshold produced on data it
never influenced. The progression across feature tiers (12 features -> 0.797, 22 -> 0.847, 34 ->
0.880) is genuinely different feature sets, not the same model re-scored.

**If pushed on "why not higher":** the generator caps separability on purpose (~15% of anomalies
are deliberately "subtle," plus label noise) specifically so the ceiling isn't 1.0 -- a perfect
score on synthetic data you control the difficulty of would be the *less* credible result.

### 3. "240ms p95" (in load tests, not per-request)

**Say:** "240ms p95, measured with Locust at 80 concurrent users over 2 minutes on my own dev
machine -- which was also running Kafka, Redis, MLflow and the load generator itself at the same
time. It missed my <120ms target."

**Don't say:** "the API serves in 240ms" as a blanket claim, and don't hide that it missed the
target -- that's the more interesting part of the story, not a weakness to bury.

**If pushed on why it's not lower:** a single request, once warmed up, serves in ~9ms end to end
-- the model is fully in memory and the only I/O on the hot path is one Redis GET. The gap
between that and 240ms under load is real infrastructure contention, and three specific fixes
were made and verified with before/after measurements: (1) moving from one process to multiple
worker processes, since a single process serializes CPU-bound scoring on Python's GIL; (2)
discovering (via `/proc/<pid>/task`) that each worker process was independently spinning up
~120 OS threads for BLAS/OpenMP, so capping those to 1 thread per worker cut p95 from >2000ms to
490ms in one change; (3) replacing a per-request pandas DataFrame construction with a raw numpy
array, roughly halving it again to 240ms. What's left is genuine contention on an 8-core box
running the whole stack plus the load generator -- not a code defect. That's a stronger answer
than a clean number would have been, because it shows the debugging, not just the result.

### 4. "63.1% fewer work units" (vs. full recompute)

**Say:** "63.1% fewer work units in an incremental-vs-full recompute benchmark -- 'work unit' is
defined as the number of raw bronze rows in the partitions actually rebuilt."

**Don't say:** "64% faster" without qualifying what got 64% smaller. Work units and wall-clock
are reported *separately* on purpose (see `reports/METRICS.md`) -- they don't have to match, and
in this run they didn't exactly (wall-clock reduction varied run to run, ~52-70%, since it
includes each task's own Spark startup cost).

**If pushed on the scenario:** the incremental run touches 11 of 30 day-partitions (the newest
day, always "new," plus every third of the older days, simulating scattered late-arriving
corrections) -- a *designed* scenario, since the underlying synthetic batch has no real
late-arrival dimension to measure directly. That's stated in the benchmark's own module
docstring, not hidden. What's real is the row-count-based reduction from actually running the
same Spark feature-build job over the full set vs. that partition subset, timed both ways.

### 5. "34 reusable features"

**Say:** "34 features across four trailing windows -- 5 minutes, 10 minutes, 1 hour, 24 hours --
covering velocity, amount statistics, current-vs-baseline z-scores, device/geo change signals,
failure-rate signals, and temporal features."

**If pushed on "reusable":** the same catalog is tagged into three cumulative tiers (v1/v2/v3,
12/22/34 features) purely by editing a dict in one module -- that tiering is what produced the
three training runs' feature sets without duplicating any feature logic, and it's the same
feature vector both offline (Spark batch) and online (the Redis-served value the API reads).

## Things worth mentioning even if not asked

- The p95 miss and the three specific fixes for it (above) is usually a better story than any of
  the metrics that hit target -- it's the part that demonstrates debugging under pressure.
- The dbt bad-data demo (`make dbt-bad-data-demo`) actually injects five kinds of bad rows into a
  scratch copy of the data and shows the gates fail, naming the offending test and row count,
  then shows the same suite pass clean -- a concrete "I built quality gates and verified they
  work" answer instead of an assertion.
- Two infrastructure bugs outside the "happy path" got found and fixed with root-cause diagnosis:
  a Docker Desktop/WSL2 bind-mount permission issue that silently hung MLflow artifact uploads,
  and an Airflow `BashOperator.env` field colliding with Jinja's per-value template-file lookup
  (a `GIT_ASKPASS=...sh` env var got misread as a template reference). Both are the kind of
  issue that only shows up when you actually run the system under real conditions, not in a
  notebook.
