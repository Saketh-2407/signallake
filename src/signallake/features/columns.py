"""The 34-feature catalog (see SignalLake_BUILD_PLAN.md §4), as small, independently testable
PySpark column-adding functions, keyed by `customer_id` over trailing windows [5m, 10m, 1h, 24h].

Each `add_*_features` function takes a DataFrame with at least `customer_id`, `timestamp`,
`amount`, `status`, `device_id`, `location`, `is_new_device` and returns it with its group's
columns appended. Every function computes whatever windowed helpers it needs internally (via the
shared `_` helpers below), so any one of them can be unit-tested on a tiny hand-built DataFrame
without first calling the others. `build_gold_features` chains all seven groups and is what
`build_features.py` actually calls.

Windows are trailing and inclusive of the current row: "5m" means "this event plus every prior
event from the same customer in the preceding 5 minutes". `amount_vs_cust_mean_24h` and
`amount_zscore_24h` are the one exception -- they compare the current amount against the
customer's PRIOR 24h baseline (current event excluded), so an outlier doesn't dilute its own
baseline. Every feature is guaranteed non-null: the few structurally-empty cases (a customer's
first-ever event, a single-row window) get a documented, sensible fill.
"""

from pyspark.sql import Column, DataFrame, Window, WindowSpec
from pyspark.sql import functions as F

WINDOWS_SECONDS = {"5m": 300, "10m": 600, "1h": 3600, "24h": 86400}
NO_PRIOR_EVENT_SENTINEL_S = 30 * 86400.0  # "no transaction in the last 30 days" (our full span)
GEO_VELOCITY_THRESHOLD_S = 1800  # a location change within 30 min of the last event is suspicious

# name -> tier it first appears in. Cumulative: v2 = v1 + v2's own, v3 = v2 + v3's own (all 34).
FEATURE_TIERS: dict[str, str] = {
    # v1 (12): the obvious velocity/amount/failure/time signals.
    "txn_count_5m": "v1",
    "txn_count_10m": "v1",
    "txn_count_1h": "v1",
    "txn_count_24h": "v1",
    "amount": "v1",
    "amount_mean_24h": "v1",
    "amount_max_24h": "v1",
    "failed_count_1h": "v1",
    "failed_ratio_1h": "v1",
    "hour_of_day": "v1",
    "is_night": "v1",
    "new_device_flag": "v1",
    # v2 (+10 = 22): finer-grained amount stats, 1h device/geo, weekend.
    "amount_mean_1h": "v2",
    "amount_max_1h": "v2",
    "amount_std_1h": "v2",
    "amount_std_24h": "v2",
    "amount_sum_1h": "v2",
    "amount_sum_24h": "v2",
    "amount_vs_cust_mean_24h": "v2",
    "distinct_devices_1h": "v2",
    "distinct_locations_1h": "v2",
    "is_weekend": "v2",
    # v3 (+12 = 34): baseline z-score, 24h device/geo, and the derived ratios/flags.
    "amount_zscore_24h": "v3",
    "distinct_devices_24h": "v3",
    "distinct_locations_24h": "v3",
    "location_changes_1h": "v3",
    "failed_count_24h": "v3",
    "time_since_last_txn_s": "v3",
    "txn_freq_per_min_10m": "v3",
    "interarrival_std_1h": "v3",
    "amount_to_max_ratio_24h": "v3",
    "velocity_ratio_5m_vs_24h": "v3",
    "geo_velocity_flag": "v3",
    "amount_pct_of_daily_sum": "v3",
}
FEATURE_NAMES: list[str] = list(FEATURE_TIERS)  # catalog order; == v3
_TIER_RANK = {"v1": 1, "v2": 2, "v3": 3}
assert len(FEATURE_NAMES) == 34


def features_for_tier(tier: str) -> list[str]:
    """v1 -> its 12 features, v2 -> those 12 + its 10 (22), v3 -> all 34."""
    rank = _TIER_RANK[tier]
    return [name for name, t in FEATURE_TIERS.items() if _TIER_RANK[t] <= rank]


def _with_epoch(df: DataFrame) -> DataFrame:
    """`ts_epoch`: UTC epoch seconds. Safe to add repeatedly -- later calls just overwrite it."""
    return df.withColumn("ts_epoch", F.col("timestamp").cast("long"))


def _row_window() -> WindowSpec:
    """Per-customer event order, ties on identical timestamps broken deterministically by id."""
    return Window.partitionBy("customer_id").orderBy("ts_epoch", "event_id")


def _range_window(seconds: int) -> WindowSpec:
    """Trailing `seconds`-wide window, inclusive of the current row."""
    return Window.partitionBy("customer_id").orderBy("ts_epoch").rangeBetween(-seconds, 0)


def _seconds_since_prev_event(df: DataFrame) -> Column:
    """Gap since this customer's previous event, in seconds; `NO_PRIOR_EVENT_SENTINEL_S` if none."""
    prev = F.lag("ts_epoch").over(_row_window())
    return F.coalesce(F.col("ts_epoch") - prev, F.lit(NO_PRIOR_EVENT_SENTINEL_S)).cast("double")


def _location_changed(df: DataFrame) -> Column:
    """1 if `location` differs from this customer's previous event, else 0 (0 if there is none)."""
    prev = F.lag("location").over(_row_window())
    return F.when(prev.isNotNull() & (F.col("location") != prev), 1).otherwise(0)


def add_velocity_features(df: DataFrame) -> DataFrame:
    """txn_count_5m/10m/1h/24h: transactions by this customer in each trailing window."""
    df = _with_epoch(df)
    for name, seconds in WINDOWS_SECONDS.items():
        df = df.withColumn(f"txn_count_{name}", F.count(F.lit(1)).over(_range_window(seconds)))
    return df


def add_amount_stat_features(df: DataFrame) -> DataFrame:
    """amount_mean/max/std/sum_1h/24h: plain rolling stats, inclusive of the current amount."""
    df = _with_epoch(df)
    for name in ("1h", "24h"):
        w = _range_window(WINDOWS_SECONDS[name])
        df = (
            df.withColumn(f"amount_mean_{name}", F.avg("amount").over(w))
            .withColumn(f"amount_max_{name}", F.max("amount").over(w))
            # stddev_samp is null for a 1-row window (no variance defined yet) -> 0.0.
            .withColumn(
                f"amount_std_{name}", F.coalesce(F.stddev_samp("amount").over(w), F.lit(0.0))
            )
            .withColumn(f"amount_sum_{name}", F.sum("amount").over(w))
        )
    return df


def add_current_vs_baseline_features(df: DataFrame) -> DataFrame:
    """amount, amount_vs_cust_mean_24h, amount_zscore_24h -- vs. the PRIOR 24h (self excluded).

    A customer's very first event has no prior baseline: amount_vs_cust_mean_24h defaults to 1.0
    (looks exactly average) and amount_zscore_24h to 0.0 (no evidence of deviation yet).
    """
    df = _with_epoch(df)
    w = _range_window(WINDOWS_SECONDS["24h"])
    n = F.count(F.lit(1)).over(w)
    s = F.sum("amount").over(w)
    ss = F.sum(F.col("amount") ** 2).over(w)
    prior_n = (n - 1).cast("double")
    prior_mean = F.when(prior_n > 0, (s - F.col("amount")) / prior_n).otherwise(F.col("amount"))
    prior_var = F.when(
        prior_n > 1,
        F.greatest(
            ((ss - F.col("amount") ** 2) - prior_n * prior_mean**2) / (prior_n - 1), F.lit(0.0)
        ),
    ).otherwise(F.lit(0.0))
    prior_std = F.sqrt(prior_var)
    return df.withColumn("amount_vs_cust_mean_24h", F.col("amount") / prior_mean).withColumn(
        "amount_zscore_24h",
        F.when(prior_std > 1e-6, (F.col("amount") - prior_mean) / prior_std).otherwise(F.lit(0.0)),
    )


def add_device_geo_features(df: DataFrame) -> DataFrame:
    """distinct_devices/locations_1h/24h, location_changes_1h, new_device_flag."""
    df = _with_epoch(df)
    for name in ("1h", "24h"):
        w = _range_window(WINDOWS_SECONDS[name])
        df = df.withColumn(
            f"distinct_devices_{name}", F.size(F.collect_set("device_id").over(w))
        ).withColumn(f"distinct_locations_{name}", F.size(F.collect_set("location").over(w)))
    df = df.withColumn(
        "location_changes_1h", F.sum(_location_changed(df)).over(_range_window(3600))
    )
    return df.withColumn("new_device_flag", F.col("is_new_device").cast("int"))


def add_failure_features(df: DataFrame) -> DataFrame:
    """failed_count_1h/24h, failed_ratio_1h, time_since_last_txn_s."""
    df = _with_epoch(df)
    is_failed = (F.col("status") == "failed").cast("int")
    for name in ("1h", "24h"):
        df = df.withColumn(
            f"failed_count_{name}", F.sum(is_failed).over(_range_window(WINDOWS_SECONDS[name]))
        )
    txn_count_1h = F.count(F.lit(1)).over(_range_window(WINDOWS_SECONDS["1h"]))
    return df.withColumn("failed_ratio_1h", F.col("failed_count_1h") / txn_count_1h).withColumn(
        "time_since_last_txn_s", _seconds_since_prev_event(df)
    )


def add_temporal_features(df: DataFrame) -> DataFrame:
    """hour_of_day, is_night, is_weekend, txn_freq_per_min_10m, interarrival_std_1h.

    Requires the Spark session timezone to be UTC (set in `build_features.py`), or `hour_of_day` /
    `is_night` / `is_weekend` will be shifted by the machine's local offset.
    """
    df = _with_epoch(df)
    gap = _seconds_since_prev_event(df)
    hour = F.hour("timestamp")
    txn_count_10m = F.count(F.lit(1)).over(_range_window(WINDOWS_SECONDS["10m"]))
    return (
        df.withColumn("hour_of_day", hour)
        .withColumn("is_night", (hour < 6).cast("int"))
        .withColumn("is_weekend", F.dayofweek("timestamp").isin(1, 7).cast("int"))  # 1=Sun, 7=Sat
        .withColumn("txn_freq_per_min_10m", txn_count_10m / 10.0)
        # A single-gap (or gap-free) 1h window has no variance yet -> 0.0.
        .withColumn(
            "interarrival_std_1h",
            F.coalesce(F.stddev_samp(gap).over(_range_window(3600)), F.lit(0.0)),
        )
    )


def add_derived_features(df: DataFrame) -> DataFrame:
    """amount_to_max_ratio_24h, velocity_ratio_5m_vs_24h, geo_velocity_flag,
    amount_pct_of_daily_sum."""
    df = _with_epoch(df)
    w24 = _range_window(WINDOWS_SECONDS["24h"])
    amount_max_24h = F.max("amount").over(w24)
    amount_sum_24h = F.sum("amount").over(w24)
    txn_count_5m = F.count(F.lit(1)).over(_range_window(WINDOWS_SECONDS["5m"]))
    txn_count_24h = F.count(F.lit(1)).over(w24)
    gap = _seconds_since_prev_event(df)
    return (
        df.withColumn("amount_to_max_ratio_24h", F.col("amount") / amount_max_24h)
        # Rate in the last 5 minutes vs. a typical 5-minute slice of the last 24h (288 = 24h/5m).
        .withColumn("velocity_ratio_5m_vs_24h", txn_count_5m / (txn_count_24h / 288.0))
        .withColumn(
            "geo_velocity_flag",
            ((_location_changed(df) == 1) & (gap < GEO_VELOCITY_THRESHOLD_S)).cast("int"),
        )
        .withColumn("amount_pct_of_daily_sum", F.col("amount") / amount_sum_24h)
    )


_GROUPS = (
    add_velocity_features,
    add_amount_stat_features,
    add_current_vs_baseline_features,
    add_device_geo_features,
    add_failure_features,
    add_temporal_features,
    add_derived_features,
)


def build_gold_features(df: DataFrame) -> DataFrame:
    """Apply every feature group and return the DataFrame with all 34 columns appended."""
    for group in _GROUPS:
        df = group(df)
    return df
