"""Unit tests for the windowed feature aggregates, on tiny hand-built DataFrames with known
expected values (some computed inline via `statistics.stdev` on the exact set of values each
window is expected to select -- still a real check: a wrong window boundary would pull in a
different set of gaps/amounts and the computed expectation would no longer match Spark's output).
"""

import statistics
from datetime import UTC, datetime, timedelta

import pytest
from pyspark.sql import Row

from signallake.features.columns import (
    FEATURE_NAMES,
    add_amount_stat_features,
    add_current_vs_baseline_features,
    add_derived_features,
    add_device_geo_features,
    add_failure_features,
    add_temporal_features,
    add_velocity_features,
    features_for_tier,
)

T0 = datetime(2026, 8, 3, 10, 0, 0, tzinfo=UTC)  # a Monday


def rows(spark, records, extra_cols=()):
    """Build a DataFrame from dicts of {customer_id, event_id, timestamp, ...extra_cols}."""
    cols = ["customer_id", "event_id", "timestamp", *extra_cols]
    return spark.createDataFrame([Row(**{c: r[c] for c in cols}) for r in records])


def test_velocity_windows_are_trailing_and_per_customer(spark):
    df = rows(
        spark,
        [
            {"customer_id": "c1", "event_id": "e1", "timestamp": T0},
            {"customer_id": "c1", "event_id": "e2", "timestamp": T0 + timedelta(minutes=2)},
            {"customer_id": "c1", "event_id": "e3", "timestamp": T0 + timedelta(minutes=6)},
            {"customer_id": "c1", "event_id": "e4", "timestamp": T0 + timedelta(minutes=90)},
            {"customer_id": "c2", "event_id": "e5", "timestamp": T0 + timedelta(minutes=2)},
        ],
    )
    out = {r.event_id: r for r in add_velocity_features(df).collect()}

    assert (out["e1"].txn_count_5m, out["e1"].txn_count_10m) == (1, 1)
    # e2 is 2 min after e1: inside the 5m/10m/1h/24h windows, so both count.
    assert (out["e2"].txn_count_5m, out["e2"].txn_count_10m, out["e2"].txn_count_1h) == (2, 2, 2)
    # e3 is 6 min after e1 (outside 5m) but only 4 min after e2 (inside everything else).
    assert (out["e3"].txn_count_5m, out["e3"].txn_count_10m, out["e3"].txn_count_1h) == (2, 3, 3)
    # e4 is 90 min after e1: outside 1h of e1/e2/e3, but still the same day (24h).
    assert (out["e4"].txn_count_1h, out["e4"].txn_count_24h) == (1, 4)
    # c2's single event is untouched by c1's cluster despite the shared timestamp.
    assert (out["e5"].txn_count_5m, out["e5"].txn_count_24h) == (1, 1)


def test_amount_stats_are_windowed_and_exclude_events_outside_the_frame(spark):
    df = rows(
        spark,
        [
            {"customer_id": "c1", "event_id": "e1", "timestamp": T0, "amount": 10.0},
            {
                "customer_id": "c1",
                "event_id": "e2",
                "timestamp": T0 + timedelta(minutes=1),
                "amount": 20.0,
            },
            {
                "customer_id": "c1",
                "event_id": "e3",
                "timestamp": T0 + timedelta(minutes=90),
                "amount": 999.0,
            },
        ],
        extra_cols=("amount",),
    )
    out = {r.event_id: r for r in add_amount_stat_features(df).collect()}

    assert out["e1"].amount_mean_1h == pytest.approx(10.0)
    assert out["e1"].amount_std_1h == pytest.approx(0.0)  # single-row window: no variance defined
    e2 = out["e2"]
    assert (e2.amount_mean_1h, e2.amount_max_1h, e2.amount_sum_1h) == (15.0, 20.0, 30.0)
    assert e2.amount_std_1h == pytest.approx(statistics.stdev([10.0, 20.0]))
    # e3 is 90 min later: outside the 1h window, so it only ever sees itself.
    assert (out["e3"].amount_mean_1h, out["e3"].amount_std_1h) == (999.0, 0.0)


def test_baseline_excludes_the_current_event_and_needs_two_priors_for_a_zscore(spark):
    df = rows(
        spark,
        [
            {"customer_id": "c1", "event_id": "e1", "timestamp": T0, "amount": 10.0},
            {
                "customer_id": "c1",
                "event_id": "e2",
                "timestamp": T0 + timedelta(minutes=1),
                "amount": 20.0,
            },
            {
                "customer_id": "c1",
                "event_id": "e3",
                "timestamp": T0 + timedelta(minutes=2),
                "amount": 100.0,
            },
        ],
        extra_cols=("amount",),
    )
    out = {r.event_id: r for r in add_current_vs_baseline_features(df).collect()}

    # e1: no prior history -> looks exactly average, zero deviation.
    assert (out["e1"].amount_vs_cust_mean_24h, out["e1"].amount_zscore_24h) == (1.0, 0.0)
    # e2: one prior event (10.0) -> the ratio is well-defined, but one point has no variance.
    assert out["e2"].amount_vs_cust_mean_24h == pytest.approx(20.0 / 10.0)
    assert out["e2"].amount_zscore_24h == pytest.approx(0.0)
    # e3: two priors (10.0, 20.0) -> both the ratio and the z-score are now well-defined.
    prior_mean, prior_std = 15.0, statistics.stdev([10.0, 20.0])
    assert out["e3"].amount_vs_cust_mean_24h == pytest.approx(100.0 / prior_mean)
    assert out["e3"].amount_zscore_24h == pytest.approx((100.0 - prior_mean) / prior_std)


def test_device_geo_features(spark):
    df = rows(
        spark,
        [
            {
                "customer_id": "c1",
                "event_id": "e1",
                "timestamp": T0,
                "device_id": "d1",
                "location": "NYC",
                "is_new_device": False,
            },
            {
                "customer_id": "c1",
                "event_id": "e2",
                "timestamp": T0 + timedelta(minutes=1),
                "device_id": "d1",
                "location": "NYC",
                "is_new_device": False,
            },
            {
                "customer_id": "c1",
                "event_id": "e3",
                "timestamp": T0 + timedelta(minutes=2),
                "device_id": "d2",
                "location": "LA",
                "is_new_device": True,
            },
            {
                "customer_id": "c1",
                "event_id": "e4",
                "timestamp": T0 + timedelta(minutes=3),
                "device_id": "d2",
                "location": "LA",
                "is_new_device": False,
            },
        ],
        extra_cols=("device_id", "location", "is_new_device"),
    )
    out = {r.event_id: r for r in add_device_geo_features(df).collect()}

    assert (out["e1"].distinct_devices_1h, out["e1"].distinct_locations_1h) == (1, 1)
    assert (out["e3"].distinct_devices_1h, out["e3"].distinct_locations_1h) == (2, 2)
    assert (out["e1"].new_device_flag, out["e3"].new_device_flag) == (0, 1)
    # Only e3 changes location relative to its predecessor (e4 repeats e3's LA).
    assert [out[e].location_changes_1h for e in ("e1", "e2", "e3", "e4")] == [0, 0, 1, 1]


def test_failure_features_and_the_no_prior_event_sentinel(spark):
    df = rows(
        spark,
        [
            {"customer_id": "c1", "event_id": "e1", "timestamp": T0, "status": "failed"},
            {
                "customer_id": "c1",
                "event_id": "e2",
                "timestamp": T0 + timedelta(seconds=90),
                "status": "failed",
            },
            {
                "customer_id": "c1",
                "event_id": "e3",
                "timestamp": T0 + timedelta(seconds=150),
                "status": "success",
            },
        ],
        extra_cols=("status",),
    )
    out = {r.event_id: r for r in add_failure_features(df).collect()}

    assert out["e1"].time_since_last_txn_s == pytest.approx(30 * 86400.0)  # documented sentinel
    assert out["e2"].time_since_last_txn_s == pytest.approx(90.0)
    assert out["e3"].time_since_last_txn_s == pytest.approx(60.0)
    assert (out["e1"].failed_count_1h, out["e3"].failed_count_1h) == (1, 2)
    assert out["e3"].failed_ratio_1h == pytest.approx(2 / 3)


def test_temporal_features_hour_weekend_and_a_clean_interarrival_std(spark):
    saturday_night = datetime(2026, 8, 1, 2, 0, 0, tzinfo=UTC)  # 2026-08-01 is a Saturday
    e0 = saturday_night - timedelta(hours=2)  # 7200s before e1: outside every 1h window below
    df = rows(
        spark,
        [
            {"customer_id": "c1", "event_id": "e0", "timestamp": e0},
            {"customer_id": "c1", "event_id": "e1", "timestamp": saturday_night},
            {
                "customer_id": "c1",
                "event_id": "e2",
                "timestamp": saturday_night + timedelta(seconds=120),
            },
            {
                "customer_id": "c1",
                "event_id": "e3",
                "timestamp": saturday_night + timedelta(seconds=300),
            },
            {
                "customer_id": "c2",
                "event_id": "e4",
                "timestamp": T0.replace(hour=14),
            },  # Monday, daytime
        ],
    )
    out = {r.event_id: r for r in add_temporal_features(df).collect()}

    assert (out["e1"].hour_of_day, out["e1"].is_night, out["e1"].is_weekend) == (2, 1, 1)
    assert (out["e4"].hour_of_day, out["e4"].is_night, out["e4"].is_weekend) == (14, 0, 0)
    # e0 is 2h before e1, well outside e1's own 1h window: that window holds only e1 (no variance).
    assert out["e1"].interarrival_std_1h == pytest.approx(0.0)
    # At e3, e0 has aged out of the 1h window too, leaving exactly {e1, e2, e3} with real
    # (non-sentinel) gaps: e1 vs e0 (7200s), e2 vs e1 (120s), e3 vs e2 (180s).
    assert out["e3"].interarrival_std_1h == pytest.approx(statistics.stdev([7200.0, 120.0, 180.0]))


def test_derived_features(spark):
    df = rows(
        spark,
        [
            {
                "customer_id": "c1",
                "event_id": "e1",
                "timestamp": T0,
                "amount": 10.0,
                "location": "NYC",
            },
            {
                "customer_id": "c1",
                "event_id": "e2",
                "timestamp": T0 + timedelta(minutes=1),
                "amount": 20.0,
                "location": "NYC",
            },
            {
                "customer_id": "c1",
                "event_id": "e3",
                "timestamp": T0 + timedelta(minutes=2),
                "amount": 200.0,
                "location": "LA",
            },
            {
                "customer_id": "c1",
                "event_id": "e4",
                "timestamp": T0 + timedelta(minutes=3),
                "amount": 50.0,
                "location": "LA",
            },
        ],
        extra_cols=("amount", "location"),
    )
    out = {r.event_id: r for r in add_derived_features(df).collect()}

    assert out["e1"].amount_to_max_ratio_24h == pytest.approx(
        1.0
    )  # only event so far -> its own max
    assert out["e4"].amount_to_max_ratio_24h == pytest.approx(
        50.0 / 200.0
    )  # e3's 200 is still the max
    assert out["e3"].amount_pct_of_daily_sum == pytest.approx(200.0 / 230.0)  # 200 of (10+20+200)
    assert out["e1"].velocity_ratio_5m_vs_24h == pytest.approx(288.0)  # one event in both windows
    # e3 changes location (NYC -> LA) 60s after e2: within the 30-min geo-velocity threshold.
    assert (
        out["e2"].geo_velocity_flag,
        out["e3"].geo_velocity_flag,
        out["e4"].geo_velocity_flag,
    ) == (0, 1, 0)


def test_feature_tiers_are_cumulative_and_cover_the_whole_catalog():
    v1, v2, v3 = (features_for_tier(t) for t in ("v1", "v2", "v3"))
    assert len(v1) == 12
    assert len(v2) == 22
    assert len(v3) == 34 == len(FEATURE_NAMES)
    assert set(v1) < set(v2) < set(v3) == set(FEATURE_NAMES)
