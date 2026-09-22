from datetime import UTC, datetime

import numpy as np
import pyarrow.dataset as ds
import pytest

from signallake.common.schemas import EVENT_FIELDS, Event
from signallake.generate.producer import ParquetSink, to_arrow, verify_bronze
from signallake.generate.simulate import (
    DAY_MS,
    USUAL_DEVICES,
    GeneratorConfig,
    generate_events,
)

N = 30_000


@pytest.fixture(scope="module")
def table():
    return generate_events(GeneratorConfig(num_events=N, num_customers=300))


def test_deterministic_for_a_seed():
    cfg = GeneratorConfig(num_events=5_000, num_customers=100, seed=7)
    a, b = generate_events(cfg), generate_events(cfg)
    assert np.array_equal(a.amount, b.amount) and np.array_equal(a.timestamp_ms, b.timestamp_ms)
    other = generate_events(GeneratorConfig(num_events=5_000, num_customers=100, seed=8))
    assert not np.array_equal(a.amount, other.amount)


def test_size_label_rate_and_time_span(table):
    assert len(table) == N
    assert table.label.sum() == round(N * 0.03)  # label noise flips as many 1->0 as 0->1
    ts = table.timestamp_ms
    assert np.all(np.diff(ts) >= 0)
    start = table.config.start_ms
    assert ts[0] >= start and ts[-1] < start + 30 * DAY_MS
    bounds = table.day_bounds()
    assert len(bounds) == 31 and bounds[0] == 0 and bounds[-1] == N
    assert np.count_nonzero(np.diff(bounds)) == 30  # every one of the 30 days has events


def test_anomalies_skew_higher_but_overlap_normal(table):
    normal, anom = table.amount[table.label == 0], table.amount[table.label == 1]
    assert np.median(anom) > 1.5 * np.median(normal)
    assert np.mean(anom <= np.percentile(normal, 90)) > 0.2  # not cleanly separable
    assert table.failed[table.label == 1].mean() > 3 * table.failed[table.label == 0].mean()


def test_new_device_flag_is_first_sighting_of_an_unusual_device(table):
    unusual = table.device >= table.config.num_customers * USUAL_DEVICES
    assert not table.is_new_device[~unusual].any()
    flagged = table.device[table.is_new_device]
    assert len(flagged) == len(set(flagged.tolist()))  # each device flagged at most once
    assert set(flagged.tolist()) == set(table.device[unusual].tolist())  # ...and at least once


def test_decoded_rows_match_the_canonical_event_schema(table):
    cols = table.decode(0, 500)
    assert list(to_arrow(cols).schema.names) == EVENT_FIELDS
    assert len(set(cols["event_id"])) == 500
    for i in range(0, 500, 50):
        row = {k: v for k, v in cols.items() if k != "timestamp_ms"}
        row = {k: (v[i].item() if hasattr(v[i], "item") else v[i]) for k, v in row.items()}
        row["timestamp"] = datetime.fromtimestamp(cols["timestamp_ms"][i] / 1000, tz=UTC)
        Event(**row)  # validates enums, ranges and types


def test_signal_strength_and_noise_knobs_move_separability():
    def separation(**kw):
        t = generate_events(GeneratorConfig(num_events=N, num_customers=300, **kw))
        return np.median(t.amount[t.label == 1]) / np.median(t.amount[t.label == 0])

    assert separation(signal_strength=2.0) > separation(signal_strength=0.5)


def test_bronze_parquet_is_partitioned_by_event_date(tmp_path):
    t = generate_events(GeneratorConfig(num_events=5_000, num_customers=100))
    sink = ParquetSink(tmp_path, t.config.start_date)
    bounds = t.day_bounds()
    for day in range(30):
        sink.write_day(day, to_arrow(t.decode(int(bounds[day]), int(bounds[day + 1]))))
    rows, partitions, first, last = verify_bronze(tmp_path, 5_000)
    assert (rows, partitions, first, last) == (5_000, 30, "2026-08-01", "2026-08-30")
    schema = ds.dataset(tmp_path, format="parquet", partitioning="hive").schema
    assert str(schema.field("timestamp").type) == "timestamp[ms, tz=UTC]"
    # Writing again replaces partitions rather than duplicating rows.
    ParquetSink(tmp_path, t.config.start_date)
    assert not list(tmp_path.glob("event_date=*"))
