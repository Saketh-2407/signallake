-- One row snapshotting the health of the silver layer each run: volumes, the
-- date range covered, and counts of the conditions the gates above check for.
select
    count(*) as total_events,
    count(distinct customer_id) as distinct_customers,
    min(event_date) as min_event_date,
    max(event_date) as max_event_date,
    sum(case when event_id is null then 1 else 0 end) as null_event_id_count,
    count(*) - count(distinct event_id) as duplicate_event_id_count,
    sum(case when amount is null then 1 else 0 end) as null_amount_count,
    sum(case when status = 'success' and amount < 0 then 1 else 0 end) as negative_success_amount_count,
    sum(case when timestamp > current_timestamp then 1 else 0 end) as future_dated_count,
    sum(case when label = 1 then 1 else 0 end) as anomaly_count,
    round(100.0 * sum(case when label = 1 then 1 else 0 end) / count(*), 3) as anomaly_rate_pct
from {{ ref('stg_silver') }}
