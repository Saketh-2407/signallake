-- Fails if any event is timestamped after "now" — a sign of a clock/generator bug
-- upstream, since the synthetic data is only ever backfilled, never forward-dated.
select event_id, timestamp
from {{ ref('stg_silver') }}
where timestamp > current_timestamp
