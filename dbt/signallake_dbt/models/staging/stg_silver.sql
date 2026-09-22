select
    event_id,
    customer_id,
    timestamp,
    amount,
    currency,
    merchant_category,
    location,
    device_id,
    is_new_device,
    status,
    label,
    anomaly_type,
    event_date
from {{ source('signallake', 'silver') }}
