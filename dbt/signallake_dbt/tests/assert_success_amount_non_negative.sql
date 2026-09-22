-- A successful transaction can't have moved a negative amount. Failed attempts
-- are exempt: a declined/reversed transaction may legitimately carry a negative
-- or zero amount in this model.
select event_id, status, amount
from {{ ref('stg_silver') }}
where status = 'success'
  and amount < 0
