"""Canonical event schema shared by the generator, Spark jobs, dbt tests and serving.

The enums below are the single source of truth for the allowed values; the dbt
`accepted_values` tests (Phase 3) should mirror them.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Status(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class MerchantCategory(StrEnum):
    GROCERY = "grocery"
    RESTAURANT = "restaurant"
    TRAVEL = "travel"
    ELECTRONICS = "electronics"
    FASHION = "fashion"
    FUEL = "fuel"
    ENTERTAINMENT = "entertainment"
    UTILITIES = "utilities"
    HEALTH = "health"
    ONLINE_SERVICES = "online_services"


class AnomalyType(StrEnum):
    VELOCITY_SPIKE = "velocity_spike"
    AMOUNT_SPIKE = "amount_spike"
    NEW_DEVICE_GEO_JUMP = "new_device_geo_jump"
    FAILED_BURST = "failed_burst"


class Event(BaseModel):
    """One transaction-like event. `label` / `anomaly_type` are synthetic ground truth."""

    model_config = ConfigDict(use_enum_values=True)

    event_id: str
    customer_id: str
    timestamp: datetime
    amount: float = Field(ge=0)
    currency: str = "USD"
    merchant_category: MerchantCategory
    location: str  # city
    device_id: str
    is_new_device: bool
    status: Status
    label: int = Field(ge=0, le=1, description="0 = normal, 1 = anomaly (ground truth)")
    anomaly_type: AnomalyType | None = None  # None when label == 0 (or label was noised)


# Column order for parquet / Kafka payloads.
EVENT_FIELDS: list[str] = list(Event.model_fields)
