"""Vectorised synthetic transaction simulator with tunable, deliberately imperfect anomalies.

Everything is generated with numpy and returned as a column-oriented, timestamp-sorted
`EventTable` of integer codes; `producer.py` decodes slices of it into parquet / Kafka messages.

Anomalies are injected as *behaviour* the Phase 2 features can observe (bursts, amount outliers,
a new device in a far-away city, runs of failures), never as a leaked flag. Separability is
controlled by two knobs, and it is never perfect:

* ``signal_strength`` scales burst sizes, amount multipliers and the odds of a device/geo jump.
  ~15% of anomaly episodes are "subtle" (signal scaled down), and a little label noise is added.
* ``noise`` widens the spread of normal amounts and scales the rate of *hard negatives* -- normal
  behaviour that looks suspicious: shopping sprees, retry streaks, big purchases, trips to
  another region and one-off new devices.

Every anomaly episode labels *all* of its events 1, so the first event of a burst (which has no
history yet) is intrinsically hard to catch -- another natural ceiling on recall.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

from signallake.common.schemas import AnomalyType, MerchantCategory

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS

# Hard-negative rates below are calibrated at this noise level and scale linearly with `noise`.
REFERENCE_NOISE = 0.35

# ---- Geography ----------------------------------------------------------------------------
REGIONS: dict[str, list[str]] = {
    "us_east": ["New York", "Boston", "Miami", "Atlanta", "Washington"],
    "us_west": ["San Francisco", "Los Angeles", "Seattle", "Denver", "Phoenix"],
    "europe": ["London", "Paris", "Berlin", "Madrid", "Amsterdam"],
    "asia": ["Tokyo", "Singapore", "Mumbai", "Seoul", "Hong Kong"],
}
CITY_NAMES: list[str] = [city for cities in REGIONS.values() for city in cities]
REGION_SIZE = np.array([len(c) for c in REGIONS.values()])
REGION_START = np.concatenate([[0], np.cumsum(REGION_SIZE)[:-1]])
CITY_REGION = np.repeat(np.arange(len(REGIONS)), REGION_SIZE)
REGION_P = np.array([0.30, 0.25, 0.25, 0.20])

# ---- Merchants: (share of normal traffic, amount multiplier, share of anomalous traffic) ----
_CATEGORY_PROFILE: dict[MerchantCategory, tuple[float, float, float]] = {
    MerchantCategory.GROCERY: (0.22, 1.0, 0.04),
    MerchantCategory.RESTAURANT: (0.18, 0.6, 0.05),
    MerchantCategory.FUEL: (0.10, 0.7, 0.04),
    MerchantCategory.UTILITIES: (0.06, 1.6, 0.03),
    MerchantCategory.HEALTH: (0.06, 1.3, 0.03),
    MerchantCategory.ENTERTAINMENT: (0.10, 0.8, 0.09),
    MerchantCategory.FASHION: (0.10, 1.2, 0.16),
    MerchantCategory.ONLINE_SERVICES: (0.08, 0.7, 0.20),
    MerchantCategory.ELECTRONICS: (0.06, 2.5, 0.22),
    MerchantCategory.TRAVEL: (0.04, 3.0, 0.14),
}
CATEGORY_NAMES: list[str] = [c.value for c in MerchantCategory]
CATEGORY_P = np.array([_CATEGORY_PROFILE[c][0] for c in MerchantCategory])
CATEGORY_AMOUNT_MULT = np.array([_CATEGORY_PROFILE[c][1] for c in MerchantCategory])
ANOMALY_CATEGORY_P = np.array([_CATEGORY_PROFILE[c][2] for c in MerchantCategory])

# Relative traffic by hour of day (UTC): quiet nights, busy evenings.
HOUR_P = np.array(
    [0.3, 0.2, 0.15, 0.15, 0.2, 0.4, 0.9, 1.5, 2.0, 2.2, 2.3, 2.4]
    + [2.6, 2.4, 2.2, 2.2, 2.3, 2.5, 2.7, 2.6, 2.3, 1.8, 1.2, 0.7]
)
HOUR_P = HOUR_P / HOUR_P.sum()
WEEKEND_BOOST = 0.15

# ---- Anomalies ----------------------------------------------------------------------------
ANOMALY_NAMES: list[str | None] = [None, *(t.value for t in AnomalyType)]  # code 0 = normal
ANOMALY_CODE = {t: i + 1 for i, t in enumerate(AnomalyType)}
# Share of anomalous *events* per type.
ANOMALY_MIX = {
    AnomalyType.VELOCITY_SPIKE: 0.30,
    AnomalyType.AMOUNT_SPIKE: 0.28,
    AnomalyType.NEW_DEVICE_GEO_JUMP: 0.20,
    AnomalyType.FAILED_BURST: 0.22,
}
SUBTLE_FACTOR = 0.5  # signal multiplier for "subtle" episodes
ANOMALY_NIGHT_P = 0.25  # extra odds an anomaly episode starts between 00:00 and 06:00
# Episode shape at signal_strength 1.0 (scaled by each episode's own strength).
VELOCITY_SIZE = (4, 5, 12)  # min txns, plus uniform extra range
FAILED_BURST_SIZE = (3, 3, 9)
AMOUNT_SPIKE_MULT = (2.0, 14.0)  # median multiplier = base + slope * strength
BURST_AMOUNT_MULT = (1.8, 4.0)  # uniform multiplier on each event of a velocity/failed burst
GEO_JUMP_ODDS = (0.35, 0.40)  # P(new device) and P(far city) = base + slope * min(strength, 1)

# ---- Normal behaviour ---------------------------------------------------------------------
USUAL_DEVICES = 3  # device code = customer * USUAL_DEVICES + slot
BASE_FAILURE_RATE = 0.03
AMOUNT_SIGMA = (0.2, 0.6)  # lognormal spread of a customer's amounts = base + slope * noise
MEDIAN_AMOUNT = 45.0
# Hard-negative rates at REFERENCE_NOISE.
BIG_PURCHASE_RATE = 0.002  # fraction of normal events with a 2.5-6x amount
LEGIT_NEW_DEVICE_RATE = 0.004  # fraction of normal events on a one-off new device
SPREE_SHARE = 0.004  # share of normal events that sit in a 2-5 txn shopping spree
RETRY_SHARE = 0.002  # share of normal events that sit in a 2-4 txn fail-then-succeed streak
TRIP_CUSTOMER_SHARE = 0.08  # share of customers with one multi-day trip to another region


@dataclass(frozen=True)
class GeneratorConfig:
    num_events: int = 2_100_000
    num_customers: int = 2_000
    anomaly_rate: float = 0.03
    signal_strength: float = 1.0
    noise: float = REFERENCE_NOISE
    subtle_fraction: float = 0.15
    label_noise: float = 0.02  # fraction of anomalies dropped to 0 (and as many normals set to 1)
    num_days: int = 30
    start_date: date = date(2026, 8, 1)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_events < 1000:
            raise ValueError("num_events must be >= 1000")
        if not 0 < self.anomaly_rate < 0.5:
            raise ValueError("anomaly_rate must be in (0, 0.5)")
        if self.signal_strength <= 0 or self.noise < 0:
            raise ValueError("signal_strength must be > 0 and noise >= 0")
        if not 0 <= self.subtle_fraction <= 1 or not 0 <= self.label_noise < 1:
            raise ValueError("subtle_fraction and label_noise must be fractions")
        if self.num_days < 8 or self.num_customers < 1:
            raise ValueError("need num_days >= 8 and num_customers >= 1")

    @property
    def start_ms(self) -> int:
        midnight = datetime(*self.start_date.timetuple()[:3], tzinfo=UTC)
        return int(midnight.timestamp()) * 1000

    @property
    def hard_negative_scale(self) -> float:
        return self.noise / REFERENCE_NOISE


@dataclass
class EventTable:
    """Timestamp-sorted events as integer codes plus the lookup tables that decode them."""

    config: GeneratorConfig
    timestamp_ms: np.ndarray
    customer: np.ndarray
    amount: np.ndarray
    category: np.ndarray
    city: np.ndarray
    device: np.ndarray
    is_new_device: np.ndarray
    failed: np.ndarray
    label: np.ndarray
    anomaly: np.ndarray  # code into ANOMALY_NAMES; 0 when normal *or* the label was noised
    customer_names: np.ndarray
    device_names: np.ndarray

    def __len__(self) -> int:
        return len(self.timestamp_ms)

    def day_bounds(self) -> np.ndarray:
        """Row offsets of each event_date partition: day d is rows [bounds[d], bounds[d + 1])."""
        edges = self.config.start_ms + np.arange(self.config.num_days + 1) * DAY_MS
        return np.searchsorted(self.timestamp_ms, edges, side="left")

    def decode(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        """Rows [lo, hi) as canonical Event columns (see `signallake.common.schemas.Event`)."""
        s = slice(lo, hi)
        n = hi - lo
        return {
            "event_id": np.array([f"evt_{i:09d}" for i in range(lo, hi)], dtype=object),
            "customer_id": self.customer_names[self.customer[s]],
            "timestamp_ms": self.timestamp_ms[s],
            "amount": self.amount[s],
            "currency": np.full(n, "USD", dtype=object),
            "merchant_category": np.array(CATEGORY_NAMES, dtype=object)[self.category[s]],
            "location": np.array(CITY_NAMES, dtype=object)[self.city[s]],
            "device_id": self.device_names[self.device[s]],
            "is_new_device": self.is_new_device[s],
            "status": np.where(self.failed[s], "failed", "success").astype(object),
            "label": self.label[s],
            "anomaly_type": np.array(ANOMALY_NAMES, dtype=object)[self.anomaly[s]],
        }


# ---- Building blocks ----------------------------------------------------------------------
@dataclass(frozen=True)
class _Timeline:
    start_ms: int
    num_days: int
    day_p: np.ndarray

    @classmethod
    def from_config(cls, cfg: GeneratorConfig) -> "_Timeline":
        days = [cfg.start_date + timedelta(days=d) for d in range(cfg.num_days)]
        weight = np.array([1.0 + WEEKEND_BOOST * (d.weekday() >= 5) for d in days])
        return cls(cfg.start_ms, cfg.num_days, weight / weight.sum())

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.num_days * DAY_MS

    def sample(self, rng: np.random.Generator, n: int, night_p: float = 0.0) -> np.ndarray:
        day = rng.choice(self.num_days, n, p=self.day_p).astype(np.int64)
        hour = rng.choice(24, n, p=HOUR_P).astype(np.int64)
        if night_p:
            hour = np.where(rng.random(n) < night_p, rng.integers(0, 6, n), hour)
        return self.start_ms + day * DAY_MS + hour * HOUR_MS + rng.integers(0, HOUR_MS, n)


@dataclass(frozen=True)
class _Customers:
    home_city: np.ndarray
    log_median_amount: np.ndarray
    sigma: np.ndarray  # spread of this customer's log-amounts
    volume_p: np.ndarray  # share of normal traffic
    num_devices: np.ndarray
    trip_start_ms: np.ndarray  # start == end == 0 when the customer never travels
    trip_end_ms: np.ndarray
    trip_city: np.ndarray


class _DeviceAllocator:
    """Hands out device codes: usual devices are `customer * 3 + slot`, new ones are unique."""

    def __init__(self, num_customers: int) -> None:
        self._next = num_customers * USUAL_DEVICES
        self._owners = [np.repeat(np.arange(num_customers, dtype=np.int32), USUAL_DEVICES)]

    def allocate(self, owners: np.ndarray) -> np.ndarray:
        codes = np.arange(self._next, self._next + len(owners), dtype=np.int32)
        self._next += len(owners)
        self._owners.append(owners.astype(np.int32))
        return codes

    def names(self) -> np.ndarray:
        owners = np.concatenate(self._owners)
        codes = np.arange(len(owners))
        return np.array(
            [f"dev_{o:05d}_{c:06d}" for o, c in zip(owners, codes, strict=True)], dtype=object
        )


@dataclass
class _Block:
    """Columns for one group of events (base traffic, or one episode family)."""

    timestamp_ms: np.ndarray
    customer: np.ndarray
    amount: np.ndarray
    category: np.ndarray
    city: np.ndarray
    device: np.ndarray
    failed: np.ndarray
    anomaly: np.ndarray


def _far_city(rng: np.random.Generator, home: np.ndarray) -> np.ndarray:
    region = (CITY_REGION[home] + rng.integers(1, len(REGIONS), len(home))) % len(REGIONS)
    return REGION_START[region] + rng.integers(0, REGION_SIZE[region])


def _near_city(rng: np.random.Generator, home: np.ndarray) -> np.ndarray:
    region = CITY_REGION[home]
    offset = rng.integers(1, REGION_SIZE[region])
    return REGION_START[region] + (home - REGION_START[region] + offset) % REGION_SIZE[region]


def _make_customers(
    rng: np.random.Generator, cfg: GeneratorConfig, timeline: _Timeline
) -> _Customers:
    n = cfg.num_customers
    region = rng.choice(len(REGIONS), n, p=REGION_P)
    home = REGION_START[region] + rng.integers(0, REGION_SIZE[region])
    volume = rng.lognormal(0.0, 0.5, n)

    trips = rng.random(n) < min(0.5, TRIP_CUSTOMER_SHARE * cfg.hard_negative_scale)
    start = timeline.start_ms + rng.integers(2, cfg.num_days - 5, n) * DAY_MS
    end = start + rng.integers(2, 6, n) * DAY_MS
    return _Customers(
        home_city=home.astype(np.int16),
        log_median_amount=rng.normal(np.log(MEDIAN_AMOUNT), 0.6, n),
        sigma=(AMOUNT_SIGMA[0] + AMOUNT_SIGMA[1] * cfg.noise) * rng.uniform(0.75, 1.25, n),
        volume_p=volume / volume.sum(),
        num_devices=rng.choice([1, 2, 3], n, p=[0.45, 0.40, 0.15]),
        trip_start_ms=np.where(trips, start, 0),
        trip_end_ms=np.where(trips, end, 0),
        trip_city=np.where(trips, _far_city(rng, home), home).astype(np.int16),
    )


def _amount(
    rng: np.random.Generator, customers: _Customers, customer: np.ndarray, category: np.ndarray
) -> np.ndarray:
    z = rng.standard_normal(len(customer))
    log_amount = customers.log_median_amount[customer] + customers.sigma[customer] * z
    return np.exp(log_amount) * CATEGORY_AMOUNT_MULT[category]


def _usual_device(
    rng: np.random.Generator, customers: _Customers, customer: np.ndarray
) -> np.ndarray:
    u = rng.random(len(customer))
    slot = np.where(u < 0.7, 0, np.where(u < 0.9, 1, 2))
    slot = np.minimum(slot, customers.num_devices[customer] - 1)
    return (customer * USUAL_DEVICES + slot).astype(np.int32)


def _city_at(customers: _Customers, customer: np.ndarray, ts: np.ndarray) -> np.ndarray:
    on_trip = (ts >= customers.trip_start_ms[customer]) & (ts < customers.trip_end_ms[customer])
    return np.where(on_trip, customers.trip_city[customer], customers.home_city[customer]).astype(
        np.int16
    )


def _fit_budget(sizes: np.ndarray, budget: int) -> np.ndarray:
    """Keep leading episode sizes summing to <= budget, plus one short episode for the rest."""
    keep = int(np.searchsorted(np.cumsum(sizes), budget, side="right"))
    sizes = sizes[:keep]
    rest = budget - int(sizes.sum())
    return np.append(sizes, rest) if rest else sizes


def _expand(sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per event: its episode index and its position within the episode."""
    episode = np.repeat(np.arange(len(sizes)), sizes)
    first = np.repeat(np.cumsum(sizes) - sizes, sizes)
    return episode, np.arange(len(episode)) - first


def _strength(rng: np.random.Generator, cfg: GeneratorConfig, n: int) -> np.ndarray:
    """Per-episode signal strength; a `subtle_fraction` of episodes is much weaker."""
    subtle = rng.random(n) < cfg.subtle_fraction
    return cfg.signal_strength * np.where(subtle, SUBTLE_FACTOR, 1.0)


def _episode_block(
    rng: np.random.Generator,
    ctx: "_Context",
    sizes: np.ndarray,
    span_ms: np.ndarray,
    anomaly: int,
    *,
    night_p: float = 0.0,
    category_p: np.ndarray = CATEGORY_P,
) -> tuple[_Block, np.ndarray, np.ndarray, np.ndarray]:
    """Shared skeleton of a burst: who, when, which merchant, and a normal-looking amount.

    Returns the block plus (episode index, position in episode, per-episode customer) so the
    caller can override amounts, status or devices for its specific behaviour.
    """
    cfg, customers = ctx.cfg, ctx.customers
    episodes = len(sizes)
    episode, pos = _expand(sizes)
    customer_e = rng.integers(0, cfg.num_customers, episodes).astype(np.int32)
    start_e = ctx.timeline.sample(rng, episodes, night_p)
    category_e = rng.choice(len(category_p), episodes, p=category_p / category_p.sum())

    offset = (rng.random(len(episode)) * span_ms[episode]).astype(np.int64)
    offset[pos == 0] = 0
    ts = np.minimum(start_e[episode] + offset, ctx.timeline.end_ms - 1)
    customer = customer_e[episode]
    category = category_e[episode].astype(np.int8)
    block = _Block(
        timestamp_ms=ts,
        customer=customer,
        amount=_amount(rng, customers, customer, category),
        category=category,
        city=_city_at(customers, customer, ts),
        device=_usual_device(rng, customers, customer),
        failed=rng.random(len(episode)) < BASE_FAILURE_RATE,
        anomaly=np.full(len(episode), anomaly, dtype=np.int8),
    )
    return block, episode, pos, customer_e


@dataclass(frozen=True)
class _Context:
    cfg: GeneratorConfig
    customers: _Customers
    timeline: _Timeline
    devices: _DeviceAllocator


# ---- Normal traffic -----------------------------------------------------------------------
def _base_events(rng: np.random.Generator, ctx: _Context, n: int) -> _Block:
    cfg, customers = ctx.cfg, ctx.customers
    h = cfg.hard_negative_scale
    customer = rng.choice(cfg.num_customers, n, p=customers.volume_p).astype(np.int32)
    ts = ctx.timeline.sample(rng, n)
    category = rng.choice(len(CATEGORY_P), n, p=CATEGORY_P).astype(np.int8)
    amount = _amount(rng, customers, customer, category)
    device = _usual_device(rng, customers, customer)

    # Hard negative: a big-but-legitimate purchase.
    big = rng.random(n) < BIG_PURCHASE_RATE * h
    amount[big] *= np.exp(rng.uniform(np.log(2.5), np.log(6.0), big.sum()))
    # Hard negative: a one-off new device (new phone, borrowed laptop).
    fresh = rng.random(n) < LEGIT_NEW_DEVICE_RATE * h
    device[fresh] = ctx.devices.allocate(customer[fresh])

    return _Block(
        timestamp_ms=ts,
        customer=customer,
        amount=amount,
        category=category,
        city=_city_at(customers, customer, ts),  # trips are a hard negative for geo features
        device=device,
        failed=rng.random(n) < BASE_FAILURE_RATE,
        anomaly=np.zeros(n, dtype=np.int8),
    )


def _spree_events(rng: np.random.Generator, ctx: _Context, budget: int) -> _Block | None:
    """Hard negative: 2-5 quick purchases in a row."""
    if budget <= 0:
        return None
    sizes = _fit_budget(rng.integers(2, 6, budget + 1), budget)
    span = rng.uniform(1, 8, len(sizes)) * MINUTE_MS
    block, *_ = _episode_block(rng, ctx, sizes, span, anomaly=0)
    return block


def _retry_events(rng: np.random.Generator, ctx: _Context, budget: int) -> _Block | None:
    """Hard negative: a payment that fails once or twice, then goes through, for the same amount."""
    if budget <= 0:
        return None
    sizes = _fit_budget(rng.integers(2, 5, budget + 1), budget)
    span = rng.uniform(0.5, 3, len(sizes)) * MINUTE_MS
    block, episode, pos, _ = _episode_block(rng, ctx, sizes, span, anomaly=0)
    first = np.cumsum(sizes) - sizes
    block.amount = block.amount[first][episode]
    block.failed = pos < (sizes[episode] - 1)
    return block


# ---- Anomalies ----------------------------------------------------------------------------
def _velocity_spike(rng: np.random.Generator, ctx: _Context, budget: int) -> _Block | None:
    """Many transactions within minutes; subtle episodes are smaller and more spread out."""
    if budget <= 0:
        return None
    strength = _strength(rng, ctx.cfg, budget + 1)
    lo, a, b = VELOCITY_SIZE
    sizes = _fit_budget(lo + np.rint(rng.uniform(a, b, budget + 1) * strength).astype(int), budget)
    strength = strength[: len(sizes)]
    span = rng.uniform(1, 4, len(sizes)) / np.maximum(strength, 0.3) * MINUTE_MS
    block, episode, *_ = _episode_block(
        rng,
        ctx,
        sizes,
        span,
        ANOMALY_CODE[AnomalyType.VELOCITY_SPIKE],
        night_p=ANOMALY_NIGHT_P,
        category_p=ANOMALY_CATEGORY_P,
    )
    block.amount = block.amount * rng.uniform(*BURST_AMOUNT_MULT, len(episode))
    return block


def _failed_burst(rng: np.random.Generator, ctx: _Context, budget: int) -> _Block | None:
    """A run of declined attempts (card testing); subtle episodes fail less consistently."""
    if budget <= 0:
        return None
    strength = _strength(rng, ctx.cfg, budget + 1)
    lo, a, b = FAILED_BURST_SIZE
    sizes = _fit_budget(lo + np.rint(rng.uniform(a, b, budget + 1) * strength).astype(int), budget)
    strength = strength[: len(sizes)]
    span = rng.uniform(2, 8, len(sizes)) / np.maximum(strength, 0.3) * MINUTE_MS
    block, episode, *_ = _episode_block(
        rng,
        ctx,
        sizes,
        span,
        ANOMALY_CODE[AnomalyType.FAILED_BURST],
        night_p=ANOMALY_NIGHT_P,
        category_p=ANOMALY_CATEGORY_P,
    )
    fail_p = np.clip(0.6 + 0.4 * strength, 0, 1)[episode]
    block.failed = rng.random(len(episode)) < fail_p
    block.amount = block.amount * rng.uniform(*BURST_AMOUNT_MULT, len(episode))
    return block


def _amount_spike(rng: np.random.Generator, ctx: _Context, budget: int) -> _Block | None:
    """One (sometimes two) purchases far above the customer's baseline."""
    if budget <= 0:
        return None
    strength = _strength(rng, ctx.cfg, budget + 1)
    sizes = _fit_budget(1 + (rng.random(budget + 1) < 0.25).astype(int), budget)
    strength = strength[: len(sizes)]
    span = rng.uniform(1, 10, len(sizes)) * MINUTE_MS
    block, episode, *_ = _episode_block(
        rng,
        ctx,
        sizes,
        span,
        ANOMALY_CODE[AnomalyType.AMOUNT_SPIKE],
        night_p=ANOMALY_NIGHT_P,
        category_p=ANOMALY_CATEGORY_P,
    )
    median_multiplier = AMOUNT_SPIKE_MULT[0] + AMOUNT_SPIKE_MULT[1] * strength[episode]
    jitter = 0.2 + 0.4 * ctx.cfg.noise
    block.amount = (
        block.amount * median_multiplier * np.exp(jitter * rng.standard_normal(len(episode)))
    )
    return block


def _new_device_geo_jump(rng: np.random.Generator, ctx: _Context, budget: int) -> _Block | None:
    """1-3 purchases from a device never seen before, in another city (often another region)."""
    if budget <= 0:
        return None
    strength = _strength(rng, ctx.cfg, budget + 1)
    sizes = _fit_budget(1 + np.rint(rng.uniform(0, 2, budget + 1) * strength).astype(int), budget)
    strength = strength[: len(sizes)]
    episodes = len(sizes)
    span = rng.uniform(1, 6, episodes) * MINUTE_MS
    block, episode, _, customer_e = _episode_block(
        rng,
        ctx,
        sizes,
        span,
        ANOMALY_CODE[AnomalyType.NEW_DEVICE_GEO_JUMP],
        night_p=ANOMALY_NIGHT_P,
        category_p=ANOMALY_CATEGORY_P,
    )
    odds = GEO_JUMP_ODDS[0] + GEO_JUMP_ODDS[1] * np.minimum(strength, 1.0)
    new_device_e = rng.random(episodes) < odds
    far_e = rng.random(episodes) < odds
    home_e = ctx.customers.home_city[customer_e]
    city_e = np.where(far_e, _far_city(rng, home_e), _near_city(rng, home_e))
    device_e = _usual_device(rng, ctx.customers, customer_e)
    device_e[new_device_e] = ctx.devices.allocate(customer_e[new_device_e])

    block.city = city_e[episode].astype(np.int16)
    block.device = device_e[episode]
    block.amount = block.amount * (1 + rng.uniform(0.2, 1.5, episodes) * strength)[episode]
    return block


# ---- Assembly -----------------------------------------------------------------------------
def _concat(blocks: list[_Block]) -> _Block:
    return _Block(
        **{f: np.concatenate([getattr(b, f) for b in blocks]) for f in _Block.__annotations__}
    )


def generate_events(cfg: GeneratorConfig) -> EventTable:
    """Simulate `cfg.num_events` events over `cfg.num_days` days. Deterministic for a given seed."""
    rng = np.random.default_rng(cfg.seed)
    timeline = _Timeline.from_config(cfg)
    ctx = _Context(
        cfg, _make_customers(rng, cfg, timeline), timeline, _DeviceAllocator(cfg.num_customers)
    )

    n_anomalous = round(cfg.num_events * cfg.anomaly_rate)
    n_normal = cfg.num_events - n_anomalous
    h = cfg.hard_negative_scale
    n_spree = min(int(n_normal * SPREE_SHARE * h), n_normal // 5)
    n_retry = min(int(n_normal * RETRY_SHARE * h), n_normal // 10)

    budgets = {t: round(n_anomalous * share) for t, share in ANOMALY_MIX.items()}
    budgets[AnomalyType.AMOUNT_SPIKE] += n_anomalous - sum(budgets.values())
    blocks = [
        _base_events(rng, ctx, n_normal - n_spree - n_retry),
        _spree_events(rng, ctx, n_spree),
        _retry_events(rng, ctx, n_retry),
        _velocity_spike(rng, ctx, budgets[AnomalyType.VELOCITY_SPIKE]),
        _failed_burst(rng, ctx, budgets[AnomalyType.FAILED_BURST]),
        _amount_spike(rng, ctx, budgets[AnomalyType.AMOUNT_SPIKE]),
        _new_device_geo_jump(rng, ctx, budgets[AnomalyType.NEW_DEVICE_GEO_JUMP]),
    ]
    all_events = _concat([b for b in blocks if b is not None])

    order = np.argsort(all_events.timestamp_ms, kind="stable")
    ev = _Block(**{f: getattr(all_events, f)[order] for f in _Block.__annotations__})

    label = (ev.anomaly > 0).astype(np.int8)
    n_flip = round(cfg.label_noise * int(label.sum()))
    if n_flip:
        dropped = rng.choice(np.flatnonzero(label == 1), n_flip, replace=False)
        added = rng.choice(np.flatnonzero(label == 0), n_flip, replace=False)
        label[dropped], label[added] = 0, 1
        ev.anomaly[dropped] = 0  # anomaly_type is None whenever the label was noised

    # A device is "new" the first time we see it, unless it is one of the customer's usual ones.
    is_new = np.zeros(len(ev.device), dtype=bool)
    unusual = np.flatnonzero(ev.device >= cfg.num_customers * USUAL_DEVICES)
    _, first = np.unique(ev.device[unusual], return_index=True)
    is_new[unusual[first]] = True

    return EventTable(
        config=cfg,
        timestamp_ms=ev.timestamp_ms,
        customer=ev.customer,
        amount=np.round(np.maximum(ev.amount, 0.5), 2),
        category=ev.category,
        city=ev.city,
        device=ev.device,
        is_new_device=is_new,
        failed=ev.failed,
        label=label,
        anomaly=ev.anomaly,
        customer_names=np.array([f"cust_{i:05d}" for i in range(cfg.num_customers)], dtype=object),
        device_names=ctx.devices.names(),
    )
