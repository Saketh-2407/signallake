"""Central settings, read from environment variables / the project's .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/signallake/common/config.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kafka_broker: str = "localhost:9092"
    kafka_topic: str = "events"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"
    data_dir: Path = PROJECT_ROOT / "data"

    @field_validator("data_dir")
    @classmethod
    def _anchor_data_dir(cls, v: Path) -> Path:
        # "./data" must mean the repo's data/, regardless of the caller's cwd.
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"


@lru_cache
def get_settings() -> Settings:
    return Settings()
