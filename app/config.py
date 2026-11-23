"""Centralised app configuration, loaded from environment / .env."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://fpl:fpl@localhost:5432/fpl"
    sync_database_url: str = "postgresql+psycopg2://fpl:fpl@localhost:5432/fpl"

    redis_url: str = "redis://localhost:6379/0"

    fpl_base_url: str = "https://fantasy.premierleague.com/api"
    understat_base_url: str = "https://understat.com"

    model_dir: str = "./models"
    model_time_limit_seconds: int = 600

    log_level: str = "INFO"
    environment: str = "development"

    # Your own FPL team id (from fantasy.premierleague.com/entry/<id>) —
    # used as the default manager_id for /optimizer/transfers.
    default_manager_id: int | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
