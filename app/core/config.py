"""Typed environment configuration for the FastAPI AI service."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_environment: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    ai_service_token: SecretStr = SecretStr("")
    ai_database_url: str = "postgresql+psycopg://fund_ai_app:change-me@localhost:54329/fund_ai"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide immutable configuration instance."""
    return Settings()
