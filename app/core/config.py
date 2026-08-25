"""FastAPI AI 服务的类型化环境配置。"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量或本地 `.env` 读取的运行时配置。

    服务令牌使用 SecretStr 保存，日志或接口中不得输出其实际值。
    """

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
    """返回进程内缓存的配置实例，避免每次请求重复读取环境变量。"""
    return Settings()
