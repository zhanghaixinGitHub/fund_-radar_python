"""FastAPI AI 服务的类型化环境配置。"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量或本地 `.env` 读取的运行时配置。

    服务令牌和外部数据源凭据使用 SecretStr 保存，日志或接口中不得输出其实际值。
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
    tushare_token: SecretStr = SecretStr("")
    tushare_api_url: str = "https://api.tushare.pro"
    tushare_connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    tushare_read_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    tushare_max_retries: int = Field(default=2, ge=0, le=5)
    tushare_sync_batch_size: int = Field(default=500, ge=1, le=2_000)
    tushare_catalog_max_rows_per_query: int = Field(default=15_000, ge=1, le=100_000)
    tushare_market_nav_max_rows_per_query: int = Field(default=10_000, ge=1, le=100_000)
    tushare_market_reference_max_rows_per_query: int = Field(default=8_000, ge=1, le=100_000)
    # CSI 当前最小探测正好达到来源 8,000 行上限，默认不纳入目录同步，防止误写截断结果。
    tushare_index_catalog_markets: str = "SSE,SZSE,SW,CICC,MSCI,OTH"
    tushare_market_incremental_enabled: bool = True
    tushare_market_incremental_hour: int = Field(default=20, ge=0, le=23)
    tushare_market_incremental_minute: int = Field(default=0, ge=0, le=59)


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的配置实例，避免每次请求重复读取环境变量。"""
    return Settings()
