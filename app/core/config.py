"""FastAPI AI 服务的类型化环境配置。"""

import re
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_TUSHARE_FOCUSED_TS_CODE_PATTERN = re.compile(r"^\d{6}\.(?:OF|SZ|SH)$")


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
    tushare_focused_nav_max_rows_per_query: int = Field(default=10_000, ge=1, le=100_000)
    tushare_focused_incremental_enabled: bool = True
    tushare_focused_incremental_hour: int = Field(default=20, ge=0, le=23)
    tushare_focused_incremental_minute: int = Field(default=30, ge=0, le=59)
    # 当前用户确认的六只基金。环境变量可用逗号分隔的完整 Tushare 代码覆盖，避免误触全市场目录同步。
    tushare_focused_fund_ts_codes: str = "010710.OF,160323.SZ,013275.OF,007832.OF,002112.OF,005312.OF"

    @property
    def focused_fund_ts_codes(self) -> tuple[str, ...]:
        """返回经过格式校验且去重的重点基金 Tushare 代码。

        Raises:
            ValueError: 配置为空、重复或不是受支持的基金代码格式时抛出。
        """
        codes = tuple(code.strip().upper() for code in self.tushare_focused_fund_ts_codes.split(",") if code.strip())
        if not codes:
            raise ValueError("Tushare focused fund code list must not be empty.")
        if len(set(codes)) != len(codes):
            raise ValueError("Tushare focused fund code list must not contain duplicates.")
        if any(_TUSHARE_FOCUSED_TS_CODE_PATTERN.fullmatch(code) is None for code in codes):
            raise ValueError("Tushare focused fund codes must use six digits plus .OF, .SZ, or .SH.")
        return codes


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的配置实例，避免每次请求重复读取环境变量。"""
    return Settings()
