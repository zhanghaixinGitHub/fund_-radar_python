"""FastAPI AI 服务的类型化环境配置。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量或本地 `.env` 读取的运行时配置。

    服务令牌和外部数据源凭据使用 SecretStr 保存，日志或接口中不得输出其实际值。
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_environment: str = "local"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    # 模型正式实验产物独立于研究临时目录；备份需同时包含此目录和模型登记表。
    prediction_model_directory: str = "data/prediction-models"
    # 页面只读快照由离线构建命令原子发布；原公告和完整财务资料继续保留在采集存储中。
    fund_material_directory: str = "data/fund-materials"
    # 公共风险和历史评价的不可变结果与当前引用；不保存本人账户资料。
    fund_insight_directory: str = "data/fund-insights"
    ai_service_token: SecretStr = SecretStr("")
    # 历史字段只保留兼容性。私人凭据由 deepseek_policy 按 002112/T1 范围单独读取。
    # 普通配置及历史研究脚本不再获得付费密钥，页面解释也不调用模型润色。
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_model: str = "deepseek-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_timeout_seconds: float = Field(default=20, ge=3, le=22)
    # 公告/新闻/政策的稳定原文解析库，与预测日期和模型选择隔离；必须随数据备份。
    fund_parse_cache_path: str = "data/fund-materials/parsed-documents.sqlite3"
    # 首次接入稳定资料库时只读核对原付费解析与输入快照；不匹配的旧结果不自动重收费。
    fund_parse_import_legacy: bool = True
    # 002112 一日判断默认由当前聊天离线分析、校验后导入；网页不自动调用聊天或付费模型。
    # 旧付费模式仅保留兼容入口，还必须通过独立凭据文件的 enabled 开关。
    fund_002112_analysis_mode: Literal["CODEX_MANUAL", "DEEPSEEK"] = "CODEX_MANUAL"
    # 费率后台任务回写 Java 的地址；令牌复用 AI_SERVICE_TOKEN，禁止从浏览器传入回调地址。
    core_service_base_url: str = "http://127.0.0.1:8080"
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
    # 兼容旧环境配置；自动计划已撤销，这些字段不再注册后台任务。
    tushare_market_incremental_enabled: bool = False
    tushare_market_incremental_hour: int = Field(default=20, ge=0, le=23)
    tushare_market_incremental_minute: int = Field(default=0, ge=0, le=59)

    @field_validator("fund_parse_cache_path")
    @classmethod
    def resolve_parse_cache_path(cls, value: str) -> str:
        """相对路径固定到Python项目目录，避免换启动目录后丢失旧解析或重置费用台账。"""
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else Path(__file__).resolve().parents[2] / path)


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的配置实例，避免每次请求重复读取环境变量。"""
    # 显式空值优先于旧环境变量：历史脚本即使仍读取该字段，也不能拿到私人密钥。
    return Settings(deepseek_api_key=SecretStr(""))
