"""私人 DeepSeek 的使用边界：只向 002112 的当前一日综合预测开放凭据。"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT = Path(__file__).resolve().parents[2]


class PrivateDeepSeekSettings(BaseSettings):
    """专用配置与历史研究的 DEEPSEEK_API_KEY 隔离，避免旧脚本消耗私人余额。"""

    model_config = SettingsConfigDict(
        env_prefix="FUND_002112_DEEPSEEK_",
        env_file=PROJECT / ".env.deepseek",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    api_key: SecretStr = SecretStr("")
    # 默认关闭私人付费调用；保留历史配置和台账，不能因存在密钥而自动重新启用。
    enabled: bool = False
    model: Literal["deepseek-flash"] = "deepseek-flash"
    base_url: Literal["https://api.deepseek.com"] = "https://api.deepseek.com"
    # 按真实新增材料逐份调用，另保留综合和核验；失败不自动重发。
    max_calls_per_prediction: int = Field(default=20, ge=2, le=100)
    max_calls_per_day: int = Field(default=40, ge=2, le=200)
    # 请求前按字节上界保守预留费用，完成后按供应商 usage 结算；不是实际 token 计数。
    max_daily_cny: float = Field(default=2.0, gt=0, le=20)


@dataclass(frozen=True)
class PredictionScope:
    fund_code: str
    horizon_trading_days: int
    target_nav_date: str


_scope: ContextVar[PredictionScope | None] = ContextVar("private_deepseek_prediction_scope", default=None)


@contextmanager
def prediction_scope(fund_code: str, horizon_trading_days: int, target_nav_date: str) -> Iterator[None]:
    """仅正式当前一日推断入口创建范围；其他基金、周期和独立研究不能获得凭据。"""
    if fund_code != "002112" or horizon_trading_days != 1:
        raise ValueError("DEEPSEEK_SCOPE_DENIED")
    date.fromisoformat(target_nav_date)
    token = _scope.set(PredictionScope(fund_code, horizon_trading_days, target_nav_date))
    try:
        yield
    finally:
        _scope.reset(token)


def require_scope(stage: str, value: dict | None = None) -> PredictionScope:
    """在读密钥、查付费缓存和创建 HTTP 客户端之前检查用途及基金身份。"""
    scope = _scope.get()
    if (
        scope is None
        or scope.fund_code != "002112"
        or scope.horizon_trading_days != 1
        or stage not in {"EVENT", "SYNTHESIS", "AUDIT"}
    ):
        raise ValueError("DEEPSEEK_SCOPE_DENIED")
    value = value or {}
    if value.get("fund_code", scope.fund_code) != scope.fund_code:
        raise ValueError("DEEPSEEK_SCOPE_DENIED")
    target = value.get("target_nav_date") or value.get("window", {}).get("target_nav_date")
    if target is not None and target != scope.target_nav_date:
        raise ValueError("DEEPSEEK_SCOPE_DENIED")
    return scope


def provider_settings(stage: str, value: dict) -> PrivateDeepSeekSettings:
    """只有已授权的当前预测调用才能读取本机私有凭据；不输出密钥或完整配置。"""
    require_scope(stage, value)
    settings = PrivateDeepSeekSettings()
    if not settings.enabled:
        raise ValueError("DEEPSEEK_DISABLED")
    if not settings.api_key.get_secret_value():
        raise ValueError("ANALYSIS_NOT_CONFIGURED")
    return settings
