"""多周期实验的日期、失败和收益契约；在线生成与历史回放共用。"""

import calendar as month_calendar
import hashlib
import json
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

POLICY_FILE = Path(__file__).resolve().parents[1] / "data/prediction_policy_v3.json"
_frozen_policy = ContextVar("prediction_frozen_policy", default=None)


def fingerprint(value) -> str:
    """采用稳定 JSON 编码，时间/小数由调用方明确为 ISO 字符串或十进制字符串。"""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str, allow_nan=False
        ).encode()
    ).hexdigest()


def prediction_policy() -> dict:
    """工作线程使用任务冻结规则；没有任务上下文时读取本进程启动规则。"""
    return _frozen_policy.get() or _configured_policy()


@lru_cache(maxsize=1)
def _configured_policy():
    return json.loads(POLICY_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def legacy_policy():
    return json.loads(POLICY_FILE.with_name("prediction_policy_v1.json").read_text(encoding="utf-8"))


@contextmanager
def policy_scope(policy):
    """ContextVar隔离并发线程，异常也复原；禁止修改共享全局配置恢复旧任务。"""
    token = _frozen_policy.set(policy)
    try:
        yield
    finally:
        _frozen_policy.reset(token)


def policy_for_record(record):
    """旧原文只认已知旧目标；新原文必须含完整规则，不用当前阈值补填历史。"""
    if record.get("targetDefinitionId") == legacy_policy()["target_definition_id"]:
        return legacy_policy()
    frozen = record.get("predictionPolicySnapshot")
    if not frozen or frozen.get("target_definition_id") != record.get("targetDefinitionId"):
        raise PredictionFailure("DIRECTION_POLICY_MISSING", "READ", "预测缺少原始方向规则", retryable=False)
    return frozen


class PredictionFailure(ValueError):
    """可保存并直接映射页面的业务失败；details 只允许脱敏字段。"""

    def __init__(
        self, code: str, stage: str, summary: str, *, details=None, retryable=True, next_action="补齐资料后重试"
    ):
        super().__init__(summary)
        self.payload = {
            "code": code,
            "stage": stage,
            "summary": summary,
            "details": details or {},
            "retryable": retryable,
            "nextAction": next_action,
        }


@dataclass(frozen=True)
class ValuationCalendar:
    """独立基金估值日历；来源和适用范围必须由适配器核验，不从净值有无倒推。"""

    calendar_id: str
    sessions: tuple[date, ...]
    coverage_start: date
    coverage_end: date
    source_hash: str
    timezone: str = "Asia/Shanghai"
    cutoff: time = time(15)

    def __post_init__(self):
        if not self.sessions or tuple(sorted(set(self.sessions))) != self.sessions:
            raise ValueError("CALENDAR_SESSIONS_INVALID")
        if self.sessions[0] < self.coverage_start or self.sessions[-1] > self.coverage_end:
            raise ValueError("CALENDAR_COVERAGE_INVALID")

    def missing(self, day: date):
        raise PredictionFailure(
            "CALENDAR_RANGE_MISSING",
            "CALENDAR",
            "基金估值日历未覆盖所需日期",
            details={
                "calendarId": self.calendar_id,
                "requestedDate": str(day),
                "coverageStart": str(self.coverage_start),
                "coverageEnd": str(self.coverage_end),
            },
        )

    def start(self, generated_at: datetime) -> date:
        if generated_at.tzinfo is None:
            raise ValueError("GENERATION_TIME_REQUIRES_TIMEZONE")
        local = generated_at.astimezone(ZoneInfo(self.timezone))
        day = local.date()
        if not self.coverage_start <= day <= self.coverage_end:
            self.missing(day)
        # 截止时刻及之后顺延；节假日即使上午点击，也只能使用下一估值日。
        index = (bisect_left if local.time() < self.cutoff else bisect_right)(self.sessions, day)
        if index >= len(self.sessions):
            self.missing(day)
        return self.sessions[index]

    def required_base(self, generated_at: datetime) -> date:
        """首个待预测估值日的前一估值日；收盘后必须取得刚结束当日的净值。"""
        start = self.start(generated_at)
        index = bisect_left(self.sessions, start)
        if index == 0:
            self.missing(start)
        return self.sessions[index - 1]


def nav_anchored(policy=None):
    """仅新规则改用已取得净值作为收益基准，旧档按冻结的原规则核验。"""
    return (policy or prediction_policy()).get("generation_policy") == "CN_NAV_READY_CLOSE_V1"


def add_months(day: date, months: int) -> date:
    """月末先夹到目标月最后一天，再由估值日历顺延；不换算为固定交易日数。"""
    index = day.year * 12 + day.month - 1 + months
    year, month = index // 12, index % 12 + 1
    return date(year, month, min(day.day, month_calendar.monthrange(year, month)[1]))


def target_dates(calendar: ValuationCalendar, generated_at: datetime, horizon_id: str, *, policy=None) -> dict:
    policy = policy or prediction_policy()
    horizon = next((h for h in policy["horizons"] if h["horizon_id"] == horizon_id), None)
    if horizon is None:
        raise PredictionFailure("HORIZON_NOT_CONFIGURED", "VALIDATION", "预测周期未开放", retryable=False)
    start = calendar.start(generated_at)
    base = calendar.required_base(generated_at) if nav_anchored(policy) else start
    nominal = None
    if horizon["unit"] == "TRADING_SESSION":
        index = bisect_left(calendar.sessions, base) + horizon["length"]
        if index >= len(calendar.sessions):
            calendar.missing(start)
        end = calendar.sessions[index]
    else:
        nominal = add_months(base, horizon["length"])
        index = bisect_left(calendar.sessions, nominal)
        end = calendar.sessions[index] if index < len(calendar.sessions) else None
    return {
        **({"baseNavDate": str(base), "generationPolicy": policy["generation_policy"]} if nav_anchored(policy) else {}),
        "startDate": str(start),
        "endDate": str(end) if end else None,
        "nominalEndDate": str(nominal) if nominal else None,
        "endDateStatus": "RESOLVED" if end else "PENDING_OFFICIAL_CALENDAR",
        "calendarId": calendar.calendar_id,
        "calendarHash": calendar.source_hash,
        "targetDefinitionId": policy["target_definition_id"],
        "horizonId": horizon_id,
    }


def reinvested_series(dates: list[date], nav: dict[date, Decimal], dividends: dict[date, Decimal]) -> list[Decimal]:
    """现金分红于除息日按单位净值再投；拆分由上游显式校验，不用累计净值替代。"""
    if not dates or dates != sorted(set(dates)):
        raise ValueError("NAV_DATES_INVALID")
    missing = [str(day) for day in dates if day not in nav]
    if missing:
        raise PredictionFailure("NAV_GAP", "FEATURE_BUILD", "必要估值日缺少净值", details={"missingDates": missing})
    if any(not nav[d].is_finite() or nav[d] <= 0 for d in dates):
        raise PredictionFailure("NAV_INVALID", "FEATURE_BUILD", "净值必须为有限正数")
    if any(not value.is_finite() or value < 0 for value in dividends.values()):
        raise PredictionFailure("EVENT_ADJUSTMENT_UNRESOLVED", "FEATURE_BUILD", "分红金额无法核验")
    result = [Decimal(1)]
    for previous, current in zip(dates[:-1], dates[1:], strict=True):
        result.append(result[-1] * (nav[current] + dividends.get(current, Decimal(0))) / nav[previous])
    return result
