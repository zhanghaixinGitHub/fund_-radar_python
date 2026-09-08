"""内地沪深市场静态研究日历；来自官方休市事实，不从基金净值有无倒推交易日。"""

import hashlib
import json
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CALENDAR_FILE = Path(__file__).resolve().parents[1] / "data/calendars/cn_a_share_2021_2025_v1.json"
# 修改休市事实须新增日历版本并重新验收；不能同版本无声改日期。
EXPECTED_CALENDAR_HASH = "a7258f7368a071b9fd1df2b2ac039a60bb2be0b4e5c9bde51e836c76ca3c75fa"


class CalendarCoverageError(ValueError):
    """范围不足时拒绝，不自动用普通工作日或其他基金日期来填空。"""


class YearNotice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    year: int = Field(ge=2021, le=2025, description="本条官方安排覆盖的年度")
    announced_on: date = Field(description="年度安排公布日，不是本地抓取日")
    sources: tuple[str, str] = Field(description="沪深交易所原始公告链接；运行时不访问网页")
    closed_ranges: tuple[tuple[date, date], ...] = Field(
        min_length=1, max_length=10, description="含首尾的节假日休市区间"
    )

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.announced_on >= date(self.year, 1, 1):
            raise ValueError("annual notice must precede covered year")
        for start, end in self.closed_ranges:
            if not date(self.year - 1, 12, 30) <= start <= end <= date(self.year, 12, 31) or (end - start).days > 15:
                raise ValueError("invalid holiday range")
        return self


class CalendarDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["CN_A_SHARE_2021_2025_V1"] = Field(description="本次固定日历版本，修改事实需另出版本")
    market: Literal["CN_A_SHARE_SSE_SZSE"] = Field(description="适用内地沪深股票市场，不代表基金申赎日历")
    construction: Literal["WEEKDAYS_MINUS_OFFICIAL_HOLIDAYS"] = Field(description="周一至周五扣除官方节假日休市")
    coverage_start: date = Field(description="完整覆盖的首日，含当日")
    coverage_end: date = Field(description="完整覆盖的末日，含当日")
    reviewed_on: date = Field(description="本地核验日，不是当年公告首次公开日")
    years: tuple[YearNotice, ...] = Field(description="逐年官方公告与休市区间，必须覆盖全部五年")

    @model_validator(mode="after")
    def complete_years(self):
        if (self.coverage_start, self.coverage_end) != (date(2021, 1, 1), date(2025, 12, 31)) or tuple(
            y.year for y in self.years
        ) != (2021, 2022, 2023, 2024, 2025):
            raise ValueError("calendar coverage is incomplete")
        return self


@dataclass(frozen=True)
class TradingCalendar:
    definition: CalendarDefinition  # 规则、覆盖范围与公告来源；不是基金申赎日历。
    content_hash: str  # 整份规范化定义指纹；换行格式不影响，不是安全签名。
    sessions: tuple[date, ...]  # 全部开市日，严格递增，无普通调休周末。

    def at_or_before_index(self, day: date) -> int:
        if not self.definition.coverage_start <= day <= self.definition.coverage_end:
            raise CalendarCoverageError("日期超出已核验的2021–2025日历范围。")
        return bisect_right(self.sessions, day) - 1

    def future_sessions(self, cutoff: date, count: int = 20) -> tuple[date, ...]:
        if type(count) is not int or not 1 <= count <= 61:
            raise ValueError("session count must be 1..61")
        index = self.at_or_before_index(cutoff) + 1
        future = self.sessions[index : index + count]
        if len(future) != count:
            raise CalendarCoverageError("日历不能覆盖完整未来窗口，不用工作日猜测补齐。")
        return future


def build_calendar(definition: CalendarDefinition) -> TradingCalendar:
    closed = set()
    for notice in definition.years:
        for start, end in notice.closed_ranges:
            closed.update(start + timedelta(days=i) for i in range((end - start).days + 1))
    days = (
        definition.coverage_start + timedelta(days=i)
        for i in range((definition.coverage_end - definition.coverage_start).days + 1)
    )
    sessions = tuple(d for d in days if d.weekday() < 5 and d not in closed)
    content = json.dumps(definition.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return TradingCalendar(definition, hashlib.sha256(content.encode("utf-8")).hexdigest(), sessions)


@lru_cache(maxsize=1)
def load_calendar() -> TradingCalendar:
    """仅加载随应用发布的固定文件，最多64KiB；损坏即失败，不调用外部API兜底。"""
    with CALENDAR_FILE.open("rb") as handle:
        content = handle.read(65537)
    if len(content) > 65536:
        raise ValueError("calendar file exceeds 64KiB")
    calendar = build_calendar(CalendarDefinition.model_validate_json(content))
    if calendar.content_hash != EXPECTED_CALENDAR_HASH:
        raise ValueError("calendar definition changed without a new verified version")
    return calendar
