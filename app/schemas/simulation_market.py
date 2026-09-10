"""模拟账本专用公共行情契约；不接收用户身份、金额、份额或交易。"""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class MarketModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class SimulationCalendar(MarketModel):
    version: str
    coverage_start: date
    coverage_end: date
    sessions: tuple[date, ...]
    assumption: str = "模拟采用沪深开市日，不代表基金实际申赎开放日；未模拟渠道限购和临时暂停。"


class SimulationNav(MarketModel):
    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None
    announced_on: date | None
    revision: str


class SimulationDividend(MarketModel):
    event_key: str
    record_date: date | None
    ex_date: date | None
    pay_date: date | None
    cash_per_unit: Decimal | None
    implemented: bool
    revision: str


class SimulationMarket(MarketModel):
    fund_code: str
    fund_name: str
    supported: bool
    reason: str | None
    source: str = "TUSHARE_PRO_FUND"
    navs: tuple[SimulationNav, ...]
    dividends: tuple[SimulationDividend, ...]
    dividends_verified_at: datetime | None
    refresh_status: str
    refresh_message: str | None


class SimulationRefresh(MarketModel):
    fund_codes: list[str] = Field(min_length=1, max_length=50)
