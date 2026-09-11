"""持仓建议回看只接收公共基金与观察区间，不接收任何个人持仓。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from app.schemas.simulation_market import MarketModel


class AdviceOutcome(MarketModel):
    fund_code: str
    start_date: date
    end_date: date
    status: Literal["WAITING", "DATA_INSUFFICIENT", "ASSESSED"]
    checked_at: datetime
    total_return: Decimal | None = Field(default=None, description="现金分红再投资回报，小数比例；不是实际交易盈亏")
    message: str
    basis: str = "CASH_REINVESTMENT_20D_V1"
    source: str = "TUSHARE_PRO_FUND"
    evidence_hash: str | None = None
    # 固定21个日期与净值/分红审计，便于核验完成后保存原始计算输入。
    evidence: dict | None = None
