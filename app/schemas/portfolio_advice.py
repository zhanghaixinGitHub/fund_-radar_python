"""持仓建议回看只接收公共基金与观察区间，不接收任何个人持仓。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import Field

from app.schemas.simulation_market import MarketModel


class DiagnosisFactItem(MarketModel):
    """单项诊断事实；Python 只给当前状态与事实，基线对比由 Java 报告侧完成。"""

    item: Literal["MANAGER", "SCALE", "SAME_TYPE_RANK", "BENCHMARK", "DRAWDOWN", "FEE", "DIVIDEND"]
    verdict: Literal["VALID", "CHANGED", "INSUFFICIENT"] = Field(
        description="按文档阈值计算的参考结论；与报告基线的最终对比由 Java 侧判定"
    )
    evidence: str = Field(description="证据正文，含来源范围说明与分工说明")
    source: str = Field(description="事实来源标识（fund_ai 表或既有内部接口）")
    data_as_of_date: date | None = Field(default=None, description="该项事实的数据截至日；缺失不补造")
    facts: dict | None = Field(default=None, description="结构化事实字段；Java 以事实为准与报告基线对比")


class DiagnosisFacts(MarketModel):
    """七项持仓诊断的公共事实汇总；只读、不含任何用户身份。"""

    fund_code: str
    as_of_date: date
    overall: Literal["VALID", "CHANGED", "INSUFFICIENT"] = Field(
        description="按「任一 CHANGED→CHANGED；无 CHANGED 有 INSUFFICIENT→INSUFFICIENT；否则 VALID」的参考总体结论"
    )
    items: tuple[DiagnosisFactItem, ...]
    basis: str = "HOLDING_DIAGNOSIS_FACTS_V1"


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
