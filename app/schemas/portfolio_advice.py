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


class DraftTriggerStats(MarketModel):
    """一条草案阈值的历史触发统计；历史末端无法完整观察的触发记 censored，不补造数值。"""

    trigger_count: int = Field(description="滚动窗口穿越阈值的次数")
    median_further_decline: Decimal | None = Field(
        default=None, description="触发窗口终点后 63 个交易日内最大续跌幅度（正数）的中位；无完整前向窗口时为 null"
    )
    median_recovery_days: Decimal | None = Field(
        default=None, description="触发后净值回到窗口峰值所需交易日数的中位；无已修复样本时为 null"
    )
    censored_count: int = Field(default=0, description="因历史末端无前向窗口或始终未修复而被剔除的触发次数")


class DraftTier(MarketModel):
    """一档草案阈值；语义仅为「达到后提示复核」，不承诺触发后走势。"""

    tier: Literal["CONSERVATIVE", "BALANCED", "LOOSE"]
    reduce_drawdown_pct: Decimal = Field(description="减仓线：回撤触发阈值，负数小数（如 -0.08 表示回撤 8%）")
    take_profit_pct: Decimal = Field(description="止盈线：收益触发阈值，小数（如 0.10 表示区间收益 10%）")
    reduce_trigger: DraftTriggerStats
    take_profit_trigger: DraftTriggerStats


class DraftStats(MarketModel):
    """规则草案统计；历史不足或不适用时返回明确状态与原因，不给阈值数字。不含任何用户身份。"""

    fund_code: str
    status: Literal["AVAILABLE", "DATA_INSUFFICIENT", "NOT_APPLICABLE"]
    reason: str | None = Field(default=None, description="数据不足/不适用的具体原因；AVAILABLE 时为 null")
    stats_cutoff_date: date | None = Field(default=None, description="统计所用净值截止日")
    history_days: int = 0
    window_count: int = 0
    nav_basis: Literal["ACCUMULATED", "UNIT"] | None = None
    stats: dict | None = Field(default=None, description="DD/R/L20 分布分位、最大回撤、修复天数、波动率及同类分位")
    tiers: tuple[DraftTier, ...] = ()
    assumption: str = (
        "沿用 nav_daily 现有口径，分红不自行还原复权；草案仅为参考阈值，须本人确认后才生效，不构成投资建议。"
    )
    basis: str = "HOLDING_RULE_DRAFT_STATS_V1"


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
