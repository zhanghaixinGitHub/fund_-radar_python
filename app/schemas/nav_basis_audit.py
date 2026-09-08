"""净值口径审计契约：展示可复算差异，不把审计结果当作训练标签。"""

from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NavBasisAuditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    fund_code: Literal["001632", "006730", "008888"] = Field(alias="fundCode", description="既有三只试点之一")
    start_date: date = Field(
        alias="startDate", ge=date(2022, 1, 1), le=date(2024, 12, 31), description="审计起日，含当天"
    )
    end_date: date = Field(
        alias="endDate", ge=date(2022, 1, 1), le=date(2024, 12, 31), description="审计止日，2025测试期不开放数值审计"
    )

    @model_validator(mode="after")
    def bounded_range(self):
        if not 0 <= (self.end_date - self.start_date).days <= 365:
            raise ValueError("审计范围须从早到晚，最多366个自然日。")
        return self


class BasisAuditIssue(BaseModel):
    day: date | None = Field(description="问题对应的日期；来源未给出有效日时可为空")
    code: str = Field(description="稳定原因码，不是自动修复指令")


class BasisPeriodComparison(BaseModel):
    base_date: date = Field(description="审计首个交易日前一个交易日，仅作为收益分母，不计入审计天数")
    end_date: date = Field(description="审计范围内最后一个交易日")
    unit_nav_ratio_return: str | None = Field(description="同端点单位净值之比减1，未补现金分红")
    accumulated_nav_ratio_return: str | None = Field(description="同端点累计净值之比减1，不等同于现金分红再投资收益")
    source_adjusted_nav_ratio_return: str | None = Field(
        description="同端点来源复权净值之比减1，不宣称供应商复权公式已获确认"
    )
    cash_reinvestment_candidate_return: str | None = Field(
        description="仅在日期/事件/数值检查完整时，按现有事件假设当日现金再投的日收益连乘；不证明事件完整或历史首次可得"
    )


class DividendBasisComparison(BaseModel):
    effective_date: date = Field(description="唯一且合法的净值除权日；无此字段时使用除息日，两者冲突则不计算")
    previous_trading_date: date = Field(description="前一个准确交易日，不使用周末净值替代")
    unit_nav_before: str = Field(description="前一个交易日的单位净值")
    unit_nav_after: str = Field(description="当前交易日的单位净值")
    cash_per_share: str = Field(description="来源已实施事件中的每份现金分红，单位元；不再除以10或100")
    unit_nav_ratio_return: str = Field(description="未考虑现金分红的单位净值变化")
    cash_inclusive_day_return: str = Field(
        description="(当日单位净值+每份现金分红)/前日单位净值-1；显式算式，不冒充供应商公式"
    )
    source_adjusted_day_return: str = Field(description="当日来源复权净值/前日来源复权净值-1")
    absolute_gap_bps: str = Field(description="上面两种分红后口径的绝对差，100基点=1个百分点")
    ex_cash_denominator_hypothesis_return: str | None = Field(
        description="仅作公式假设：当日单位净值/(前日单位净值-分红)-1；数值接近不是供应商确认"
    )


class NavBasisAuditResponse(BaseModel):
    mode: Literal["READ_ONLY_NAV_BASIS_AUDIT"] = Field(default="READ_ONLY_NAV_BASIS_AUDIT", description="只读口径审计")
    audit_rule_version: Literal["NAV_BASIS_AUDIT_V1"] = Field(
        default="NAV_BASIS_AUDIT_V1", description="固定计算与拒收规则版本"
    )
    status: Literal["DIFFERENCES_FOUND", "NO_LARGE_DAILY_DIFFERENCE_FOUND", "AUDIT_INCOMPLETE"] = Field(
        description="核对结果；未发现大差异不是正式准入通过"
    )
    fund_code: str = Field(description="当前基金")
    start_date: date = Field(description="请求起日")
    end_date: date = Field(description="请求止日")
    source_code: str = Field(description="净值和分红的共同启用来源")
    source_sync_run_id: UUID = Field(description="当前净值同步水位，不证明分红历史已完整同步")
    calendar_version: str = Field(description="独立交易日历版本")
    calendar_hash: str = Field(description="日历内容指纹")
    snapshot_hash: str = Field(description="本次有限范围净值和分红审计字段指纹，包含数值；不是特征哈希")
    trading_day_count: int = Field(description="请求范围内应有的交易日数量，不包含额外前置基准日")
    raw_nav_count: int = Field(description="实际读取净值行数，包含前置日和非交易日记录")
    ignored_non_trading_dates: tuple[date, ...] = Field(description="不参与收益日序列的源记录日期，未删除")
    invalid_nav_counts: dict[str, int] = Field(description="每种口径在所需日期上的缺失/非正/非有限数量，缺记录也计入")
    accumulated_dividend_missing_count: int = Field(description="所需日期缺累计分红字段的数量；缺失不当作没有分红")
    dividend_record_count: int = Field(description="本范围可能相关的源事件数量，包含需人工核对的异常事件")
    checked_daily_pairs: int = Field(description="完成单位/复权/已知现金日收益对照的相邻交易日数量")
    daily_gap_threshold_bps: Literal["1"] = Field(
        default="1", description="固定1基点诊断展示阈值，不是收益率舍入或模型发布阈值"
    )
    daily_gap_over_threshold_count: int = Field(description="现金算式与来源复权的绝对差超过1基点的天数")
    max_daily_gap_bps: str | None = Field(description="已完成日对照中的最大绝对差；没完成则null")
    issues: tuple[BasisAuditIssue, ...] = Field(
        description="缺日期/数值、事件冲突、无事件解释的日差异等；不自动改数或补事件"
    )
    period_comparison: BasisPeriodComparison = Field(
        description="同起终点三种源口径及条件性现金再投算式对照，不生成20日标签"
    )
    dividend_comparisons: tuple[DividendBasisComparison, ...] = Field(
        description="合法实施事件日的全部可复算对照，最多为本范围交易日数量"
    )
    admission_status: Literal["NOT_APPROVED"] = Field(
        default="NOT_APPROVED", description="所有返回状态均未取得正式训练或发布资格"
    )
    automatic_basis_fallback_allowed: Literal[False] = Field(default=False, description="禁止缺哪个净值就切换成另一种")
    database_written: Literal[False] = Field(default=False, description="未写数据库")
    feature_generated: Literal[False] = Field(default=False, description="未生成特征")
    label_generated: Literal[False] = Field(default=False, description="未生成训练答案")
    model_fitted: Literal[False] = Field(default=False, description="未训练或校准")
    test_scored: Literal[False] = Field(default=False, description="未计算2025测试成绩")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(default="MODEL_NOT_RELEASED", description="尚未发布模型")
    evidence_urls: tuple[str, ...] = Field(description="供应商字段说明，运行时不访问网页")
    limitations: tuple[str, ...] = Field(description="公式、事件完整性、历史版本与研究使用限制")
