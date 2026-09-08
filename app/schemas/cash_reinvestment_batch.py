"""新版批量只读试跑契约；页大小只影响内部处理，不影响样本或内容指纹。"""

from datetime import date
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.cash_reinvestment_samples import CashSampleResponse


class CashBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)
    fund_code: Literal["001632", "006730", "008888"] = Field(alias="fundCode", description="既有三只股票型试点")
    start_date: date = Field(
        alias="startDate", ge=date(2022, 1, 1), le=date(2024, 12, 31), description="第一天信息截止日，含当天"
    )
    end_date: date = Field(
        alias="endDate",
        ge=date(2022, 1, 1),
        le=date(2024, 12, 31),
        description="最后一天信息截止日，含当天；非净值读取终点",
    )
    page_size: int = Field(
        default=10,
        alias="pageSize",
        ge=1,
        le=30,
        description="内部每页处理几道题，整次返回全部结果；不是预测周期或数据库原始行数",
    )

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if not 0 <= (self.end_date - self.start_date).days < 31:
            raise ValueError("startDate不能晚于endDate，且含首尾最多31个自然日。")
        return self


class CashBatchResponse(BaseModel):
    mode: Literal["READ_ONLY_CASH_REINVESTMENT_BATCH_DRY_RUN"] = Field(
        default="READ_ONLY_CASH_REINVESTMENT_BATCH_DRY_RUN", description="新版批量只读试跑，没有保存批次"
    )
    batch_rule_version: Literal["CASH_REINVESTMENT_BATCH_RULE_V1"] = Field(
        default="CASH_REINVESTMENT_BATCH_RULE_V1", description="交易日截止选择、分页及汇总规则版本"
    )
    sample_rule_version: Literal["CASH_REINVESTMENT_SAMPLE_RULE_V1"] = Field(
        default="CASH_REINVESTMENT_SAMPLE_RULE_V1", description="逐条严格复用既有新版单日规则"
    )
    status: Literal["DRY_RUN_COMPLETED", "NO_TRADING_CUTOFFS"] = Field(
        description="整段处理完成或范围内没有交易日；不表示所有题目都可用"
    )
    fund_code: str = Field(description="本次一只试点基金")
    start_date: date = Field(description="请求起日，含当天")
    end_date: date = Field(description="请求止日，含当天")
    cutoff_selection: Literal["TRADING_DAYS_ONLY"] = Field(
        default="TRADING_DAYS_ONLY", description="只在固定日历交易日出题，不由数据库有没有净值来选日期"
    )
    skipped_cutoff_dates: tuple[date, ...] = Field(
        description="请求范围中的周末/休市日，不出题；不是被忽略的源净值记录"
    )
    source_code: str = Field(description="本次一次性核验的启用来源")
    source_sync_run_id: UUID = Field(description="本事务共同的当前净值来源水位，不是历史首次版本证明")
    calendar_version: str = Field(description="本次固定研究日历版本")
    calendar_hash: str = Field(description="日历内容指纹")
    page_size: int = Field(description="本次请求的每页题目数；不参与batch_hash")
    page_count: int = Field(description="实际非空题目页数，不是SQL条数；不参与batch_hash")
    sample_count: int = Field(description="计划交易日题目总数，等于items长度；坏样本仍计入")
    input_available_count: int = Field(description="历史输入可计算的题数，不表示有答案或正式训练资格")
    label_available_count: int = Field(description="已算出后来答案的题数，不是预测命中数")
    ready_count: int = Field(description="输入及答案均可计算的题数")
    input_unavailable_count: int = Field(description="输入不可用的题数")
    label_unavailable_count: int = Field(description="输入已可用，但后来答案不可用的题数")
    input_issue_counts: dict[str, int] = Field(
        description="按问题代码统计受影响题数；一道题同一代码只计一次，多种原因会重复计入不同项"
    )
    label_issue_counts: dict[str, int] = Field(
        description="答案问题对应题数；INPUT_UNAVAILABLE表示因输入不可用而未附答案"
    )
    items: tuple[CashSampleResponse, ...] = Field(
        max_length=31, description="按截止日升序的全部样本，逐条结构与cash-reinvestment-preview完全相同"
    )
    batch_hash: str = Field(
        description="除page_size/page_count及本字段外整个响应的SHA256；包含后来答案，不可充当历史输入特征"
    )
    usage: Literal["LEARNING_ONLY"] = Field(default="LEARNING_ONLY", description="只供研究验收，不自动进入旧训练器")
    training_eligible: Literal[False] = Field(
        default=False, description="样本数量变多不解除事件完整性及历史版本准入缺口"
    )
    database_written: Literal[False] = Field(default=False, description="未保存、覆盖或补充任何业务数据")
    model_fitted: Literal[False] = Field(default=False, description="未训练或校准模型")
    test_scored: Literal[False] = Field(default=False, description="未做2025最终测试评分")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(
        default="MODEL_NOT_RELEASED", description="未发布用户可见预测"
    )
