"""M3-06 面向基金详情的最小分析与回测摘要内部契约。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class InternalBacktestSummary(BaseModel):
    """只披露已发布模型关联回测的用户可读摘要，不透传原始样本或运行错误。"""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    status: str
    publication_status: str
    window_start: date
    window_end: date
    test_start: date | None
    test_end: date | None
    data_cutoff: date | None
    fee_rate: Decimal
    sample_count: int | None
    rolling_fold_count: int | None
    annualized_return: Decimal | None
    max_drawdown: Decimal | None
    volatility: Decimal | None
    hit_rate: Decimal | None
    long_hold_result: Decimal | None
    dca_result: Decimal | None
    benchmark_status: str | None
    benchmark_result: Decimal | None
    completed_at: datetime | None


class InternalModelAnalysisSummary(BaseModel):
    """已发布或已暂停模型的非敏感版本摘要。"""

    model_config = ConfigDict(frozen=True)

    model_release_id: UUID
    model_version: str
    feature_version: str
    release_status: Literal["ACTIVE", "SUSPENDED"]
    effective_at: datetime | None
    suspended_at: datetime | None


class InternalFundAnalysisSummary(BaseModel):
    """基金详情读取的发布闸门状态；读取不会启动评分、回测或模型发布。"""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    fund_type: str | None
    availability_status: Literal["ACTIVE", "MODEL_PAUSED", "MODEL_UNAVAILABLE"]
    message: str
    model: InternalModelAnalysisSummary | None
    backtest: InternalBacktestSummary | None
