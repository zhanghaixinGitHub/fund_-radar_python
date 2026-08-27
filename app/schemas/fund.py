"""M0 Mock 基金读模型的 Pydantic 内部接口契约。"""

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InternalFundSummary(BaseModel):
    """传递给 Java 核心服务的最小且可公开展示的基金摘要。"""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    fund_name: str
    fund_type: str
    status: str
    # 目录可以先于首个合规净值同步落库，因此此字段允许为空。
    # 空值只能表示“尚无同步净值”，不能由调用方补成当前日期。
    as_of_date: date | None


class InternalFundDetail(InternalFundSummary):
    """带最新已落库净值快照、状态与来源信息的内部基金详情读模型。"""

    nav_status: str
    data_source: str
    # 数值为空表示该份额当前尚无已同步净值；调用方不得补零或当作实时行情。
    unit_nav: Decimal | None = None
    accumulated_nav: Decimal | None = None


class InternalFundNavPoint(BaseModel):
    """一条已落库且可展示的历史日净值，不代表实时估值或交易价格。"""

    model_config = ConfigDict(frozen=True)

    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None


class InternalFundNavHistory(BaseModel):
    """指定基金、指定日期窗口内的历史净值读模型。"""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    items: tuple[InternalFundNavPoint, ...]


class InternalSyncJobStatus(BaseModel):
    """同步中心任务的安全状态摘要，不返回 Token 或外部原始响应。"""

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    job_type: str
    status: str
    requested_nav_date: date
    fund_codes: tuple[str, ...]
    progress_current: int
    progress_total: int
    current_fund_code: str | None
    progress_message: str
    sync_run_id: UUID | None
    fetched_count: int
    created_count: int
    updated_count: int
    skipped_count: int
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None


class InternalFundPage(BaseModel):
    """返回给 Java 核心服务的兼容游标/页码分页响应。"""

    model_config = ConfigDict(frozen=True)

    items: tuple[InternalFundSummary, ...]
    next_cursor: str | None = Field(default=None, serialization_alias="next_cursor")
    page: int | None = None
    page_size: int = Field(default=20, serialization_alias="page_size")
    total_count: int = Field(default=0, ge=0, serialization_alias="total_count")
    total_pages: int = Field(default=0, ge=0, serialization_alias="total_pages")
