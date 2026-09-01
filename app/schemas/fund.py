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
    # 涨跌率均从同一份额已落库的累计净值计算；缺少对应基准净值时保持为空。
    day_change_rate: Decimal | None = None
    week_change_rate: Decimal | None = None
    month_change_rate: Decimal | None = None


class InternalFundDetail(InternalFundSummary):
    """带最新已落库净值快照、状态与来源信息的内部基金详情读模型。"""

    nav_status: str
    data_source: str
    # 数值为空表示该份额当前尚无已同步净值；调用方不得补零或当作实时行情。
    unit_nav: Decimal | None = None
    accumulated_nav: Decimal | None = None
    nav_ann_date: date | None = None
    accumulated_dividend: Decimal | None = None
    net_asset: Decimal | None = None
    total_net_asset: Decimal | None = None
    adjusted_nav: Decimal | None = None
    # 基金市场只展示本期可公开的基础资料；未完成详情基线时保持为空。
    profile_status: str = "NOT_SYNCED"
    profile_data_source: str | None = None
    management_company_name: str | None = None
    custodian_name: str | None = None
    found_date: date | None = None
    due_date: date | None = None
    list_date: date | None = None
    issue_date: date | None = None
    delist_date: date | None = None
    issue_amount: Decimal | None = None
    management_fee: Decimal | None = None
    custodian_fee: Decimal | None = None
    duration_year: Decimal | None = None
    par_value: Decimal | None = None
    min_purchase_amount: Decimal | None = None
    expected_return: Decimal | None = None
    benchmark: str | None = None
    invest_type: str | None = None
    source_fund_type: str | None = None
    trustee_name: str | None = None
    purchase_start_date: date | None = None
    redemption_start_date: date | None = None
    market: str | None = None


class InternalFundManager(BaseModel):
    """关注后可展示的基金经理任职资料，不返回简历等非必要个人信息。"""

    model_config = ConfigDict(frozen=True)

    manager_name: str
    ann_date: date | None = None
    begin_date: date | None = None
    end_date: date | None = None
    education: str | None = None
    data_source: str


class InternalFundShareSnapshot(BaseModel):
    """关注后可展示的最新基金份额规模快照。"""

    model_config = ConfigDict(frozen=True)

    trade_date: date
    fund_share: Decimal
    data_source: str


class InternalFundDividend(BaseModel):
    """关注后可展示的结构化分红事件，不包含资讯原文。"""

    model_config = ConfigDict(frozen=True)

    ann_date: date | None = None
    implementation_ann_date: date | None = None
    base_date: date | None = None
    process_status: str | None = None
    record_date: date | None = None
    ex_date: date | None = None
    pay_date: date | None = None
    earnings_pay_date: date | None = None
    nav_ex_date: date | None = None
    cash_dividend: Decimal | None = None
    base_unit: Decimal | None = None
    distributable_earnings: Decimal | None = None
    earnings_amount: Decimal | None = None
    reinvestment_arrival_date: date | None = None
    base_year: str | None = None
    data_source: str


class InternalFundWatchlistDetail(BaseModel):
    """只提供给 Java 的完整详情读模型；不包含任何用户或授权状态。"""

    model_config = ConfigDict(frozen=True)

    basic: InternalFundDetail
    managers_status: str
    managers: tuple[InternalFundManager, ...]
    latest_share_status: str
    latest_share: InternalFundShareSnapshot | None = None
    dividends_status: str
    dividends: tuple[InternalFundDividend, ...]


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


class InternalFundShareHistory(BaseModel):
    """关注后可读取的份额规模历史；状态用于阻止未同步数据被误作完整趋势。"""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    status: str
    items: tuple[InternalFundShareSnapshot, ...]


class InternalFundSameTypeComparisonItem(BaseModel):
    """当前基金市场内、同基金类型的一条事实比较项。"""

    model_config = ConfigDict(frozen=True)

    rank: int
    fund_code: str
    fund_name: str
    fund_type: str
    as_of_date: date
    month_change_rate: Decimal
    data_source: str


class InternalFundSameTypeComparison(BaseModel):
    """受控样本的同类型比较；不得解释为全市场排名。"""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    fund_type: str
    scope: str
    status: str
    as_of_date: date | None = None
    target_rank: int | None = None
    comparable_count: int = 0
    items: tuple[InternalFundSameTypeComparisonItem, ...] = ()


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


class InternalSyncJobLastSuccess(BaseModel):
    """一类同步任务最近一次完整成功的持久化时间。"""

    model_config = ConfigDict(frozen=True)

    job_type: str
    last_successful_at: datetime | None


class InternalFundPage(BaseModel):
    """返回给 Java 核心服务的兼容游标/页码分页响应。"""

    model_config = ConfigDict(frozen=True)

    items: tuple[InternalFundSummary, ...]
    next_cursor: str | None = Field(default=None, serialization_alias="next_cursor")
    page: int | None = None
    page_size: int = Field(default=20, serialization_alias="page_size")
    total_count: int = Field(default=0, ge=0, serialization_alias="total_count")
    total_pages: int = Field(default=0, ge=0, serialization_alias="total_pages")
