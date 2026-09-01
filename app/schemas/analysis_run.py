"""M3-05 Java 与 Python 间的受控分析运行内部契约。"""

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InternalRollingBacktestRequest(BaseModel):
    """固定候选模型的受控回测参数；基准必须是已启用的本地授权序列。"""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    fee_rate: Decimal = Field(default=Decimal("0.001500"), ge=0, lt=1)
    benchmark_code: str | None = Field(
        default=None,
        min_length=2,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$",
        alias="benchmarkCode",
    )


class InternalBenchmarkRegistrationRequest(BaseModel):
    """登记一个股票型候选回测基准；不会自动启用来源或模型。"""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    display_name: str = Field(min_length=1, max_length=128, alias="displayName")
    source_code: str = Field(min_length=1, max_length=64, alias="sourceCode")
    license_reference: str = Field(min_length=1, max_length=512, alias="licenseReference")


class InternalBenchmarkNavPointRequest(BaseModel):
    """人工核验后导入的一条基准日收盘点。"""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    nav_date: date = Field(alias="navDate")
    closing_value: Decimal = Field(gt=0, alias="closingValue")
    source_published_at: datetime | None = Field(default=None, alias="sourcePublishedAt")


class InternalBenchmarkPointImportRequest(BaseModel):
    """单次受控批量导入，限制请求规模避免无界写入。"""

    model_config = ConfigDict(frozen=True)

    points: tuple[InternalBenchmarkNavPointRequest, ...] = Field(min_length=1, max_length=10_000)


class InternalBenchmarkSeriesStatus(BaseModel):
    """供 Java 管理端读取的基准状态与覆盖摘要。"""

    model_config = ConfigDict(frozen=True)

    benchmark_code: str
    display_name: str
    fund_type: str
    source_code: str
    source_enabled: bool
    status: str
    license_reference: str
    point_count: int
    first_nav_date: date | None
    last_nav_date: date | None


class InternalAnalysisRunStatus(BaseModel):
    """持久分析任务的安全状态摘要，不返回行情原始数据或模型参数细节。"""

    model_config = ConfigDict(frozen=True)

    analysis_run_id: UUID
    run_type: str
    status: str
    fund_type: str
    task_id: str | None
    backtest_run_id: UUID | None
    model_release_id: UUID | None
    model_release_status: str | None
    failure_reason: str | None
    requested_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class InternalModelReleaseTransitionRequest(BaseModel):
    """管理员主动改变模型发布状态时必须给出可审计理由。"""

    model_config = ConfigDict(frozen=True)

    reason: str = Field(min_length=1, max_length=400)


class InternalModelReleaseStatus(BaseModel):
    """模型发布状态机转换后的最小状态摘要。"""

    model_config = ConfigDict(frozen=True)

    model_release_id: UUID
    model_code: str
    model_version: str
    feature_version: str
    fund_type: str
    backtest_run_id: UUID
    release_status: str
    effective_at: datetime | None
    suspended_at: datetime | None
    release_reason: str
