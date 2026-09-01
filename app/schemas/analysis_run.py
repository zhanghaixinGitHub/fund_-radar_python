"""M3-05 Java 与 Python 间的受控分析运行内部契约。"""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InternalRollingBacktestRequest(BaseModel):
    """只允许调整本地回测摩擦成本；基准数据仍须通过受控配置另行授权。"""

    model_config = ConfigDict(frozen=True)

    fee_rate: Decimal = Field(default=Decimal("0.001500"), ge=0, lt=1)


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
