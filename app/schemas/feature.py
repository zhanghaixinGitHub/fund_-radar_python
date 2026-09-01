"""M3 股票型特征快照的内部只读契约。"""

from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class InternalStockFeatureMetrics(BaseModel):
    """固定版本的统计特征；全部是历史净值统计量，不含预测字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    return_5d: Decimal
    return_20d: Decimal
    return_60d: Decimal
    volatility_20d: Decimal
    max_drawdown_60d: Decimal


class InternalFeatureSnapshot(BaseModel):
    """一条供 Java 服务映射的可追溯特征快照。"""

    model_config = ConfigDict(frozen=True)

    fund_code: str
    as_of_date: date
    fund_type: str
    feature_version: str
    completeness: Decimal
    eligibility_status: str
    unavailable_reason: str | None
    source_code: str | None
    source_sync_finished_at: datetime | None
    nav_value_basis: str | None
    metrics: InternalStockFeatureMetrics | None
    computed_at: datetime


class InternalFeatureStatus(BaseModel):
    """指定基金的特征可用性；无快照是正常业务状态而非预测失败。"""

    model_config = ConfigDict(frozen=True)

    status: str
    snapshot: InternalFeatureSnapshot | None
