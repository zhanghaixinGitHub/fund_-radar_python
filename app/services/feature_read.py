"""将已持久化 M3-G1 特征映射为 Java 到 Python 的内部读取契约。"""

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.session import get_engine
from app.models.analysis import FeatureSnapshot
from app.repositories.feature_read import get_latest_feature_snapshot
from app.schemas.feature import InternalFeatureSnapshot, InternalFeatureStatus, InternalStockFeatureMetrics
from app.services.stock_feature_snapshot import STOCK_FEATURE_VERSION


def get_latest_stock_feature_status(fund_code: str) -> InternalFeatureStatus:
    """只返回已经落库的股票型特征状态；无记录时不临时计算也不返回预测。"""
    with Session(get_engine()) as session:
        snapshot = get_latest_feature_snapshot(
            session,
            fund_code=fund_code,
            feature_version=STOCK_FEATURE_VERSION,
        )
    if snapshot is None:
        return InternalFeatureStatus(status="NOT_AVAILABLE", snapshot=None)
    return InternalFeatureStatus(status="AVAILABLE", snapshot=_to_internal_snapshot(snapshot))


def _to_internal_snapshot(snapshot: FeatureSnapshot) -> InternalFeatureSnapshot:
    source = _mapping_value(snapshot.feature_payload, "source")
    metrics = _mapping_value(snapshot.feature_payload, "metrics")
    return InternalFeatureSnapshot(
        fund_code=snapshot.fund_code,
        as_of_date=snapshot.as_of_date,
        fund_type=snapshot.fund_type,
        feature_version=snapshot.feature_version,
        completeness=snapshot.completeness,
        eligibility_status=snapshot.eligibility_status,
        unavailable_reason=snapshot.unavailable_reason,
        source_code=_optional_text(source, "source_code"),
        source_sync_finished_at=_optional_datetime(source, "source_sync_finished_at"),
        nav_value_basis=_optional_text(source, "nav_value_basis"),
        metrics=InternalStockFeatureMetrics.model_validate(metrics) if metrics is not None else None,
        computed_at=snapshot.computed_at,
    )


def _mapping_value(payload: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _optional_text(payload: Mapping[str, object] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _optional_datetime(payload: Mapping[str, object] | None, key: str) -> datetime | None:
    value = _optional_text(payload, key)
    return datetime.fromisoformat(value) if value is not None else None
