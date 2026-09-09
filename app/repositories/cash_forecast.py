"""现金结果有界读取；原始净值仅查询最新已公告日期，不在GET中重新读取价格。"""

from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from app.models.cash_forecast import CashForecastRecord
from app.models.fund import FundShareClass, NavDaily, SourceRegistry, SourceSyncRun


def find_forecast(session: Session, *, forecast_id: UUID | None = None, request_key: UUID | None = None):
    if (forecast_id is None) == (request_key is None):
        raise ValueError("exactly one forecast lookup key is required")
    clause = (
        CashForecastRecord.forecast_id == forecast_id if forecast_id else CashForecastRecord.request_key == request_key
    )
    return session.scalar(select(CashForecastRecord).where(clause))


def find_latest_forecast(session: Session, *, fund_code: str):
    """只取最新一份生成记录；不按分数/状态挑模型，也不越过失效记录回退到旧结果。"""
    return session.scalar(
        select(CashForecastRecord)
        .where(CashForecastRecord.fund_code == fund_code)
        .order_by(
            CashForecastRecord.cutoff_date.desc(),
            CashForecastRecord.created_at.desc(),
            CashForecastRecord.forecast_id.desc(),
        )
        .limit(1)
    )


def find_identical_forecast(
    session: Session,
    *,
    fund_code: str,
    cutoff: date,
    authorization_hash: str,
    feature_hash: str,
    source_revision_id: UUID,
):
    return session.scalar(
        select(CashForecastRecord).where(
            CashForecastRecord.fund_code == fund_code,
            CashForecastRecord.cutoff_date == cutoff,
            CashForecastRecord.authorization_hash == authorization_hash,
            CashForecastRecord.feature_hash == feature_hash,
            CashForecastRecord.source_revision_id == source_revision_id,
        )
    )


def current_cash_source_revision(session: Session, *, fund_code: str) -> UUID | None:
    """包括分红更新在内的来源水位；有运行中任务或最近同步失败时拒绝继续使用旧快照。"""
    running = aliased(SourceSyncRun)
    has_running = (
        select(running.sync_run_id)
        .where(running.source_id == SourceSyncRun.source_id, running.status == "RUNNING")
        .exists()
    )
    row = session.execute(
        select(
            SourceSyncRun.sync_run_id, SourceSyncRun.status, SourceSyncRun.finished_at, has_running.label("has_running")
        )
        .join(SourceRegistry, SourceRegistry.source_id == SourceSyncRun.source_id)
        .join(FundShareClass, FundShareClass.source_code == SourceRegistry.source_code)
        .where(
            FundShareClass.fund_code == fund_code, SourceRegistry.enabled.is_(True), SourceSyncRun.status != "PENDING"
        )
        .order_by(SourceSyncRun.finished_at.desc().nulls_first(), SourceSyncRun.sync_run_id.desc())
        .limit(1)
    ).one_or_none()
    if row is None or row.status != "SUCCEEDED" or row.finished_at is None or row.has_running:
        return None
    return row.sync_run_id


def latest_announced_nav_date(session: Session, *, fund_code: str, source_id: UUID, closed_day: date) -> date | None:
    return session.scalar(
        select(NavDaily.nav_date)
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.source_id == source_id,
            NavDaily.nav_date <= closed_day,
            NavDaily.ann_date <= closed_day,
            NavDaily.ann_date >= NavDaily.nav_date,
        )
        .order_by(NavDaily.nav_date.desc())
        .limit(1)
    )
