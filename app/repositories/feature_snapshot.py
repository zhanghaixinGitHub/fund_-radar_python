"""M3 特征快照的受控读取与幂等写入。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.analysis import FeatureSnapshot
from app.models.fund import FundShareClass, NavDaily, SourceRegistry, SourceSyncRun

FEATURE_NAV_SYNC_TYPES = (
    "NAV_DAILY",
    "MARKET_NAV_HISTORY",
    "MARKET_DETAIL_NAV",
    "MARKET_NAV_INCREMENTAL",
)


@dataclass(frozen=True)
class FeatureSourceReadiness:
    """已登记且启用的数据源最小元数据，不含任何外部凭据。"""

    source_id: UUID
    source_code: str
    source_sync_run_id: UUID
    source_sync_finished_at: datetime


@dataclass(frozen=True)
class FeatureNavPoint:
    """一条来自同一受控来源的基金日净值。"""

    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None


@dataclass(frozen=True)
class StockFeatureInput:
    """一只试点股票型基金的有限历史净值输入。"""

    fund_code: str
    fund_type: str
    source_code: str
    source_sync_run_id: UUID
    source_sync_finished_at: datetime
    nav_points: tuple[FeatureNavPoint, ...]


@dataclass(frozen=True)
class FeatureSnapshotUpsert:
    """已完成计算、可以持久化的一条特征快照。"""

    fund_code: str
    as_of_date: date
    fund_type: str
    feature_version: str
    completeness: Decimal
    eligibility_status: str
    unavailable_reason: str | None
    feature_payload: dict[str, object]
    feature_hash: str


@dataclass(frozen=True)
class FeatureSnapshotWriteStats:
    """特征快照幂等写入统计。"""

    created_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0


def get_enabled_feature_source(session: Session, source_code: str) -> FeatureSourceReadiness | None:
    """只接受已登记、已启用的数据源，禁用或缺失时不允许构建新快照。"""
    source_row = session.execute(
        select(SourceRegistry.source_id, SourceRegistry.source_code, SourceRegistry.last_success_at).where(
            SourceRegistry.source_code == source_code,
            SourceRegistry.enabled.is_(True),
        )
    ).one_or_none()
    if source_row is None:
        return None
    sync_row = session.execute(
        select(SourceSyncRun.sync_run_id, SourceSyncRun.finished_at)
        .where(
            SourceSyncRun.source_id == source_row.source_id,
            SourceSyncRun.sync_type.in_(FEATURE_NAV_SYNC_TYPES),
            SourceSyncRun.status == "SUCCEEDED",
            SourceSyncRun.finished_at.is_not(None),
        )
        .order_by(SourceSyncRun.finished_at.desc(), SourceSyncRun.sync_run_id.desc())
        .limit(1)
    ).one_or_none()
    if sync_row is None:
        return None
    return FeatureSourceReadiness(
        source_id=source_row.source_id,
        source_code=source_row.source_code,
        source_sync_run_id=sync_row.sync_run_id,
        source_sync_finished_at=sync_row.finished_at,
    )


def list_stock_feature_inputs(
    session: Session,
    *,
    source_code: str,
    source_sync_run_id: UUID,
    source_sync_finished_at: datetime,
    history_limit: int,
) -> tuple[StockFeatureInput, ...]:
    """批量读取股票型试点基金最近净值，不混入其他来源或基金类型。"""
    if history_limit < 1:
        raise ValueError("history_limit must be positive")

    fund_codes = tuple(
        session.scalars(
            select(FundShareClass.fund_code)
            .where(
                FundShareClass.fund_type == "STOCK",
                FundShareClass.status == "ACTIVE",
                FundShareClass.source_code == source_code,
            )
            .order_by(FundShareClass.fund_code.asc())
        ).all()
    )
    if not fund_codes:
        return ()

    ranked_nav = (
        select(
            NavDaily.fund_code.label("fund_code"),
            NavDaily.nav_date.label("nav_date"),
            NavDaily.unit_nav.label("unit_nav"),
            NavDaily.accumulated_nav.label("accumulated_nav"),
            func.row_number()
            .over(partition_by=NavDaily.fund_code, order_by=NavDaily.nav_date.desc())
            .label("recency_rank"),
        )
        .select_from(NavDaily)
        .join(SourceRegistry, SourceRegistry.source_id == NavDaily.source_id)
        .where(
            NavDaily.fund_code.in_(fund_codes),
            SourceRegistry.source_code == source_code,
            SourceRegistry.enabled.is_(True),
        )
        .subquery()
    )
    points_by_code: dict[str, list[FeatureNavPoint]] = {fund_code: [] for fund_code in fund_codes}
    rows = session.execute(
        select(
            ranked_nav.c.fund_code,
            ranked_nav.c.nav_date,
            ranked_nav.c.unit_nav,
            ranked_nav.c.accumulated_nav,
        )
        .where(ranked_nav.c.recency_rank <= history_limit)
        .order_by(ranked_nav.c.fund_code.asc(), ranked_nav.c.nav_date.asc())
    ).all()
    for fund_code, nav_date, unit_nav, accumulated_nav in rows:
        points_by_code[fund_code].append(
            FeatureNavPoint(nav_date=nav_date, unit_nav=unit_nav, accumulated_nav=accumulated_nav)
        )
    return tuple(
        StockFeatureInput(
            fund_code=fund_code,
            fund_type="STOCK",
            source_code=source_code,
            source_sync_run_id=source_sync_run_id,
            source_sync_finished_at=source_sync_finished_at,
            nav_points=tuple(points_by_code[fund_code]),
        )
        for fund_code in fund_codes
    )


def upsert_feature_snapshots(
    session: Session, *, records: tuple[FeatureSnapshotUpsert, ...]
) -> FeatureSnapshotWriteStats:
    """按基金、估值日和特征版本幂等更新，未变化记录不推动计算时间。"""
    if not records:
        return FeatureSnapshotWriteStats()

    record_keys = {(record.fund_code, record.as_of_date, record.feature_version) for record in records}
    existing_by_key = {
        (snapshot.fund_code, snapshot.as_of_date, snapshot.feature_version): snapshot
        for snapshot in session.scalars(
            select(FeatureSnapshot).where(
                FeatureSnapshot.fund_code.in_({record.fund_code for record in records}),
                FeatureSnapshot.as_of_date.in_({record.as_of_date for record in records}),
                FeatureSnapshot.feature_version.in_({record.feature_version for record in records}),
            )
        ).all()
        if (snapshot.fund_code, snapshot.as_of_date, snapshot.feature_version) in record_keys
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        key = (record.fund_code, record.as_of_date, record.feature_version)
        existing = existing_by_key.get(key)
        if existing is None:
            session.add(
                FeatureSnapshot(
                    fund_code=record.fund_code,
                    as_of_date=record.as_of_date,
                    fund_type=record.fund_type,
                    feature_version=record.feature_version,
                    completeness=record.completeness,
                    eligibility_status=record.eligibility_status,
                    unavailable_reason=record.unavailable_reason,
                    feature_payload=record.feature_payload,
                    feature_hash=record.feature_hash,
                )
            )
            created_count += 1
            continue
        if existing.feature_hash == record.feature_hash:
            skipped_count += 1
            continue
        existing.fund_type = record.fund_type
        existing.completeness = record.completeness
        existing.eligibility_status = record.eligibility_status
        existing.unavailable_reason = record.unavailable_reason
        existing.feature_payload = record.feature_payload
        existing.feature_hash = record.feature_hash
        updated_count += 1
    return FeatureSnapshotWriteStats(
        created_count=created_count,
        updated_count=updated_count,
        skipped_count=skipped_count,
    )
