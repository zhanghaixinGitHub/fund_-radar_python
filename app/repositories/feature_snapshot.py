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

    # 来源登记表主键；读nav_daily时按source_id筛选，避免混用其他来源。
    source_id: UUID
    # 人可读的来源编码，例如TUSHARE_PRO_FUND。
    source_code: str
    # 该来源最近一次成功净值同步的运行标识；不是每条历史净值各自的首次来源记录。
    source_sync_run_id: UUID
    # 上述同步任务完成时间；反映本地数据来源水位，不代表历史净值公告日期。
    source_sync_finished_at: datetime


@dataclass(frozen=True)
class FeatureNavPoint:
    """旧版最新特征快照的净值输入，不带公告日；阶段2历史样本使用HistoricalNavPoint。"""

    # 净值所属业务日期。
    nav_date: date
    # 单位净值，用Decimal接收数据库小数。
    unit_nav: Decimal
    # 累计净值，可缺失；是否回退由使用此对象的计算器按版本规则决定。
    accumulated_nav: Decimal | None


@dataclass(frozen=True)
class StockFeatureInput:
    """一只试点股票型基金的有限历史净值输入。"""

    # 当前快照对应的基金份额代码。
    fund_code: str
    # 基金品类，股票型试点为STOCK。
    fund_type: str
    # 所有净值来自哪个登记来源。
    source_code: str
    # 对应来源当前最近一次成功净值同步ID，用于追溯。
    source_sync_run_id: UUID
    # 同步完成时刻，不是历史样本公告可得时刻。
    source_sync_finished_at: datetime
    # 从早到晚的有限净值序列；这个旧结构只用于现有最新快照计算。
    nav_points: tuple[FeatureNavPoint, ...]


@dataclass(frozen=True)
class FeatureSnapshotUpsert:
    """已完成计算、可以持久化的一条特征快照。"""

    # 快照归属的基金份额代码。
    fund_code: str
    # 当前快照使用的最后一个净值业务日。
    as_of_date: date
    # 该基金的类型，供不同类型快照隔离处理。
    fund_type: str
    # 生成快照的特征规则版本，与基金和业务日共同构成业务唯一键。
    feature_version: str
    # 输入完整度，范围0到1；不是预测准确率或上涨概率。
    completeness: Decimal
    # 特征是否满足该版本的计算门槛，如SCORABLE或DATA_INSUFFICIENT。
    eligibility_status: str
    # 未达门槛时的原因，满足时为空。
    unavailable_reason: str | None
    # 要保存的特征内容字典；不是模型预测结果。
    feature_payload: dict[str, object]
    # 内容校验指纹，用于比较重跑后的快照是否真的变化。
    feature_hash: str


@dataclass(frozen=True)
class FeatureSnapshotWriteStats:
    """特征快照幂等写入统计。"""

    # 本次新插入的快照数量。
    created_count: int = 0
    # 已存在但内容发生变化、本次执行了更新的数量。
    updated_count: int = 0
    # 内容未变而跳过写入的数量；这就是相同输入重跑不重复写的依据。
    skipped_count: int = 0


def get_enabled_feature_source(session: Session, source_code: str) -> FeatureSourceReadiness | None:
    """只接受已登记、已启用的数据源，禁用或缺失时不允许构建新快照。"""
    # 先确认来源仍启用；有历史净值不代表一个已禁用的来源仍可被当前功能使用。
    source_row = session.execute(
        select(SourceRegistry.source_id, SourceRegistry.source_code, SourceRegistry.last_success_at).where(
            SourceRegistry.source_code == source_code,
            SourceRegistry.enabled.is_(True),
        )
    ).one_or_none()
    if source_row is None:
        return None
    # 再找最近一条已完成且成功的“净值类同步”，不能把其他类型同步成功误当成净值就绪。
    # finished_at倒序取1条；相同时再按运行ID排序，使选择结果稳定。
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
    # 两项检查都通过才返回来源信息；调用方负责将None转为“来源未就绪”，不自动发起同步。
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
