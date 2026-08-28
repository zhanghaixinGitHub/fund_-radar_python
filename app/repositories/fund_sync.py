"""基金目录、净值与来源同步运行记录的数据访问实现。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from app.models.fund import FundMaster, FundShareClass, NavDaily, SourceRegistry, SourceSyncRun

TUSHARE_SOURCE_CODE = "TUSHARE_PRO_FUND"
TUSHARE_SOURCE_DISPLAY_NAME = "Tushare Pro 公募基金数据"
TUSHARE_SOURCE_KIND = "MARKET_DATA"
TUSHARE_SOURCE_LICENSE_SCOPE = (
    "用户已授权的个人 Tushare 积分范围内 fund_company、fund_basic、fund_nav；仅供本机基金雷达使用。"
)
TUSHARE_SOURCE_RATE_LIMIT_PER_MINUTE = 200
TUSHARE_SOURCE_RETENTION_DAYS = 365


@dataclass(frozen=True)
class FundCatalogUpsert:
    """落库前已规范化的基金份额目录记录。"""

    fund_code: str
    manager_name: str
    master_name: str
    fund_name: str
    fund_type: str
    status: str
    share_class: str
    established_date: date | None
    source_fund_code: str


@dataclass(frozen=True)
class NavDailyUpsert:
    """落库前已校验、哈希化的单条基金日净值。"""

    fund_code: str
    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None
    content_hash: str


@dataclass(frozen=True)
class MarketSyncTarget:
    """基金市场中需要由指定来源补齐净值的启用份额。"""

    fund_code: str
    source_fund_code: str | None


@dataclass(frozen=True)
class WriteStats:
    """单批写入的新增、更新和跳过计数。"""

    created_count: int = 0
    updated_count: int = 0
    skipped_count: int = 0

    def combine(self, other: WriteStats) -> WriteStats:
        """返回两个批次统计相加后的不可变结果。"""
        return WriteStats(
            created_count=self.created_count + other.created_count,
            updated_count=self.updated_count + other.updated_count,
            skipped_count=self.skipped_count + other.skipped_count,
        )


def ensure_tushare_source(session: Session) -> SourceRegistry:
    """幂等登记并启用已获用户授权的 Tushare 公募基金来源。"""
    source = session.scalar(select(SourceRegistry).where(SourceRegistry.source_code == TUSHARE_SOURCE_CODE))
    if source is None:
        source = SourceRegistry(
            source_code=TUSHARE_SOURCE_CODE,
            display_name=TUSHARE_SOURCE_DISPLAY_NAME,
            source_kind=TUSHARE_SOURCE_KIND,
            license_scope=TUSHARE_SOURCE_LICENSE_SCOPE,
            rate_limit_per_minute=TUSHARE_SOURCE_RATE_LIMIT_PER_MINUTE,
            retention_days=TUSHARE_SOURCE_RETENTION_DAYS,
            enabled=True,
        )
        session.add(source)
        session.flush()
        return source

    source.display_name = TUSHARE_SOURCE_DISPLAY_NAME
    source.source_kind = TUSHARE_SOURCE_KIND
    source.license_scope = TUSHARE_SOURCE_LICENSE_SCOPE
    source.rate_limit_per_minute = TUSHARE_SOURCE_RATE_LIMIT_PER_MINUTE
    source.retention_days = TUSHARE_SOURCE_RETENTION_DAYS
    source.enabled = True
    return source


def create_sync_run(
    session: Session, *, source_id: UUID, sync_type: str, requested_nav_date: date | None
) -> SourceSyncRun:
    """创建 RUNNING 状态的同步记录并返回已持久化的运行标识。"""
    run = SourceSyncRun(
        source_id=source_id,
        sync_type=sync_type,
        requested_nav_date=requested_nav_date,
        status="RUNNING",
    )
    session.add(run)
    session.flush()
    return run


def complete_sync_run(
    session: Session,
    *,
    source_id: UUID,
    sync_run_id: UUID,
    fetched_count: int,
    write_stats: WriteStats,
) -> None:
    """将同步运行标为成功并更新数据源最近成功时间。"""
    run = _get_sync_run(session, sync_run_id)
    source = _get_source(session, source_id)
    completed_at = datetime.now(UTC)
    run.status = "SUCCEEDED"
    run.fetched_count = fetched_count
    run.created_count = write_stats.created_count
    run.updated_count = write_stats.updated_count
    run.skipped_count = write_stats.skipped_count
    run.error_summary = None
    run.finished_at = completed_at
    source.last_success_at = completed_at
    if source.last_error_summary and source.last_error_summary.startswith(f"{run.sync_type}:"):
        source.last_error_at = None
        source.last_error_summary = None


def fail_sync_run(session: Session, *, source_id: UUID, sync_run_id: UUID, error_summary: str) -> None:
    """将同步运行和来源诊断状态标为失败，保留最后一次成功时间。"""
    run = _get_sync_run(session, sync_run_id)
    source = _get_source(session, source_id)
    failed_at = datetime.now(UTC)
    safe_summary = f"{run.sync_type}: {error_summary}"[:512]
    run.status = "FAILED"
    run.error_summary = safe_summary
    run.finished_at = failed_at
    source.last_error_at = failed_at
    source.last_error_summary = safe_summary


def upsert_fund_catalog_batch(session: Session, records: tuple[FundCatalogUpsert, ...]) -> WriteStats:
    """按基金代码幂等写入目录；不因本批缺失而删除历史基金。"""
    if not records:
        return WriteStats()
    fund_codes = tuple(record.fund_code for record in records)
    existing_shares = {
        share.fund_code: share
        for share in session.scalars(select(FundShareClass).where(FundShareClass.fund_code.in_(fund_codes))).all()
    }
    master_keys = {
        (record.manager_name, record.master_name) for record in records if record.fund_code not in existing_shares
    }
    existing_masters: dict[tuple[str, str], FundMaster] = {}
    if master_keys:
        statement = select(FundMaster).where(
            tuple_(FundMaster.manager_name, FundMaster.fund_name).in_(tuple(master_keys))
        )
        existing_masters = {
            (master.manager_name, master.fund_name): master for master in session.scalars(statement).all()
        }

    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        share = existing_shares.get(record.fund_code)
        if share is None:
            master_key = (record.manager_name, record.master_name)
            master = existing_masters.get(master_key)
            if master is None:
                master = FundMaster(
                    fund_master_id=uuid4(),
                    fund_name=record.master_name,
                    manager_name=record.manager_name,
                    fund_type=record.fund_type,
                    status=record.status,
                    established_date=record.established_date,
                )
                session.add(master)
                existing_masters[master_key] = master
            session.add(
                FundShareClass(
                    fund_code=record.fund_code,
                    fund_master_id=master.fund_master_id,
                    share_class=record.share_class,
                    fund_name=record.fund_name,
                    fund_type=record.fund_type,
                    status=record.status,
                    source_code=TUSHARE_SOURCE_CODE,
                    source_fund_code=record.source_fund_code,
                )
            )
            created_count += 1
            continue

        changed = False
        for field_name, expected_value in (
            ("fund_name", record.fund_name),
            ("fund_type", record.fund_type),
            ("status", record.status),
            ("source_code", TUSHARE_SOURCE_CODE),
            ("source_fund_code", record.source_fund_code),
        ):
            if getattr(share, field_name) != expected_value:
                setattr(share, field_name, expected_value)
                changed = True
        if share.share_class == "UNSPECIFIED" and record.share_class != "UNSPECIFIED":
            share.share_class = record.share_class
            changed = True
        if changed:
            updated_count += 1
        else:
            skipped_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def list_active_market_sync_targets(session: Session) -> tuple[MarketSyncTarget, ...]:
    """返回基金市场中全部启用且由 Tushare 维护的份额，不读取用户关注列表。"""
    rows = session.execute(
        select(FundShareClass.fund_code, FundShareClass.source_fund_code)
        .where(FundShareClass.status == "ACTIVE", FundShareClass.source_code == TUSHARE_SOURCE_CODE)
        .order_by(FundShareClass.fund_code.asc())
    ).all()
    return tuple(
        MarketSyncTarget(fund_code=fund_code, source_fund_code=source_fund_code)
        for fund_code, source_fund_code in rows
    )


def assign_source_fund_codes(session: Session, source_fund_codes: dict[str, str]) -> None:
    """为已有市场基金补齐已校验的来源精确代码；未知代码或重复映射均失败关闭。"""
    if not source_fund_codes:
        return
    shares = {
        share.fund_code: share
        for share in session.scalars(
            select(FundShareClass).where(FundShareClass.fund_code.in_(tuple(source_fund_codes)))
        ).all()
    }
    if set(shares) != set(source_fund_codes):
        raise ValueError("market source code resolution contains unknown fund codes")
    if len(set(source_fund_codes.values())) != len(source_fund_codes):
        raise ValueError("market source code resolution contains duplicate source fund codes")
    for fund_code, source_fund_code in source_fund_codes.items():
        share = shares[fund_code]
        if share.source_fund_code not in (None, source_fund_code):
            raise ValueError(f"source fund code conflicts for fund_code={fund_code}")
        share.source_fund_code = source_fund_code


def upsert_nav_daily_batch(
    session: Session, *, source_id: UUID, records: tuple[NavDailyUpsert, ...]
) -> WriteStats:
    """按基金、日期、来源和内容哈希幂等写入净值；未知份额只计数跳过。"""
    if not records:
        return WriteStats()
    fund_codes = tuple(record.fund_code for record in records)
    available_codes = set(
        session.scalars(select(FundShareClass.fund_code).where(FundShareClass.fund_code.in_(fund_codes))).all()
    )
    dates = tuple({record.nav_date for record in records})
    existing_navs = {
        (nav.fund_code, nav.nav_date): nav
        for nav in session.scalars(
            select(NavDaily).where(
                NavDaily.source_id == source_id,
                NavDaily.fund_code.in_(fund_codes),
                NavDaily.nav_date.in_(dates),
            )
        ).all()
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        if record.fund_code not in available_codes:
            skipped_count += 1
            continue
        existing = existing_navs.get((record.fund_code, record.nav_date))
        if existing is None:
            session.add(
                NavDaily(
                    fund_code=record.fund_code,
                    nav_date=record.nav_date,
                    source_id=source_id,
                    unit_nav=record.unit_nav,
                    accumulated_nav=record.accumulated_nav,
                    content_hash=record.content_hash,
                )
            )
            created_count += 1
        elif existing.content_hash == record.content_hash:
            skipped_count += 1
        else:
            existing.unit_nav = record.unit_nav
            existing.accumulated_nav = record.accumulated_nav
            existing.content_hash = record.content_hash
            updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def get_latest_nav_dates(
    session: Session, *, source_id: UUID, fund_codes: tuple[str, ...]
) -> dict[str, date]:
    """按数据源返回指定基金已持久化的最新净值日期。

    增量同步只能以同一来源的数据推进水位，不能让其他来源的记录掩盖
    Tushare 数据缺口。
    """
    if not fund_codes:
        return {}
    rows = session.execute(
        select(NavDaily.fund_code, func.max(NavDaily.nav_date))
        .where(NavDaily.source_id == source_id, NavDaily.fund_code.in_(fund_codes))
        .group_by(NavDaily.fund_code)
    ).all()
    return {fund_code: nav_date for fund_code, nav_date in rows}


def _get_sync_run(session: Session, sync_run_id: UUID) -> SourceSyncRun:
    run = session.get(SourceSyncRun, sync_run_id)
    if run is None:
        raise LookupError(f"source sync run does not exist: {sync_run_id}")
    return run


def _get_source(session: Session, source_id: UUID) -> SourceRegistry:
    source = session.get(SourceRegistry, source_id)
    if source is None:
        raise LookupError(f"source does not exist: {source_id}")
    return source
