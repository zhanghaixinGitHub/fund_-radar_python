"""基金目录、净值与来源同步运行记录的数据访问实现。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from app.models.fund import (
    FundDividend,
    FundManagerAssignment,
    FundMaster,
    FundProfile,
    FundShareClass,
    FundShareSnapshot,
    NavDaily,
    SourceRegistry,
    SourceSyncRun,
)

TUSHARE_SOURCE_CODE = "TUSHARE_PRO_FUND"
TUSHARE_SOURCE_DISPLAY_NAME = "Tushare Pro 公募基金数据"
TUSHARE_SOURCE_KIND = "MARKET_DATA"
TUSHARE_SOURCE_LICENSE_SCOPE = (
    "已最小验权的个人 Tushare 2000 积分接口：fund_company、fund_basic、fund_nav、fund_manager、"
    "fund_share、fund_div、fund_daily、daily、daily_basic、index_basic、index_classify、index_daily、"
    "index_weight；仅供本机基金雷达使用，不含 fund_portfolio、fund_adj、新闻或公告。"
)
TUSHARE_SOURCE_RATE_LIMIT_PER_MINUTE = 200
TUSHARE_SOURCE_RETENTION_DAYS = 365
TUSHARE_AUTHORIZED_API_NAMES: tuple[str, ...] = (
    "fund_company",
    "fund_basic",
    "fund_nav",
    "fund_manager",
    "fund_share",
    "fund_div",
    "fund_daily",
    "daily",
    "daily_basic",
    "index_basic",
    "index_classify",
    "index_daily",
    "index_weight",
)


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
    ann_date: date | None = None
    accumulated_dividend: Decimal | None = None
    net_asset: Decimal | None = None
    total_net_asset: Decimal | None = None
    adjusted_nav: Decimal | None = None


@dataclass(frozen=True)
class FundProfileUpsert:
    """一只基金份额当前基础资料的规范化快照。"""

    fund_code: str
    management_company_name: str | None
    custodian_name: str | None
    found_date: date | None
    due_date: date | None
    list_date: date | None
    issue_date: date | None
    delist_date: date | None
    issue_amount: Decimal | None
    management_fee: Decimal | None
    custodian_fee: Decimal | None
    duration_year: Decimal | None
    par_value: Decimal | None
    min_purchase_amount: Decimal | None
    expected_return: Decimal | None
    benchmark: str | None
    invest_type: str | None
    source_fund_type: str | None
    trustee_name: str | None
    purchase_start_date: date | None
    redemption_start_date: date | None
    market: str | None
    content_hash: str


@dataclass(frozen=True)
class FundManagerAssignmentUpsert:
    """基金经理任职历史的一条规范化来源记录。"""

    fund_code: str
    source_record_key: str
    manager_name: str
    ann_date: date | None
    begin_date: date | None
    end_date: date | None
    education: str | None
    content_hash: str


@dataclass(frozen=True)
class FundShareSnapshotUpsert:
    """基金份额规模的一条规范化来源记录。"""

    fund_code: str
    trade_date: date
    fund_share: Decimal
    content_hash: str


@dataclass(frozen=True)
class FundDividendUpsert:
    """基金分红事件的一条规范化来源记录。"""

    fund_code: str
    source_event_key: str
    ann_date: date | None
    implementation_ann_date: date | None
    base_date: date | None
    process_status: str | None
    record_date: date | None
    ex_date: date | None
    pay_date: date | None
    earnings_pay_date: date | None
    nav_ex_date: date | None
    cash_dividend: Decimal | None
    base_unit: Decimal | None
    distributable_earnings: Decimal | None
    earnings_amount: Decimal | None
    reinvestment_arrival_date: date | None
    base_year: str | None
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
            authorized_api_names=list(TUSHARE_AUTHORIZED_API_NAMES),
            authorization_verified_at=datetime.now(UTC),
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
    source.authorized_api_names = list(TUSHARE_AUTHORIZED_API_NAMES)
    source.authorization_verified_at = datetime.now(UTC)
    return source


def create_sync_run(
    session: Session,
    *,
    source_id: UUID,
    sync_type: str,
    requested_nav_date: date | None,
    parent_sync_run_id: UUID | None = None,
    requested_window_start: date | None = None,
    requested_window_end: date | None = None,
    data_as_of_date: date | None = None,
) -> SourceSyncRun:
    """创建 RUNNING 状态的同步记录并返回已持久化的运行标识。"""
    run = SourceSyncRun(
        source_id=source_id,
        parent_sync_run_id=parent_sync_run_id,
        sync_type=sync_type,
        requested_nav_date=requested_nav_date,
        requested_window_start=requested_window_start,
        requested_window_end=requested_window_end,
        data_as_of_date=data_as_of_date,
        status="RUNNING",
    )
    session.add(run)
    session.flush()
    return run


def link_sync_run_to_parent(session: Session, *, sync_run_id: UUID, parent_sync_run_id: UUID) -> None:
    """为既有子运行补充父运行关联，保留原同步类型和统计。"""
    run = _get_sync_run(session, sync_run_id)
    parent = _get_sync_run(session, parent_sync_run_id)
    if run.source_id != parent.source_id:
        raise ValueError("sync run parent must use the same source")
    run.parent_sync_run_id = parent_sync_run_id


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


def get_latest_successful_sync_time(session: Session, *, sync_types: tuple[str, ...]) -> datetime | None:
    """返回指定同步类型最近一次完整成功的结束时间。"""
    if not sync_types:
        return None
    return session.scalar(
        select(SourceSyncRun.finished_at)
        .where(
            SourceSyncRun.sync_type.in_(sync_types),
            SourceSyncRun.status == "SUCCEEDED",
            SourceSyncRun.finished_at.is_not(None),
        )
        .order_by(SourceSyncRun.finished_at.desc())
        .limit(1)
    )


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
                    ann_date=record.ann_date,
                    accumulated_dividend=record.accumulated_dividend,
                    net_asset=record.net_asset,
                    total_net_asset=record.total_net_asset,
                    adjusted_nav=record.adjusted_nav,
                    content_hash=record.content_hash,
                )
            )
            created_count += 1
        elif existing.content_hash == record.content_hash:
            skipped_count += 1
        else:
            existing.unit_nav = record.unit_nav
            existing.accumulated_nav = record.accumulated_nav
            existing.ann_date = record.ann_date
            existing.accumulated_dividend = record.accumulated_dividend
            existing.net_asset = record.net_asset
            existing.total_net_asset = record.total_net_asset
            existing.adjusted_nav = record.adjusted_nav
            existing.content_hash = record.content_hash
            updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def upsert_fund_profiles_batch(
    session: Session, *, source_id: UUID, records: tuple[FundProfileUpsert, ...]
) -> WriteStats:
    """按基金份额和来源幂等写入当前基础资料快照。"""
    if not records:
        return WriteStats()
    fund_codes = tuple({record.fund_code for record in records})
    available_codes = set(
        session.scalars(select(FundShareClass.fund_code).where(FundShareClass.fund_code.in_(fund_codes))).all()
    )
    existing_by_fund_code = {
        profile.fund_code: profile
        for profile in session.scalars(
            select(FundProfile).where(FundProfile.source_id == source_id, FundProfile.fund_code.in_(fund_codes))
        ).all()
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        if record.fund_code not in available_codes:
            skipped_count += 1
            continue
        existing = existing_by_fund_code.get(record.fund_code)
        if existing is None:
            existing = _new_fund_profile(source_id, record)
            session.add(existing)
            existing_by_fund_code[record.fund_code] = existing
            created_count += 1
        elif existing.content_hash == record.content_hash:
            skipped_count += 1
        else:
            _update_fund_profile(existing, record)
            updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def upsert_fund_manager_assignments_batch(
    session: Session, *, source_id: UUID, records: tuple[FundManagerAssignmentUpsert, ...]
) -> WriteStats:
    """按来源稳定记录键幂等写入基金经理任职历史。"""
    if not records:
        return WriteStats()
    fund_codes = tuple({record.fund_code for record in records})
    available_codes = set(
        session.scalars(select(FundShareClass.fund_code).where(FundShareClass.fund_code.in_(fund_codes))).all()
    )
    existing_by_key = {
        (item.fund_code, item.source_record_key): item
        for item in session.scalars(
            select(FundManagerAssignment).where(
                FundManagerAssignment.source_id == source_id, FundManagerAssignment.fund_code.in_(fund_codes)
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
        existing = existing_by_key.get((record.fund_code, record.source_record_key))
        if existing is None:
            existing = FundManagerAssignment(
                fund_code=record.fund_code,
                source_id=source_id,
                source_record_key=record.source_record_key,
                manager_name=record.manager_name,
                ann_date=record.ann_date,
                begin_date=record.begin_date,
                end_date=record.end_date,
                education=record.education,
                content_hash=record.content_hash,
            )
            session.add(existing)
            existing_by_key[(record.fund_code, record.source_record_key)] = existing
            created_count += 1
        elif existing.content_hash == record.content_hash:
            skipped_count += 1
        else:
            existing.manager_name = record.manager_name
            existing.ann_date = record.ann_date
            existing.begin_date = record.begin_date
            existing.end_date = record.end_date
            existing.education = record.education
            existing.content_hash = record.content_hash
            updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def upsert_fund_share_snapshots_batch(
    session: Session, *, source_id: UUID, records: tuple[FundShareSnapshotUpsert, ...]
) -> WriteStats:
    """按基金、变动日期和来源幂等写入基金份额规模历史。"""
    if not records:
        return WriteStats()
    fund_codes = tuple({record.fund_code for record in records})
    dates = tuple({record.trade_date for record in records})
    available_codes = set(
        session.scalars(select(FundShareClass.fund_code).where(FundShareClass.fund_code.in_(fund_codes))).all()
    )
    existing_by_key = {
        (item.fund_code, item.trade_date): item
        for item in session.scalars(
            select(FundShareSnapshot).where(
                FundShareSnapshot.source_id == source_id,
                FundShareSnapshot.fund_code.in_(fund_codes),
                FundShareSnapshot.trade_date.in_(dates),
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
        existing = existing_by_key.get((record.fund_code, record.trade_date))
        if existing is None:
            existing = FundShareSnapshot(
                fund_code=record.fund_code,
                trade_date=record.trade_date,
                source_id=source_id,
                fund_share=record.fund_share,
                content_hash=record.content_hash,
            )
            session.add(existing)
            existing_by_key[(record.fund_code, record.trade_date)] = existing
            created_count += 1
        elif existing.content_hash == record.content_hash:
            skipped_count += 1
        else:
            existing.fund_share = record.fund_share
            existing.content_hash = record.content_hash
            updated_count += 1
    return WriteStats(created_count=created_count, updated_count=updated_count, skipped_count=skipped_count)


def upsert_fund_dividends_batch(
    session: Session, *, source_id: UUID, records: tuple[FundDividendUpsert, ...]
) -> WriteStats:
    """按稳定事件键幂等写入分红事件，并允许来源更新实施状态。"""
    if not records:
        return WriteStats()
    fund_codes = tuple({record.fund_code for record in records})
    available_codes = set(
        session.scalars(select(FundShareClass.fund_code).where(FundShareClass.fund_code.in_(fund_codes))).all()
    )
    existing_by_key = {
        (item.fund_code, item.source_event_key): item
        for item in session.scalars(
            select(FundDividend).where(FundDividend.source_id == source_id, FundDividend.fund_code.in_(fund_codes))
        ).all()
    }
    created_count = 0
    updated_count = 0
    skipped_count = 0
    for record in records:
        if record.fund_code not in available_codes:
            skipped_count += 1
            continue
        existing = existing_by_key.get((record.fund_code, record.source_event_key))
        if existing is None:
            existing = _new_fund_dividend(source_id, record)
            session.add(existing)
            existing_by_key[(record.fund_code, record.source_event_key)] = existing
            created_count += 1
        elif existing.content_hash == record.content_hash:
            skipped_count += 1
        else:
            _update_fund_dividend(existing, record)
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


def _new_fund_profile(source_id: UUID, record: FundProfileUpsert) -> FundProfile:
    return FundProfile(
        fund_code=record.fund_code,
        source_id=source_id,
        management_company_name=record.management_company_name,
        custodian_name=record.custodian_name,
        found_date=record.found_date,
        due_date=record.due_date,
        list_date=record.list_date,
        issue_date=record.issue_date,
        delist_date=record.delist_date,
        issue_amount=record.issue_amount,
        management_fee=record.management_fee,
        custodian_fee=record.custodian_fee,
        duration_year=record.duration_year,
        par_value=record.par_value,
        min_purchase_amount=record.min_purchase_amount,
        expected_return=record.expected_return,
        benchmark=record.benchmark,
        invest_type=record.invest_type,
        source_fund_type=record.source_fund_type,
        trustee_name=record.trustee_name,
        purchase_start_date=record.purchase_start_date,
        redemption_start_date=record.redemption_start_date,
        market=record.market,
        content_hash=record.content_hash,
    )


def _update_fund_profile(profile: FundProfile, record: FundProfileUpsert) -> None:
    for field_name in (
        "management_company_name",
        "custodian_name",
        "found_date",
        "due_date",
        "list_date",
        "issue_date",
        "delist_date",
        "issue_amount",
        "management_fee",
        "custodian_fee",
        "duration_year",
        "par_value",
        "min_purchase_amount",
        "expected_return",
        "benchmark",
        "invest_type",
        "source_fund_type",
        "trustee_name",
        "purchase_start_date",
        "redemption_start_date",
        "market",
        "content_hash",
    ):
        setattr(profile, field_name, getattr(record, field_name))


def _new_fund_dividend(source_id: UUID, record: FundDividendUpsert) -> FundDividend:
    return FundDividend(
        fund_code=record.fund_code,
        source_id=source_id,
        source_event_key=record.source_event_key,
        ann_date=record.ann_date,
        implementation_ann_date=record.implementation_ann_date,
        base_date=record.base_date,
        process_status=record.process_status,
        record_date=record.record_date,
        ex_date=record.ex_date,
        pay_date=record.pay_date,
        earnings_pay_date=record.earnings_pay_date,
        nav_ex_date=record.nav_ex_date,
        cash_dividend=record.cash_dividend,
        base_unit=record.base_unit,
        distributable_earnings=record.distributable_earnings,
        earnings_amount=record.earnings_amount,
        reinvestment_arrival_date=record.reinvestment_arrival_date,
        base_year=record.base_year,
        content_hash=record.content_hash,
    )


def _update_fund_dividend(dividend: FundDividend, record: FundDividendUpsert) -> None:
    for field_name in (
        "ann_date",
        "implementation_ann_date",
        "base_date",
        "process_status",
        "record_date",
        "ex_date",
        "pay_date",
        "earnings_pay_date",
        "nav_ex_date",
        "cash_dividend",
        "base_unit",
        "distributable_earnings",
        "earnings_amount",
        "reinvestment_arrival_date",
        "base_year",
        "content_hash",
    ):
        setattr(dividend, field_name, getattr(record, field_name))


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
