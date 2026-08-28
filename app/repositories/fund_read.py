"""基金目录与最新有效净值的只读仓储。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import Select, and_, case, func, or_, select, true
from sqlalchemy.orm import Session

from app.models.fund import (
    FundDividend,
    FundManagerAssignment,
    FundProfile,
    FundShareClass,
    FundShareSnapshot,
    NavDaily,
    SourceRegistry,
    SourceSyncRun,
)


@dataclass(frozen=True)
class FundDetailSnapshot:
    """同一条最新净值记录的详情投影，避免净值数值与来源错配。"""

    fund: FundShareClass
    nav_date: date | None
    unit_nav: Decimal | None
    accumulated_nav: Decimal | None
    source_code: str | None
    ann_date: date | None
    accumulated_dividend: Decimal | None
    net_asset: Decimal | None
    total_net_asset: Decimal | None
    adjusted_nav: Decimal | None


@dataclass(frozen=True)
class FundProfileSnapshot:
    """一条按来源代码稳定选择的基金资料快照。"""

    profile: FundProfile
    source_code: str


@dataclass(frozen=True)
class FundManagerDetailSnapshot:
    """一条基金经理任职资料及其来源代码。"""

    assignment: FundManagerAssignment
    source_code: str


@dataclass(frozen=True)
class FundShareDetailSnapshot:
    """最新基金份额规模快照及其来源代码。"""

    snapshot: FundShareSnapshot
    source_code: str


@dataclass(frozen=True)
class FundDividendDetailSnapshot:
    """结构化分红事件及其来源代码。"""

    dividend: FundDividend
    source_code: str


@dataclass(frozen=True)
class FundWatchlistDetailSnapshot:
    """完整详情的仓储投影；不包含任何用户标识或关注关系。"""

    detail: FundDetailSnapshot
    profile: FundProfileSnapshot | None
    managers: tuple[FundManagerDetailSnapshot, ...]
    latest_share: FundShareDetailSnapshot | None
    dividends: tuple[FundDividendDetailSnapshot, ...]
    succeeded_sync_types: frozenset[str]


@dataclass(frozen=True)
class FundNavHistorySnapshot:
    """一条按来源稳定去重后的历史净值投影。"""

    nav_date: date
    unit_nav: Decimal
    accumulated_nav: Decimal | None


@dataclass(frozen=True)
class FundSummaryPage:
    """基金目录分页查询的数据库结果，包含与筛选条件一致的总记录数。"""

    rows: tuple[FundSummarySnapshot, ...]
    total_count: int


@dataclass(frozen=True)
class FundPerformanceSnapshot:
    """基于同一份额累计净值计算的列表展示涨跌率。"""

    day_change_rate: Decimal | None
    week_change_rate: Decimal | None
    month_change_rate: Decimal | None


@dataclass(frozen=True)
class FundSummarySnapshot:
    """基金目录、最新净值日期与列表展示涨跌率的稳定投影。"""

    fund: FundShareClass
    nav_date: date | None
    performance: FundPerformanceSnapshot


_FUND_TYPE_SORT_ORDER = {
    "MONEY": 10,
    "BOND": 20,
    "MIXED": 30,
    "STOCK": 40,
    "INDEX": 50,
    "QDII": 60,
    "FOF": 70,
    "OTHER": 80,
}
_PERFORMANCE_LOOKBACK_DAYS = 45


def list_fund_summaries(
    session: Session,
    keyword: str | None,
    fund_type: str | None,
    page_size: int,
    cursor: str | None,
    page: int | None,
) -> FundSummaryPage:
    """读取已落库目录、最新净值日期与展示涨跌率。

    目录与净值同步分开执行。尚无已授权净值来源时，仍返回经过核验的目录，
    但 latest_nav_date 为 ``None``，由服务层明确标识为未同步。页码模式按基金
    类型连续排序，供界面分组展示；旧游标模式保持基金代码顺序以兼容既有调用方。
    """
    latest_nav = _latest_nav_date_subquery()
    conditions = ()
    if keyword:
        normalized_keyword = f"%{keyword.strip()}%"
        conditions = (
            FundShareClass.fund_code.ilike(normalized_keyword)
            | FundShareClass.fund_name.ilike(normalized_keyword),
        )
    if fund_type:
        conditions += (FundShareClass.fund_type == fund_type,)

    total_count = session.scalar(
        select(func.count()).select_from(FundShareClass).where(*conditions)
    )
    statement: Select[tuple[FundShareClass, date | None]] = (
        select(FundShareClass, latest_nav.c.latest_nav_date)
        .outerjoin(latest_nav, latest_nav.c.fund_code == FundShareClass.fund_code)
        .where(*conditions)
    )
    if page is not None:
        statement = statement.order_by(_fund_type_order(), FundShareClass.fund_code.asc())
        statement = statement.offset((page - 1) * page_size).limit(page_size)
    else:
        if cursor:
            statement = statement.where(FundShareClass.fund_code > cursor)
        statement = statement.order_by(FundShareClass.fund_code.asc())
        statement = statement.limit(page_size + 1)
    rows = tuple(session.execute(statement).all())
    return FundSummaryPage(
        rows=_attach_performance(session, rows),
        total_count=int(total_count or 0),
    )


def list_fund_summaries_by_codes(
    session: Session, fund_codes: tuple[str, ...]
) -> tuple[FundSummarySnapshot, ...]:
    """批量读取指定基金的列表摘要，供 Java 组合当前用户关注页时使用。"""
    if not fund_codes:
        return ()
    latest_nav = _latest_nav_date_subquery()
    rows = tuple(
        session.execute(
            select(FundShareClass, latest_nav.c.latest_nav_date)
            .outerjoin(latest_nav, latest_nav.c.fund_code == FundShareClass.fund_code)
            .where(FundShareClass.fund_code.in_(fund_codes))
            .order_by(FundShareClass.fund_code.asc())
        ).all()
    )
    return _attach_performance(session, rows)


def get_fund_summary(session: Session, fund_code: str) -> FundDetailSnapshot | None:
    """返回最新净值快照；目录不存在时返回 ``None``。

    同一日期可能存在多个已登记来源，按最新净值日期倒序、来源代码正序确定展示记录，
    并从该同一行读取数值与来源，避免跨行拼接出错误的详情。
    """
    latest_nav = (
        select(
            NavDaily.nav_date.label("nav_date"),
            NavDaily.unit_nav.label("unit_nav"),
            NavDaily.accumulated_nav.label("accumulated_nav"),
            SourceRegistry.source_code.label("source_code"),
            NavDaily.ann_date.label("ann_date"),
            NavDaily.accumulated_dividend.label("accumulated_dividend"),
            NavDaily.net_asset.label("net_asset"),
            NavDaily.total_net_asset.label("total_net_asset"),
            NavDaily.adjusted_nav.label("adjusted_nav"),
        )
        .select_from(NavDaily)
        .join(SourceRegistry, SourceRegistry.source_id == NavDaily.source_id)
        .where(NavDaily.fund_code == fund_code)
        .order_by(NavDaily.nav_date.desc(), SourceRegistry.source_code.asc())
        .limit(1)
        .subquery()
    )
    row = session.execute(
        select(
            FundShareClass,
            latest_nav.c.nav_date,
            latest_nav.c.unit_nav,
            latest_nav.c.accumulated_nav,
            latest_nav.c.source_code,
            latest_nav.c.ann_date,
            latest_nav.c.accumulated_dividend,
            latest_nav.c.net_asset,
            latest_nav.c.total_net_asset,
            latest_nav.c.adjusted_nav,
        )
        .outerjoin(latest_nav, true())
        .where(FundShareClass.fund_code == fund_code)
    ).one_or_none()
    if row is None:
        return None
    (
        fund,
        nav_date,
        unit_nav,
        accumulated_nav,
        source_code,
        ann_date,
        accumulated_dividend,
        net_asset,
        total_net_asset,
        adjusted_nav,
    ) = row
    return FundDetailSnapshot(
        fund=fund,
        nav_date=nav_date,
        unit_nav=unit_nav,
        accumulated_nav=accumulated_nav,
        source_code=source_code,
        ann_date=ann_date,
        accumulated_dividend=accumulated_dividend,
        net_asset=net_asset,
        total_net_asset=total_net_asset,
        adjusted_nav=adjusted_nav,
    )


def get_fund_watchlist_detail(session: Session, fund_code: str) -> FundWatchlistDetailSnapshot | None:
    """返回完整详情的本地快照；仅由 Java 完成用户关注关系校验后调用。"""
    detail = get_fund_summary(session, fund_code)
    if detail is None:
        return None

    profile = get_fund_profile_snapshot(session, fund_code)

    manager_rows = session.execute(
        select(FundManagerAssignment, SourceRegistry.source_code)
        .join(SourceRegistry, SourceRegistry.source_id == FundManagerAssignment.source_id)
        .where(FundManagerAssignment.fund_code == fund_code)
        .order_by(
            FundManagerAssignment.begin_date.desc().nulls_last(),
            FundManagerAssignment.ann_date.desc().nulls_last(),
            FundManagerAssignment.source_record_key.asc(),
            SourceRegistry.source_code.asc(),
        )
        .limit(50)
    ).all()
    managers = tuple(
        FundManagerDetailSnapshot(assignment=assignment, source_code=source_code)
        for assignment, source_code in manager_rows
    )

    share_row = session.execute(
        select(FundShareSnapshot, SourceRegistry.source_code)
        .join(SourceRegistry, SourceRegistry.source_id == FundShareSnapshot.source_id)
        .where(FundShareSnapshot.fund_code == fund_code)
        .order_by(FundShareSnapshot.trade_date.desc(), SourceRegistry.source_code.asc())
        .limit(1)
    ).one_or_none()
    latest_share = (
        FundShareDetailSnapshot(snapshot=share_row[0], source_code=share_row[1]) if share_row is not None else None
    )

    dividend_rows = session.execute(
        select(FundDividend, SourceRegistry.source_code)
        .join(SourceRegistry, SourceRegistry.source_id == FundDividend.source_id)
        .where(FundDividend.fund_code == fund_code)
        .order_by(
            FundDividend.ann_date.desc().nulls_last(),
            FundDividend.implementation_ann_date.desc().nulls_last(),
            FundDividend.source_event_key.asc(),
            SourceRegistry.source_code.asc(),
        )
        .limit(100)
    ).all()
    dividends = tuple(
        FundDividendDetailSnapshot(dividend=dividend, source_code=source_code)
        for dividend, source_code in dividend_rows
    )
    succeeded_sync_types = frozenset(
        session.scalars(
            select(SourceSyncRun.sync_type)
            .where(
                SourceSyncRun.status == "SUCCEEDED",
                SourceSyncRun.sync_type.in_(
                    (
                        "MARKET_DETAIL_MANAGER",
                        "MARKET_DETAIL_SHARE",
                        "MARKET_DETAIL_DIVIDEND",
                    )
                ),
            )
            .distinct()
        ).all()
    )
    return FundWatchlistDetailSnapshot(
        detail=detail,
        profile=profile,
        managers=managers,
        latest_share=latest_share,
        dividends=dividends,
        succeeded_sync_types=succeeded_sync_types,
    )


def get_fund_profile_snapshot(session: Session, fund_code: str) -> FundProfileSnapshot | None:
    """按来源代码稳定选择一条当前基础资料快照。"""
    profile_row = session.execute(
        select(FundProfile, SourceRegistry.source_code)
        .join(SourceRegistry, SourceRegistry.source_id == FundProfile.source_id)
        .where(FundProfile.fund_code == fund_code)
        .order_by(SourceRegistry.source_code.asc())
        .limit(1)
    ).one_or_none()
    return (
        FundProfileSnapshot(profile=profile_row[0], source_code=profile_row[1]) if profile_row is not None else None
    )


def list_fund_nav_history(
    session: Session, fund_code: str, start_date: date, end_date: date
) -> tuple[FundNavHistorySnapshot, ...]:
    """按日期正序读取指定窗口历史净值，同日期多来源时按来源代码稳定选取一条。"""
    ranked_nav = (
        select(
            NavDaily.nav_date.label("nav_date"),
            NavDaily.unit_nav.label("unit_nav"),
            NavDaily.accumulated_nav.label("accumulated_nav"),
            func.row_number()
            .over(partition_by=NavDaily.nav_date, order_by=SourceRegistry.source_code.asc())
            .label("source_rank"),
        )
        .select_from(NavDaily)
        .join(SourceRegistry, SourceRegistry.source_id == NavDaily.source_id)
        .where(
            NavDaily.fund_code == fund_code,
            NavDaily.nav_date >= start_date,
            NavDaily.nav_date <= end_date,
        )
        .subquery()
    )
    rows = session.execute(
        select(ranked_nav.c.nav_date, ranked_nav.c.unit_nav, ranked_nav.c.accumulated_nav)
        .where(ranked_nav.c.source_rank == 1)
        .order_by(ranked_nav.c.nav_date.asc())
    ).all()
    return tuple(
        FundNavHistorySnapshot(nav_date=nav_date, unit_nav=unit_nav, accumulated_nav=accumulated_nav)
        for nav_date, unit_nav, accumulated_nav in rows
    )


def _latest_nav_date_subquery():
    """返回每只基金的最近净值日期；净值数值另按来源确定性规则读取。"""
    return (
        select(NavDaily.fund_code, func.max(NavDaily.nav_date).label("latest_nav_date"))
        .group_by(NavDaily.fund_code)
        .subquery()
    )


def _fund_type_order():
    """定义市场与关注列表共同使用的基金类型展示顺序。"""
    return case(_FUND_TYPE_SORT_ORDER, value=FundShareClass.fund_type, else_=99)


def _attach_performance(
    session: Session, rows: tuple[tuple[FundShareClass, date | None], ...]
) -> tuple[FundSummarySnapshot, ...]:
    """一次批量查询当前页的近期净值，并组装各基金展示涨跌率。"""
    latest_date_by_code = {
        fund.fund_code: nav_date
        for fund, nav_date in rows
        if nav_date is not None
    }
    points_by_code = _list_recent_nav_points(session, latest_date_by_code)
    return tuple(
        FundSummarySnapshot(
            fund=fund,
            nav_date=nav_date,
            performance=_build_performance(points_by_code.get(fund.fund_code, ())),
        )
        for fund, nav_date in rows
    )


def _list_recent_nav_points(
    session: Session, latest_date_by_code: dict[str, date]
) -> dict[str, tuple[FundNavHistorySnapshot, ...]]:
    """批量读取每只基金最近 45 天的确定性净值点，避免按行或按基金查询。"""
    if not latest_date_by_code:
        return {}
    window_conditions = tuple(
        and_(
            NavDaily.fund_code == fund_code,
            NavDaily.nav_date >= latest_date - timedelta(days=_PERFORMANCE_LOOKBACK_DAYS),
            NavDaily.nav_date <= latest_date,
        )
        for fund_code, latest_date in latest_date_by_code.items()
    )
    ranked_nav = (
        select(
            NavDaily.fund_code.label("fund_code"),
            NavDaily.nav_date.label("nav_date"),
            NavDaily.unit_nav.label("unit_nav"),
            NavDaily.accumulated_nav.label("accumulated_nav"),
            func.row_number()
            .over(
                partition_by=(NavDaily.fund_code, NavDaily.nav_date),
                order_by=SourceRegistry.source_code.asc(),
            )
            .label("source_rank"),
        )
        .select_from(NavDaily)
        .join(SourceRegistry, SourceRegistry.source_id == NavDaily.source_id)
        .where(or_(*window_conditions))
        .subquery()
    )
    rows = session.execute(
        select(
            ranked_nav.c.fund_code,
            ranked_nav.c.nav_date,
            ranked_nav.c.unit_nav,
            ranked_nav.c.accumulated_nav,
        )
        .where(ranked_nav.c.source_rank == 1)
        .order_by(ranked_nav.c.fund_code.asc(), ranked_nav.c.nav_date.asc())
    ).all()
    points_by_code: dict[str, list[FundNavHistorySnapshot]] = {}
    for fund_code, nav_date, unit_nav, accumulated_nav in rows:
        points_by_code.setdefault(fund_code, []).append(
            FundNavHistorySnapshot(
                nav_date=nav_date,
                unit_nav=unit_nav,
                accumulated_nav=accumulated_nav,
            )
        )
    return {fund_code: tuple(points) for fund_code, points in points_by_code.items()}


def _build_performance(points: tuple[FundNavHistorySnapshot, ...]) -> FundPerformanceSnapshot:
    """按上一净值日、近七日和近三十日的可用基准计算基金涨跌率。"""
    if not points:
        return FundPerformanceSnapshot(None, None, None)
    latest = points[-1]
    day_base = points[-2] if len(points) >= 2 else None
    week_base = next(
        (point for point in reversed(points) if point.nav_date <= latest.nav_date - timedelta(days=7)),
        None,
    )
    month_base = next(
        (point for point in reversed(points) if point.nav_date <= latest.nav_date - timedelta(days=30)),
        None,
    )
    return FundPerformanceSnapshot(
        day_change_rate=_calculate_change_rate(latest, day_base),
        week_change_rate=_calculate_change_rate(latest, week_base),
        month_change_rate=_calculate_change_rate(latest, month_base),
    )


def _calculate_change_rate(
    latest: FundNavHistorySnapshot, base: FundNavHistorySnapshot | None
) -> Decimal | None:
    """优先使用累计净值；两端均缺累计净值时才使用单位净值。"""
    if base is None:
        return None
    if latest.accumulated_nav is not None and base.accumulated_nav is not None:
        latest_value = latest.accumulated_nav
        base_value = base.accumulated_nav
    else:
        latest_value = latest.unit_nav
        base_value = base.unit_nav
    if base_value == 0:
        return None
    return latest_value / base_value - Decimal("1")
