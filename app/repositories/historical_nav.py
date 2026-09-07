"""从数据库准备一道历史练习题需要的数据，不计算收益，也不自动补数。

读取顺序：基金身份 → 可用来源 → 起点及是否过时 → 起点前最多60条已知净值 → 起点后20条净值。
“最多81条”是本次读取的净值记录上限，不包括前面检查基金、来源时读取的元数据。
批量读取对每份题目保持同样的81条上限；按起点日期分页，每页合并查询计算资料。
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy import literal, select, union_all
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import ColumnElement

from app.models.fund import FundShareClass, NavDaily
from app.repositories.feature_snapshot import FeatureSourceReadiness, get_enabled_feature_source
from app.services.historical_nav_samples import (
    HORIZON_TRADING_DAYS,
    MIN_FEATURE_NAV_OBSERVATIONS,
    HistoricalNavPoint,
    HistoricalNavSampleInput,
)


class HistoricalNavPreviewReadError(RuntimeError):
    """基金、来源或目标日期不能用于预览时的可解释业务错误。"""

    def __init__(self, code: str, message: str) -> None:
        """code供HTTP入口判断错误类别；message是可以让用户直接阅读的原因说明。"""
        # 人话说明保存在异常自身，后续str(error)就能取出。
        super().__init__(message)
        # 稳定错误码，如NAV_NOT_FOUND（缺指定日净值），不是数据库异常堆栈或HTTP状态码。
        self.code = code


def read_historical_nav_source(session: Session, *, fund_code: str) -> FeatureSourceReadiness:
    """单日和批量共用的入口检查；批量请求只检查一次，不在每页重新选来源。"""
    # 1. 从基金目录读真实品类、启用状态和来源，不能仅凭调用方自称“股票型”就接受。
    # select只取需要的字段；one_or_none表示期望唯一一行，没有则为None。
    fund = session.execute(
        select(FundShareClass.fund_type, FundShareClass.status, FundShareClass.source_code)
        .where(FundShareClass.fund_code == fund_code)
    ).one_or_none()
    if fund is None:
        raise HistoricalNavPreviewReadError("FUND_NOT_FOUND", "数据库中没有这只基金。")
    if fund.fund_type != "STOCK" or fund.status != "ACTIVE":
        raise HistoricalNavPreviewReadError("NOT_APPLICABLE", "当前预览只支持启用的股票型基金。")
    # 2. 复用既有来源就绪检查：来源必须启用，且有完成的净值同步记录。
    # 返回当前来源水位，不意味着这条历史净值就是在该次同步中首次公开的。
    source = get_enabled_feature_source(session, fund.source_code)
    if source is None:
        raise HistoricalNavPreviewReadError("SOURCE_NOT_READY", "该基金的净值来源未启用或尚无成功同步记录。")
    return source


def _has_newer_announced_nav() -> ColumnElement[bool]:
    """给起点查询附带一个是/否：同基金、同来源是否已有更新净值在该截止日公布。

    aliased给同一张表起另一个查询名字；外层是正在出题的起点，内层是寻找的更新记录。
    EXISTS只检查是否存在，不取出全部记录，也不额外为每个样本发送一次数据库请求。
    """
    newer = aliased(NavDaily)
    return (
        select(literal(1))
        .where(
            newer.fund_code == NavDaily.fund_code,
            newer.source_id == NavDaily.source_id,
            newer.nav_date > NavDaily.nav_date,
            # 有效公告不能早于净值所属日，所以搜索日期只需到起点公告日，避免扫描更远未来。
            newer.nav_date <= NavDaily.ann_date,
            newer.ann_date >= newer.nav_date,
            newer.ann_date <= NavDaily.ann_date,
        )
        .correlate(NavDaily)
        .exists()
        .label("has_newer_announced_nav")
    )


def read_historical_nav_sample_input(
    session: Session, *, fund_code: str, as_of_date: date
) -> HistoricalNavSampleInput:
    """读取同一启用来源的最多60条已知历史、1条起点和20条未来净值。

    Args:
        session: 调用者持有的只读、一致性事务。
        fund_code: 基金份额代码。
        as_of_date: 必须精确匹配的起点净值业务日，不向相邻日期回退。

    Raises:
        HistoricalNavPreviewReadError: 基金/净值不存在、基金不适用或来源未就绪。
    """
    source = read_historical_nav_source(session, fund_code=fund_code)

    # 3. 起点、历史、未来都复用相同基金+来源条件，避免把两家来源的净值拼成一条序列。
    # 此处只是组装SQL，还没查询；四列顺序与HistoricalNavPoint的四个构造参数一致。
    nav_query = select(NavDaily.nav_date, NavDaily.ann_date, NavDaily.unit_nav, NavDaily.accumulated_nav).where(
        NavDaily.fund_code == fund_code, NavDaily.source_id == source.source_id
    )
    anchor = session.execute(
        nav_query.add_columns(_has_newer_announced_nav()).where(NavDaily.nav_date == as_of_date)
    ).one_or_none()
    if anchor is None:
        raise HistoricalNavPreviewReadError("NAV_NOT_FOUND", "数据库中没有该基金在指定日期的净值，请换一个净值日。")

    # 前四列才是净值，新增的是/否列不混进数值对象；末尾逗号表示只有一条记录的tuple。
    # 公告缺失/异常时只返回起点，让计算层统一输出DATA_INSUFFICIENT，不猜测公告日期。
    points = (HistoricalNavPoint(*anchor[:4]),)
    if anchor.ann_date is not None and anchor.ann_date >= anchor.nav_date:
        # 4. 只筛选当时可见的历史：业务日早于起点，公告不早于业务日，且不晚于起点公告日。
        # 先按日期倒序LIMIT 60，拿到最近的60条；后面再反转成计算需要的时间正序。
        history = session.execute(
            nav_query.where(
                NavDaily.nav_date < as_of_date,
                NavDaily.ann_date >= NavDaily.nav_date,
                NavDaily.ann_date <= anchor.ann_date,
            )
            .order_by(NavDaily.nav_date.desc())
            .limit(MIN_FEATURE_NAV_OBSERVATIONS - 1)
        ).all()
        # 5. 未来按日期正序直接取接下来的20条。不能提前过滤缺公告/坏值的记录，
        # 否则本来第20条无效时可能悄悄用第21条作终点；应交由计算层明确拒收。
        future = session.execute(
            nav_query.where(NavDaily.nav_date > as_of_date)
            .order_by(NavDaily.nav_date.asc())
            .limit(HORIZON_TRADING_DAYS)
        ).all()
        # .all()只取得上述LIMIT约束下的结果，并非读取全库。
        # 按“历史60条 + 起点1条 + 未来20条”拼接，不把基金的全部历史装进内存。
        points = (
            tuple(HistoricalNavPoint(*row) for row in reversed(history))
            + points
            + tuple(HistoricalNavPoint(*row) for row in future)
        )
    # 6. 把数据库结果装成纯数据对象。计算层接收它即可，不需要了解SQL或数据库连接。
    return HistoricalNavSampleInput(
        fund_code=fund_code,
        fund_type="STOCK",
        source_code=source.source_code,
        source_sync_run_id=source.source_sync_run_id,
        nav_points=points,
        stale_anchor_dates=frozenset({as_of_date}) if anchor.has_newer_announced_nav else frozenset(),
    )


@dataclass(frozen=True)
class HistoricalNavSampleWindow:
    """一份练习题的完整计算资料；不是数据库表，也不是已经计算完的答案。"""

    # 本份资料要计算哪一天；它不一定是 nav_points 的第一天。
    as_of_date: date
    # 包含最多 60 条历史、1 条起点和20条未来；不足时原样交给构建器报告原因。
    input_record: HistoricalNavSampleInput


def read_historical_nav_input_page(
    session: Session,
    *,
    fund_code: str,
    source: FeatureSourceReadiness,
    start_date: date,
    end_date: date,
    after_date: date | None,
    page_size: int,
) -> tuple[HistoricalNavSampleWindow, ...]:
    """读取一页样本起点及各自的完整窗口，非空页最多执行两条净值查询。

    start_date/end_date 均包含，限制的是“要出题的日期”，不是计算资料的日期。
    after_date 是上一页最后一天（游标）；下一页严格从它之后开始，不用 OFFSET 跳行。
    page_size 只控制每页起点数量，不能改变某个起点的计算窗口。
    """
    # 仓储层也保留硬上限，避免内部调用绕过 HTTP 校验后构造过大的合并查询。
    if not 1 <= page_size <= 30 or not 0 <= (end_date - start_date).days < 31:
        raise ValueError("批量预览最多31个自然日，每页1至30个样本起点。")
    nav_query = select(NavDaily.nav_date, NavDaily.ann_date, NavDaily.unit_nav, NavDaily.accumulated_nav).where(
        NavDaily.fund_code == fund_code, NavDaily.source_id == source.source_id
    )
    anchor_query = nav_query.add_columns(_has_newer_announced_nav()).where(
        NavDaily.nav_date >= start_date, NavDaily.nav_date <= end_date
    )
    if after_date is not None:
        anchor_query = anchor_query.where(NavDaily.nav_date > after_date)
    # 第一条 SQL：只取这一页的起点及过时标记。EXISTS不受页边界或未来20条窗口限制。
    anchors = session.execute(anchor_query.order_by(NavDaily.nav_date).limit(page_size)).all()
    if not anchors:
        return ()

    points_by_date = {row.nav_date: [HistoricalNavPoint(*row[:4])] for row in anchors}
    window_queries = []
    for anchor in anchors:
        # 与单日 GET 一致：起点公告无效时只保留起点，让构建器解释原因。
        if anchor.ann_date is None or anchor.ann_date < anchor.nav_date:
            continue
        # 给查询结果贴上“属于哪份题目”的日期标签。同一条净值可以服务于不同题目。
        tagged_query = nav_query.add_columns(literal(anchor.nav_date).label("sample_date"))
        history = (
            tagged_query.where(
                NavDaily.nav_date < anchor.nav_date,
                NavDaily.ann_date >= NavDaily.nav_date,
                NavDaily.ann_date <= anchor.ann_date,
            )
            .order_by(NavDaily.nav_date.desc())
            .limit(MIN_FEATURE_NAV_OBSERVATIONS - 1)
            .subquery()
        )
        future = (
            tagged_query.where(NavDaily.nav_date > anchor.nav_date)
            .order_by(NavDaily.nav_date)
            .limit(HORIZON_TRADING_DAYS)
            .subquery()
        )
        # 这里仅组装 SQL，还没访问数据库。每个小查询先独立取最近60/后20条，再合并。
        # 未来不按公告日或数值筛掉坏记录，否则可能把第21条错当成第20条。
        window_queries.extend((select(history), select(future)))
    if window_queries:
        # 第二条 SQL：UNION ALL 把本页所有窗口合并返回，最多 page_size × 80 行。
        # 不去重：同一净值在不同样本中的用途不同；组内之后按日期恢复计算顺序。
        for row in session.execute(union_all(*window_queries)):
            points_by_date[row.sample_date].append(HistoricalNavPoint(*row[:4]))
    return tuple(
        HistoricalNavSampleWindow(
            as_of_date=anchor.nav_date,
            input_record=HistoricalNavSampleInput(
                fund_code=fund_code,
                fund_type="STOCK",
                source_code=source.source_code,
                source_sync_run_id=source.source_sync_run_id,
                nav_points=tuple(sorted(points_by_date[anchor.nav_date], key=lambda point: point.nav_date)),
                # 标记只对应本份资料的目标日，不能误把它传给同一窗口中的辅助日期。
                stale_anchor_dates=(
                    frozenset({anchor.nav_date}) if anchor.has_newer_announced_nav else frozenset()
                ),
            ),
        )
        for anchor in anchors
    )
