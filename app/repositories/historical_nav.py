"""从数据库准备一道历史练习题需要的数据，不计算收益，也不自动补数。

读取顺序：基金身份 → 可用来源 → 指定起点 → 起点前最多60条已知净值 → 起点后20条净值。
“最多81条”是本次读取的净值记录上限，不包括前面检查基金、来源时读取的元数据。
"""

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.fund import FundShareClass, NavDaily
from app.repositories.feature_snapshot import get_enabled_feature_source
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

    # 3. 起点、历史、未来都复用相同基金+来源条件，避免把两家来源的净值拼成一条序列。
    # 此处只是组装SQL，还没查询；四列顺序与HistoricalNavPoint的四个构造参数一致。
    nav_query = select(NavDaily.nav_date, NavDaily.ann_date, NavDaily.unit_nav, NavDaily.accumulated_nav).where(
        NavDaily.fund_code == fund_code, NavDaily.source_id == source.source_id
    )
    anchor = session.execute(nav_query.where(NavDaily.nav_date == as_of_date)).one_or_none()
    if anchor is None:
        raise HistoricalNavPreviewReadError("NAV_NOT_FOUND", "数据库中没有该基金在指定日期的净值，请换一个净值日。")

    # *anchor将四列展开为对象参数；末尾逗号表示这是“只有一条记录的tuple”，不是普通括号。
    # 公告缺失/异常时只返回起点，让计算层统一输出DATA_INSUFFICIENT，不猜测公告日期。
    points = (HistoricalNavPoint(*anchor),)
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
        fund_type=fund.fund_type,
        source_code=source.source_code,
        source_sync_run_id=source.source_sync_run_id,
        nav_points=points,
    )
