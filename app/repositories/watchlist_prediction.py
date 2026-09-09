"""预测状态只读投影：基金/来源、日期元数据和一份已保存研究，不重训、不评分。"""

from datetime import date

from sqlalchemy import Integer, cast, or_, select

from app.models.cash_reinvestment import CashResearchRun
from app.models.fund import FundShareClass, NavDaily, SourceRegistry


def read_prediction_inputs(session, fund_code: str, today: date, *, include_research: bool = True):
    fund = session.execute(
        select(
            FundShareClass.fund_code, FundShareClass.fund_type, FundShareClass.status, FundShareClass.source_code
        ).where(FundShareClass.fund_code == fund_code)
    ).one_or_none()
    if fund is None:
        return None, None, None, None
    source = session.execute(
        select(SourceRegistry.source_id, SourceRegistry.enabled).where(SourceRegistry.source_code == fund.source_code)
    ).one_or_none()
    latest_date = None
    if source is not None and source.enabled:
        # 仅查询日期，不读取或计算保留测试期的净值数值；沿用基金/来源/日期索引。
        latest_date = session.scalar(
            select(NavDaily.nav_date)
            .where(
                NavDaily.fund_code == fund_code,
                NavDaily.source_id == source.source_id,
                NavDaily.nav_date <= today,
                NavDaily.ann_date <= today,
                NavDaily.ann_date >= NavDaily.nav_date,
            )
            .order_by(NavDaily.nav_date.desc())
            .limit(1)
        )
    run = None
    if include_research and fund.fund_type == "STOCK":
        run = find_latest_fund_research(session, fund_code)
    return fund, source, latest_date, run


def find_latest_fund_research(session, fund_code: str):
    """只读研究状态；不参与选择已发布结果，也不按研究成绩挑模型。"""
    counts = CashResearchRun.report["preparation"]["fund_counts"][fund_code]
    return session.scalar(
        select(CashResearchRun)
        .where(or_(cast(counts["TRAIN"].astext, Integer) > 0, cast(counts["VALIDATION"].astext, Integer) > 0))
        .order_by(CashResearchRun.created_at.desc(), CashResearchRun.run_id.desc())
        .limit(1)
    )
