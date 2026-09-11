"""按建议实际生成后的固定区间核验后续表现；只读已存公共资料，不回填或训练。"""

from datetime import UTC, date, datetime
from decimal import Decimal, localcontext

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.models.fund import NavDaily
from app.repositories.cash_prediction_features import read_known_cash_dividends
from app.repositories.cash_reinvestment_samples import CashNavPoint
from app.repositories.historical_nav import read_historical_nav_source
from app.schemas.portfolio_advice import AdviceOutcome
from app.services.direction_nav_data import cash_series
from app.services.direction_training_artifacts import digest
from app.services.trading_calendar import load_prediction_calendar


def read_outcome_inputs(fund_code, dates, today):
    """固定21日净值；最多100条分红由已有仓储校验，来源核验和价格读取使用同一事务快照。"""
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        source = read_historical_nav_source(session, fund_code=fund_code)
        state = (
            session.execute(
                text("SELECT status, dividends_verified_at FROM simulation_market_refresh WHERE fund_code=:code"),
                {"code": fund_code},
            )
            .mappings()
            .first()
        )
        if not state or state["status"] != "SUCCEEDED" or state["dividends_verified_at"] is None:
            return (), (), None, "分红资料尚未核验完成，保留原建议等待核验。"
        if state["dividends_verified_at"].date() < dates[-1]:
            return (), (), None, "观察期结束后的分红资料尚未核验，暂不计算回报。"
        rows = session.execute(
            select(NavDaily.nav_date, NavDaily.ann_date, NavDaily.unit_nav, NavDaily.content_hash)
            .where(
                NavDaily.fund_code == fund_code,
                NavDaily.source_id == source.source_id,
                NavDaily.nav_date.in_(dates),
            )
            .order_by(NavDaily.nav_date)
            .limit(22)
        ).all()
        if len(rows) != 21 or len({r[0] for r in rows}) != 21:
            return (), (), None, "观察期内净值尚未齐全；缺失日期不补值，也不提前判定效果。"
        if any(r[1] is not None and r[1] > today for r in rows):
            return (), (), None, "部分净值的来源公告日尚未到，等待实际公开后核验。"
        # 分红读取仍限制在192日内；过久未完成的记录交给人工核验，不无限扩大查询。
        if (today - dates[0]).days >= 192:
            return (), (), None, "该记录待核验时间过长，需要核对历史分红版本后处理。"
        events = read_known_cash_dividends(
            session, fund_code=fund_code, source_id=source.source_id, start=dates[0], cutoff=today
        )
        versions = {
            "nav_revisions": [r[3] for r in rows],
            "original_ann_dates": [str(r[1]) if r[1] else None for r in rows],
            "dividends_verified_at": state["dividends_verified_at"].isoformat(),
        }
        return tuple(CashNavPoint(*r[:3]) for r in rows), events, versions, None


def get_advice_outcome(fund_code: str, start: date, end: date, *, now: datetime | None = None) -> AdviceOutcome:
    """终点必须恰为起点之后第20个交易日；只在窗口结束且净值可用时计算。"""
    from zoneinfo import ZoneInfo

    now = now or datetime.now(UTC)
    today = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
    calendar = load_prediction_calendar(start)
    if start.year < 2026 or start not in calendar.sessions or end != calendar.future_sessions(start, 20)[-1]:
        raise ValueError("观察区间必须是2026年起、已核验日历内连续20个交易日。")
    fields = {"fund_code": fund_code, "start_date": start, "end_date": end, "checked_at": now}
    if today <= end:
        return AdviceOutcome(**fields, status="WAITING", message="观察期尚未结束，暂不判定建议效果。")
    dates = (start, *calendar.future_sessions(start, 20))
    nav, events, versions, issue = read_outcome_inputs(fund_code, dates, today)
    if issue:
        return AdviceOutcome(**fields, status="DATA_INSUFFICIENT", message=issue)
    series, audit, issues = cash_series(dates, {p.nav_date: p for p in nav}, events, today, calendar=calendar)
    if issues:
        return AdviceOutcome(**fields, status="DATA_INSUFFICIENT", message="净值或分红校验未通过，保留原建议等待核验。")
    with localcontext() as context:
        context.prec = 40
        value = (series[-1] / series[0] - 1).quantize(Decimal("0.000000000001"))
    evidence = {**audit, **versions, "calendar_hash": calendar.content_hash, "basis": "CASH_REINVESTMENT_20D_V1"}
    return AdviceOutcome(
        **fields,
        status="ASSESSED",
        total_return=value,
        message="按已存净值与现金分红再投核验；净值采用下一交易日可用假设，未计申赎费和资金机会成本。",
        evidence_hash=digest(evidence),
        evidence=evidence,
    )
