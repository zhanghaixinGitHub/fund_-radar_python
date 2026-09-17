"""按建议实际生成后的固定区间核验后续表现；只读已存公共资料，不回填或训练。"""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext
from typing import Literal
from zoneinfo import ZoneInfo

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.models.benchmark import BenchmarkNavDaily, BenchmarkSeries
from app.models.fund import (
    FundDividend,
    FundManagerAssignment,
    FundShareClass,
    NavDaily,
    SimulationMarketRefresh,
)
from app.repositories.cash_prediction_features import read_known_cash_dividends
from app.repositories.cash_reinvestment_samples import CashNavPoint
from app.repositories.fund_read import (
    FundNavHistorySnapshot,
    FundShareHistorySnapshot,
    get_current_market_same_type_comparison,
    get_fund_profile_snapshot,
    list_fund_share_history,
)
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_source
from app.schemas.portfolio_advice import AdviceOutcome, DiagnosisFactItem, DiagnosisFacts
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


# ---------------------------------------------------------------------------
# 持仓诊断事实（HOLDING_DIAGNOSIS_FACTS_V1）
#
# Python 侧没有用户历史报告基线，只提供事实与可对比的当前状态；各项 verdict 是
# 按设计文档阈值计算的参考结论，与报告基线的最终对比由 Java 侧完成。失败或缺失
# 一律记 INSUFFICIENT 并写明原因，不补造数值。
# ---------------------------------------------------------------------------

CONTROLLED_SAMPLE_NOTE = "受控当前市场样本（ACTIVE、TUSHARE_PRO_FUND 份额），不是全市场同类平均，不得解释为全市场排名"


@dataclass(frozen=True)
class DiagnosisManagerInput:
    """一条经理任职事实的纯数据投影；不携带 ORM 实例，session 关闭后可安全访问。"""

    manager_name: str
    ann_date: date | None
    begin_date: date | None
    end_date: date | None


@dataclass(frozen=True)
class DiagnosisProfileInput:
    """fund_profile 覆盖式快照中诊断所需字段的纯数据投影。"""

    benchmark: str | None
    management_fee: Decimal | None
    custodian_fee: Decimal | None
    updated_at: datetime | None


@dataclass(frozen=True)
class DiagnosisDividendInput:
    """一条分红事件的纯数据投影。"""

    ann_date: date | None
    ex_date: date | None
    cash_dividend: Decimal | None


@dataclass(frozen=True)
class DiagnosisComparisonRow:
    """受控同类样本中一只基金的纯数据投影。"""

    fund_code: str
    nav_date: date | None
    month_change_rate: Decimal | None
    data_source: str | None


@dataclass(frozen=True)
class DiagnosisComparisonInput:
    """同类比较结果的纯数据投影；范围说明与排名口径与 same-type-comparison 一致。"""

    target_status: str
    target_source_code: str
    as_of_date: date | None
    target_month_change_rate: Decimal | None
    rows: tuple[DiagnosisComparisonRow, ...]


@dataclass(frozen=True)
class DiagnosisInputs:
    """一次诊断的全部公共事实输入；同一事务快照读取，只含纯数据，不含任何用户身份或 ORM 实例。"""

    fund_code: str
    fund_type: str
    fund_status: str
    source_code: str
    benchmark_code: str | None
    # 净值来源未通过既有就绪检查时的可读原因；为 None 时 nav_history 有效。
    nav_unavailable_reason: str | None
    nav_history: tuple[FundNavHistorySnapshot, ...]
    managers: tuple[DiagnosisManagerInput, ...]
    shares: tuple[FundShareHistorySnapshot, ...]
    comparison: DiagnosisComparisonInput | None
    profile: DiagnosisProfileInput | None
    benchmark_series_status: str | None
    benchmark_points: tuple[tuple[date, Decimal], ...]
    dividends_verified_at: datetime | None
    dividends: tuple[DiagnosisDividendInput, ...]


def read_diagnosis_inputs(fund_code: str, as_of: date) -> DiagnosisInputs:
    """七项诊断所需的公共事实在同一事务快照读取；只读落库资料，不触发外部同步。

    所有 ORM 属性必须在 session 存活期间取出并投影为纯数据：事务结束后实例过期，
    任何延迟访问都会抛 DetachedInstanceError。
    """
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        fund = session.get(FundShareClass, fund_code)
        if fund is None:
            raise HistoricalNavPreviewReadError("FUND_NOT_FOUND", "数据库中没有这只基金。")
        fund_type, fund_status, source_code, benchmark_code = (
            fund.fund_type,
            fund.status,
            fund.source_code,
            fund.benchmark_code,
        )
        nav_history: tuple[FundNavHistorySnapshot, ...] = ()
        nav_reason = None
        try:
            source = read_historical_nav_source(session, fund_code=fund_code)
        except HistoricalNavPreviewReadError as error:
            # 单项独立成立：净值类项目记 INSUFFICIENT，经理/规模/费用/分红不受影响。
            nav_reason = str(error)
        else:
            nav_history = tuple(
                FundNavHistorySnapshot(
                    nav_date=row.nav_date, unit_nav=row.unit_nav, accumulated_nav=row.accumulated_nav
                )
                for row in session.execute(
                    select(NavDaily.nav_date, NavDaily.unit_nav, NavDaily.accumulated_nav)
                    .where(
                        NavDaily.fund_code == fund_code,
                        NavDaily.source_id == source.source_id,
                        NavDaily.nav_date <= as_of,
                    )
                    .order_by(NavDaily.nav_date)
                ).all()
            )
        managers = tuple(
            DiagnosisManagerInput(
                manager_name=row.manager_name,
                ann_date=row.ann_date,
                begin_date=row.begin_date,
                end_date=row.end_date,
            )
            for row in session.execute(
                select(FundManagerAssignment)
                .where(FundManagerAssignment.fund_code == fund_code)
                .order_by(
                    FundManagerAssignment.begin_date.asc().nulls_last(),
                    FundManagerAssignment.ann_date.asc().nulls_last(),
                )
                .limit(50)
            )
            .scalars()
            .all()
        )
        shares = list_fund_share_history(session, fund_code, date(1970, 1, 1), as_of)
        comparison_result = get_current_market_same_type_comparison(session, fund_code)
        comparison = None
        if comparison_result is not None:
            target = comparison_result.target
            comparison = DiagnosisComparisonInput(
                target_status=target.fund.status,
                target_source_code=target.fund.source_code,
                as_of_date=target.nav_date,
                target_month_change_rate=target.performance.month_change_rate,
                rows=tuple(
                    DiagnosisComparisonRow(
                        fund_code=item.summary.fund.fund_code,
                        nav_date=item.summary.nav_date,
                        month_change_rate=item.summary.performance.month_change_rate,
                        data_source=item.data_source,
                    )
                    for item in comparison_result.items
                ),
            )
        profile_snapshot = get_fund_profile_snapshot(session, fund_code)
        profile = None
        if profile_snapshot is not None:
            profile_row = profile_snapshot.profile
            profile = DiagnosisProfileInput(
                benchmark=profile_row.benchmark,
                management_fee=profile_row.management_fee,
                custodian_fee=profile_row.custodian_fee,
                updated_at=profile_row.updated_at,
            )
        benchmark_status = None
        benchmark_points: tuple[tuple[date, Decimal], ...] = ()
        if benchmark_code:
            series = session.get(BenchmarkSeries, benchmark_code)
            benchmark_status = series.status if series is not None else None
            benchmark_points = tuple(
                (row.nav_date, row.closing_value)
                for row in session.execute(
                    select(BenchmarkNavDaily.nav_date, BenchmarkNavDaily.closing_value)
                    .where(
                        BenchmarkNavDaily.benchmark_code == benchmark_code,
                        BenchmarkNavDaily.nav_date <= as_of,
                    )
                    .order_by(BenchmarkNavDaily.nav_date)
                ).all()
            )
        refresh = session.get(SimulationMarketRefresh, fund_code)
        dividends_verified_at = refresh.dividends_verified_at if refresh is not None else None
        dividends = tuple(
            DiagnosisDividendInput(ann_date=row.ann_date, ex_date=row.ex_date, cash_dividend=row.cash_dividend)
            for row in session.execute(
                select(FundDividend)
                .where(FundDividend.fund_code == fund_code)
                .order_by(FundDividend.ann_date.asc().nulls_last())
                .limit(200)
            )
            .scalars()
            .all()
        )
    return DiagnosisInputs(
        fund_code=fund_code,
        fund_type=fund_type,
        fund_status=fund_status,
        source_code=source_code,
        benchmark_code=benchmark_code,
        nav_unavailable_reason=nav_reason,
        nav_history=nav_history,
        managers=managers,
        shares=shares,
        comparison=comparison,
        profile=profile,
        benchmark_series_status=benchmark_status,
        benchmark_points=benchmark_points,
        dividends_verified_at=dividends_verified_at,
        dividends=dividends,
    )


def get_diagnosis_facts(
    fund_code: str, as_of_date: date | None = None, *, now: datetime | None = None
) -> DiagnosisFacts:
    """逐项计算七项诊断事实；asOfDate 缺省按当前北京时间日期口径，与 outcome 接口一致。"""
    if as_of_date is None:
        now = now or datetime.now(UTC)
        as_of = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
    else:
        as_of = as_of_date
    inputs = read_diagnosis_inputs(fund_code, as_of)
    items = (
        _manager_item(inputs, as_of),
        _scale_item(inputs, as_of),
        _same_type_item(inputs),
        _benchmark_item(inputs, as_of),
        _drawdown_item(inputs),
        _fee_item(inputs),
        _dividend_item(inputs, as_of),
    )
    verdicts = {item.verdict for item in items}
    overall: Literal["VALID", "CHANGED", "INSUFFICIENT"] = (
        "CHANGED" if "CHANGED" in verdicts else "INSUFFICIENT" if "INSUFFICIENT" in verdicts else "VALID"
    )
    return DiagnosisFacts(fund_code=fund_code, as_of_date=as_of, overall=overall, items=items)


def _insufficient(
    item: Literal["MANAGER", "SCALE", "SAME_TYPE_RANK", "BENCHMARK", "DRAWDOWN", "FEE", "DIVIDEND"],
    source: str,
    evidence: str,
    data_as_of_date: date | None = None,
    facts: dict | None = None,
) -> DiagnosisFactItem:
    return DiagnosisFactItem(
        item=item, verdict="INSUFFICIENT", evidence=evidence, source=source,
        data_as_of_date=data_as_of_date, facts=facts,
    )


def _q6(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.000001"))


def _manager_item(inputs: DiagnosisInputs, as_of: date) -> DiagnosisFactItem:
    visible = tuple(a for a in inputs.managers if a.ann_date is None or a.ann_date <= as_of)
    if not visible:
        return _insufficient("MANAGER", "fund_manager_assignment", "无基金经理任职记录。")
    current = sorted({(a.manager_name, a.begin_date) for a in visible if a.end_date is None})
    begin_dates = [a.begin_date for a in visible if a.begin_date is not None]
    ann_dates = [a.ann_date for a in visible if a.ann_date is not None]
    latest_change = max(begin_dates) if begin_dates else None
    latest_ann = max(ann_dates) if ann_dates else None
    managers_text = "、".join(f"{name}（{begin} 起）" if begin else str(name) for name, begin in current)
    evidence = (
        f"当前在任经理组合：{managers_text or '无'}；最近任职变更日 {latest_change}，最近公告日 {latest_ann}。"
        "经理变更本身不等于管理能力恶化；是否与报告基线不一致由 Java 侧对比判定。"
    )
    facts = {
        "current_managers": [{"manager_name": name, "begin_date": begin} for name, begin in current],
        "latest_change_date": latest_change,
        "latest_ann_date": latest_ann,
    }
    return DiagnosisFactItem(
        item="MANAGER", verdict="VALID", evidence=evidence, source="fund_manager_assignment",
        data_as_of_date=latest_ann or latest_change, facts=facts,
    )


def _scale_item(inputs: DiagnosisInputs, as_of: date) -> DiagnosisFactItem:
    shares = tuple(s for s in inputs.shares if s.trade_date <= as_of)
    if not shares:
        return _insufficient("SCALE", "fund_share_snapshot", "无规模历史记录。")
    latest = shares[-1]
    previous = shares[-2] if len(shares) > 1 else None
    change_ratio = None
    if previous is not None and previous.fund_share > 0:
        change_ratio = _q6(latest.fund_share / previous.fund_share - 1)
    window_start = as_of - timedelta(days=365)
    year = tuple(s for s in shares if s.trade_date >= window_start) or shares[-1:]
    high = max(year, key=lambda s: (s.fund_share, s.trade_date))
    low = min(year, key=lambda s: (s.fund_share, s.trade_date))
    verdict: Literal["VALID", "CHANGED"] = "VALID"
    if previous is None:
        note = "仅一条规模快照，无法计算相对上一时点的变化。"
    elif change_ratio is None:
        note = "上一时点规模为零，变化率不可计算，不补造数值。"
    elif change_ratio >= 1:
        verdict, note = "CHANGED", "达到文档 +100% 暴涨参考阈值。"
    elif change_ratio <= Decimal("-0.5"):
        verdict, note = "CHANGED", "达到文档 −50% 腰斩参考阈值。"
    else:
        note = "未达到文档参考阈值（+100% / −50%）。"
    change_text = (
        f"相对上一可得时点 {previous.trade_date} 的 {previous.fund_share} 份变化 {change_ratio:+.2%}；"
        if previous is not None and change_ratio is not None
        else ""
    )
    evidence = (
        f"最新规模 {latest.fund_share} 份（{latest.trade_date}）；{change_text}{note}"
        f"近一年最高 {high.fund_share} 份（{high.trade_date}）、最低 {low.fund_share} 份（{low.trade_date}）。"
        "与报告基线的最终对比由 Java 侧以事实字段为准判定。"
    )
    facts = {
        "latest_share": latest.fund_share,
        "latest_trade_date": latest.trade_date,
        "previous_share": previous.fund_share if previous is not None else None,
        "previous_trade_date": previous.trade_date if previous is not None else None,
        "change_ratio": change_ratio,
        "one_year_max_share": high.fund_share,
        "one_year_max_date": high.trade_date,
        "one_year_min_share": low.fund_share,
        "one_year_min_date": low.trade_date,
    }
    return DiagnosisFactItem(
        item="SCALE", verdict=verdict, evidence=evidence, source="fund_share_snapshot",
        data_as_of_date=latest.trade_date, facts=facts,
    )


def _same_type_item(inputs: DiagnosisInputs) -> DiagnosisFactItem:
    source = "same-type-comparison"
    result = inputs.comparison
    if result is None:
        return _insufficient("SAME_TYPE_RANK", source, "同类比较不可用：该基金不在目录中。")
    as_of = result.as_of_date
    if result.target_status != "ACTIVE" or result.target_source_code != "TUSHARE_PRO_FUND":
        return _insufficient(
            "SAME_TYPE_RANK", source,
            f"该基金不在受控当前市场范围内（非 ACTIVE 或非 TUSHARE_PRO_FUND 份额）。{CONTROLLED_SAMPLE_NOTE}。",
            data_as_of_date=as_of,
        )
    if result.as_of_date is None or result.target_month_change_rate is None:
        return _insufficient(
            "SAME_TYPE_RANK", source, "目标基金最新净值或近一月涨跌率缺失。", data_as_of_date=as_of
        )
    comparable = sorted(
        (
            row
            for row in result.rows
            if row.nav_date == result.as_of_date
            and row.month_change_rate is not None
            and row.data_source is not None
        ),
        key=lambda row: (-row.month_change_rate, row.fund_code),
    )
    rank = next(
        (index for index, row in enumerate(comparable, start=1) if row.fund_code == inputs.fund_code),
        None,
    )
    if rank is None:
        return _insufficient(
            "SAME_TYPE_RANK", source, "目标基金不在可比样本内。", data_as_of_date=as_of
        )
    count = len(comparable)
    if count < 5:
        return _insufficient(
            "SAME_TYPE_RANK", source,
            f"同类样本不足（{count} < 5 只）。{CONTROLLED_SAMPLE_NOTE}。",
            data_as_of_date=as_of,
            facts={"comparable_count": count},
        )
    percentile = _q6(Decimal(rank) / count)
    evidence = (
        f"{CONTROLLED_SAMPLE_NOTE}，共 {count} 只，截至 {result.as_of_date}；"
        f"该基金近一月涨跌率 {result.target_month_change_rate} 排名第 {rank}（分位 {percentile}）。"
        "连续 3 个诊断日处于后 25% 才记 CHANGED 的判定由 Java 侧按诊断日序列执行。"
    )
    facts = {
        "rank": rank,
        "comparable_count": count,
        "month_change_rate": result.target_month_change_rate,
        "percentile": percentile,
        "scope": "CURRENT_MARKET_ACTIVE_TUSHARE_PRO_FUND",
    }
    return DiagnosisFactItem(
        item="SAME_TYPE_RANK", verdict="VALID", evidence=evidence, source=source,
        data_as_of_date=as_of, facts=facts,
    )


def _nav_series(
    nav_history: tuple[FundNavHistorySnapshot, ...],
) -> tuple[Literal["ACCUMULATED", "UNIT"] | None, tuple[tuple[date, Decimal], ...]]:
    """优先整段累计净值，全部有效才使用；否则回退单位净值，不自行还原复权。"""
    if not nav_history:
        return None, ()
    if all(p.accumulated_nav is not None and p.accumulated_nav > 0 for p in nav_history):
        return "ACCUMULATED", tuple((p.nav_date, p.accumulated_nav) for p in nav_history)
    if all(p.unit_nav > 0 for p in nav_history):
        return "UNIT", tuple((p.nav_date, p.unit_nav) for p in nav_history)
    return None, ()


def _benchmark_item(inputs: DiagnosisInputs, as_of: date) -> DiagnosisFactItem:
    source = "fund_profile + benchmark_nav_daily"
    if inputs.profile is None:
        return _insufficient("BENCHMARK", source, "fund_profile 快照缺失，无法取得业绩基准文本。")
    benchmark_text = inputs.profile.benchmark
    if not benchmark_text:
        return _insufficient("BENCHMARK", source, "fund_profile.benchmark 业绩基准文本缺失。")
    if not inputs.benchmark_code:
        return _insufficient("BENCHMARK", source, "未登记参考基准编码（fund_share_class.benchmark_code 为空）。")
    if inputs.benchmark_series_status != "ACTIVE":
        return _insufficient(
            "BENCHMARK", source,
            f"基准序列 {inputs.benchmark_code} 不存在或未启用（status={inputs.benchmark_series_status}）。",
        )
    points = tuple((d, v) for d, v in inputs.benchmark_points if d <= as_of)
    if len(points) < 60:
        return _insufficient(
            "BENCHMARK", source,
            f"基准净值覆盖不足（{len(points)} < 60 个交易日），缺失 benchmark_nav_daily 覆盖。",
        )
    if inputs.nav_unavailable_reason:
        return _insufficient("BENCHMARK", source, f"基金净值不可用：{inputs.nav_unavailable_reason}")
    basis, values = _nav_series(inputs.nav_history)
    if basis is None:
        return _insufficient("BENCHMARK", source, "基金净值数值无效，不补造收益差。")
    if len(values) < 60:
        return _insufficient(
            "BENCHMARK", source, f"基金净值不足 60 个交易日（{len(values)}）。", facts={"history_days": len(values)}
        )
    window = values[-60:]
    fund_return = window[-1][1] / window[0][1] - 1
    bench_dates = [d for d, _ in points]
    bench_map = dict(points)
    end_index = bisect_right(bench_dates, window[-1][0]) - 1
    start_index = bisect_right(bench_dates, window[0][0]) - 1
    if end_index < 0 or start_index < 0 or end_index == start_index:
        return _insufficient("BENCHMARK", source, "基准在基金近 60 个交易日窗口端点没有可用收盘点。")
    bench_return = bench_map[bench_dates[end_index]] / bench_map[bench_dates[start_index]] - 1
    diff = fund_return - bench_return
    verdict: Literal["VALID", "CHANGED"] = "CHANGED" if diff < 0 else "VALID"
    evidence = (
        f"业绩基准：{benchmark_text}（{inputs.benchmark_code}）；近 60 个交易日（{window[0][0]} 至 {window[-1][0]}）"
        f"基金收益 {_q6(fund_return):+.2%}、基准收益 {_q6(bench_return):+.2%}、差值 {_q6(diff):+.2%}。"
        "以基准序列为口径，不以同类排名代替基准；连续口径的最终判定由 Java 侧按诊断日序列执行。"
    )
    facts = {
        "benchmark_code": inputs.benchmark_code,
        "benchmark_text": benchmark_text,
        "nav_basis": basis,
        "window_start": window[0][0],
        "window_end": window[-1][0],
        "fund_return_60d": _q6(fund_return),
        "benchmark_return_60d": _q6(bench_return),
        "return_diff_60d": _q6(diff),
        "benchmark_coverage_days": len(points),
    }
    return DiagnosisFactItem(
        item="BENCHMARK", verdict=verdict, evidence=evidence, source=source,
        data_as_of_date=window[-1][0], facts=facts,
    )


def _drawdown_item(inputs: DiagnosisInputs) -> DiagnosisFactItem:
    source = "nav_daily"
    if inputs.nav_unavailable_reason:
        return _insufficient("DRAWDOWN", source, f"基金净值不可用：{inputs.nav_unavailable_reason}")
    basis, values = _nav_series(inputs.nav_history)
    if basis is None:
        return _insufficient("DRAWDOWN", source, "净值数值无效，不补造回撤。")
    if len(values) < 250:
        return _insufficient(
            "DRAWDOWN", source,
            f"历史净值不足 250 个交易日（{len(values)}），回撤统计不稳定，不输出数值。",
            facts={"history_days": len(values)},
        )
    peak_value, peak_date = values[0][1], values[0][0]
    max_dd = Decimal(0)
    max_peak_date = max_trough_date = values[0][0]
    for nav_date, value in values:
        if value > peak_value:
            peak_value, peak_date = value, nav_date
        drawdown = value / peak_value - 1
        if drawdown < max_dd:
            max_dd, max_peak_date, max_trough_date = drawdown, peak_date, nav_date
    current_dd = values[-1][1] / peak_value - 1
    verdict: Literal["VALID", "CHANGED"] = (
        "CHANGED" if max_dd < 0 and current_dd <= max_dd * Decimal("0.8") else "VALID"
    )
    evidence = (
        f"历史 {len(values)} 个交易日（{basis} 口径）；当前相对历史最高点回撤 {_q6(current_dd):+.2%}，"
        f"历史最大回撤 {_q6(max_dd):+.2%}（{max_peak_date} 至 {max_trough_date}）。"
        + ("当前回撤达到历史最大回撤的 80% 以上；风险提示，不是卖出指令。" if verdict == "CHANGED" else "")
    )
    facts = {
        "nav_basis": basis,
        "history_days": len(values),
        "current_drawdown": _q6(current_dd),
        "max_drawdown": _q6(max_dd),
        "max_drawdown_peak_date": max_peak_date,
        "max_drawdown_trough_date": max_trough_date,
        "current_vs_max_ratio": _q6(current_dd / max_dd) if max_dd < 0 else Decimal(0),
    }
    return DiagnosisFactItem(
        item="DRAWDOWN", verdict=verdict, evidence=evidence, source=source,
        data_as_of_date=values[-1][0], facts=facts,
    )


def _fee_item(inputs: DiagnosisInputs) -> DiagnosisFactItem:
    source = "fund_profile"
    profile = inputs.profile
    if profile is None:
        return _insufficient("FEE", source, "fund_profile 快照缺失，费用字段不可得。")
    if profile.management_fee is None or profile.custodian_fee is None:
        return _insufficient(
            "FEE", source,
            "fund_profile 缺少费用字段值"
            f"（management_fee={profile.management_fee}，custodian_fee={profile.custodian_fee}），不补造数值。",
            data_as_of_date=profile.updated_at.date() if profile.updated_at else None,
        )
    evidence = (
        f"当前管理费率 {profile.management_fee}、托管费率 {profile.custodian_fee}。"
        "fund_profile 为覆盖式快照、无历史版本；费用是否与上份报告不同由 Java 侧以报告基线对比判定。"
    )
    facts = {"management_fee": profile.management_fee, "custodian_fee": profile.custodian_fee}
    return DiagnosisFactItem(
        item="FEE", verdict="VALID", evidence=evidence, source=source,
        data_as_of_date=profile.updated_at.date() if profile.updated_at else None, facts=facts,
    )


def _dividend_item(inputs: DiagnosisInputs, as_of: date) -> DiagnosisFactItem:
    source = "fund_dividend"
    if inputs.dividends_verified_at is None:
        return _insufficient("DIVIDEND", source, "分红历史尚未完成同步核验（dividends_verified_at 为空）。")
    visible = tuple(d for d in inputs.dividends if d.ann_date is None or d.ann_date <= as_of)
    latest = (
        max(visible, key=lambda d: (d.ann_date or date.min, d.ex_date or date.min)) if visible else None
    )
    latest_text = (
        f"；最近事件公告日 {latest.ann_date}、除息日 {latest.ex_date}、每份现金分红 {latest.cash_dividend}"
        if latest is not None
        else "；无分红事件记录"
    )
    evidence = (
        f"已核验分红记录 {len(visible)} 条{latest_text}。"
        "基线之后是否出现新分红由 Java 侧对比判定；分红仅提示现金分红再投口径，不解释为恶化。"
    )
    facts = {
        "total_events": len(visible),
        "latest_ann_date": latest.ann_date if latest is not None else None,
        "latest_ex_date": latest.ex_date if latest is not None else None,
        "latest_cash_dividend": latest.cash_dividend if latest is not None else None,
    }
    return DiagnosisFactItem(
        item="DIVIDEND", verdict="VALID", evidence=evidence, source=source,
        data_as_of_date=(latest.ann_date if latest is not None else None) or inputs.dividends_verified_at.date(),
        facts=facts,
    )
