"""页面读取与研究发布检查隔离；旧ACTIVE记录和旧规则分数不能充当新版预测。"""

from datetime import date
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.cash_forecast import find_latest_forecast
from app.repositories.watchlist_prediction import find_latest_fund_research, read_prediction_inputs
from app.schemas.cash_forecast import CashForecastView
from app.schemas.watchlist_prediction import WatchlistPrediction
from app.services.cash_forecast import read_cash_forecast_in_session, utc_now
from app.services.cash_prediction_check import inspect_cash_research
from app.services.cash_reinvestment_research import FUNDS, restore_research
from app.services.historical_nav_storage import HistoricalNavStorageError

REASONS = {
    "HISTORICAL_FIRST_VERSIONS_UNVERIFIED": "尚不能证实历史数据就是当时首次公开的版本。",
    "DIVIDEND_COMPLETENESS_UNVERIFIED": "历史分红是否记录完整，还缺少核验证据。",
    "NON_CASH_ADJUSTMENTS_UNVERIFIED": "基金拆分、份额折算等非现金调整尚未核验完整。",
    "INDEPENDENT_TEST_NOT_EVALUATED": "最终独立测试尚未完成，不能据研究成绩发布概率。",
    "ROLLING_WINDOWS_INCOMPLETE": "部分滚动验证窗口的有效样本不足，尚未完成全部验证。",
    "REPEATABLE_BASELINE_GAIN_NOT_ESTABLISHED": "已有研究成绩尚未证明模型能稳定优于简单对照规则。",
    "FORMAL_RELEASE_NOT_ISSUED": "尚未取得正式模型发布凭证。",
    "PUBLICATION_AUTHORIZATION_CHANGED": "原结果的模型发布授权已变更，不能继续显示旧概率。",
    "SOURCE_REVISION_CHANGED": "净值或分红等来源资料已更新，原结果需要重新生成。",
    "SOURCE_UPDATE_NOT_READY": "来源同步尚未完成或最近同步失败，暂不使用旧结果。",
    "LATEST_NAV_CHANGED": "已有更新的已公告净值，原结果不再代表最新资料。",
    "NAV_TOO_OLD": "最新净值已超过允许的时效，暂不展示预测数字。",
    "FORECAST_WINDOW_ENDED": "这份预测的20交易日期间已结束，不继续展示旧概率。",
    "FORECAST_CUTOFF_NOT_CLOSED": "这份结果的信息截止日尚未结束，不能展示。",
    "CALENDAR_COVERAGE_INSUFFICIENT": "当前日期不在已核验日历范围，无法确认结果时效。",
}


def build_prediction_status(fund, source, latest_date, run) -> WatchlistPrediction:
    if fund is None:
        raise HistoricalNavStorageError("FUND_NOT_FOUND", "基金不存在。", 404)
    payload = {"fund_code": fund.fund_code, "latest_nav_date": latest_date}
    if fund.fund_type != "STOCK":
        return WatchlistPrediction(
            **payload,
            status="NOT_APPLICABLE",
            message="这一版只研究股票型基金。",
            reason_codes=("FUND_TYPE_NOT_SUPPORTED",),
            reasons=("债券型、混合型、指数型等暂不使用此模型。",),
        )
    if fund.status != "ACTIVE" or source is None or not source.enabled:
        return WatchlistPrediction(
            **payload,
            status="DATA_INSUFFICIENT",
            message="基金或数据来源当前不可用。",
            reason_codes=("SOURCE_OR_FUND_INACTIVE",),
            reasons=("未使用停用来源计算或展示预测。",),
        )
    if fund.fund_code not in FUNDS:
        return WatchlistPrediction(
            **payload,
            status="DATA_INSUFFICIENT",
            message="这只基金还没有完成本版研究。",
            reason_codes=("OUTSIDE_RESEARCH_PILOT",),
            reasons=("目前仅核验001632、006730、008888三只试点，不能直接套用到其他基金。",),
        )
    if run is None:
        return WatchlistPrediction(
            **payload,
            status="MODEL_NOT_RELEASED",
            message="模型尚未发布，暂不展示上涨概率。",
            reason_codes=("NO_RESEARCH_RUN",),
            reasons=("尚无这只基金的已保存新版研究报告。",),
        )
    stored = restore_research(run)  # 内容损坏必须报错，不能将未知状态当可用。
    payload.update(
        research_run_id=stored.run_id,
        research_evaluated_at=stored.created_at,
        model_version=stored.report.protocol_version,
    )
    try:
        blockers, _, _ = inspect_cash_research(stored)
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        raise HistoricalNavStorageError("CASH_RESEARCH_CORRUPTED", "研究报告比较证据校验失败。", 503) from error
    reasons = tuple(REASONS[code] for code in blockers)
    # 固定门禁只允许无概率状态；将来有新证据必须新增正式发布协议，不能偷偷改研究表标志。
    return WatchlistPrediction(
        **payload,
        status="MODEL_NOT_RELEASED",
        message="模型仍在研究验证，暂不展示上涨概率。",
        reason_codes=blockers,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def get_watchlist_prediction(fund_code: str) -> WatchlistPrediction:
    now = utc_now()
    today = now.astimezone(ZoneInfo("Asia/Shanghai")).date()
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        fund, source, latest, run = read_prediction_inputs(session, fund_code, today, include_research=False)
        if (
            fund is not None
            and fund.fund_type == "STOCK"
            and fund.status == "ACTIVE"
            and fund.fund_code in FUNDS
            and source is not None
            and source.enabled
        ):
            row = find_latest_forecast(session, fund_code=fund_code)
            if row is not None:
                result = read_cash_forecast_in_session(session, row, now=now)
                return project_cash_forecast(result, latest_date=latest, research_run_id=row.research_run_id)
        # 没有结果时才找研究；该次只读仍在同一个快照内，不触发生成或模型运行。
        if fund is not None and fund.fund_type == "STOCK" and fund.fund_code in FUNDS:
            run = find_latest_fund_research(session, fund_code)
        return build_prediction_status(fund, source, latest, run)


def project_cash_forecast(
    result: CashForecastView, *, latest_date: date | None, research_run_id: UUID
) -> WatchlistPrediction:
    """结果投影无模型、授权和输入快照；任何未知拒绝原因用安全中文说明，不透传异常。"""
    messages = {
        "AVAILABLE": "已读取有效模型结果；概率不是收益率，也不保证未来涨跌。",
        "STALE": "原预测已失效，暂不展示概率。",
        "MODEL_NOT_RELEASED": "模型当前未获正式发布授权，暂不展示概率。",
        "DATA_INSUFFICIENT": "预测所需资料尚未就绪。",
    }
    fields = result.model_dump(exclude={"created", "horizon_trading_days"})
    return WatchlistPrediction(
        **fields,
        latest_nav_date=latest_date,
        research_run_id=research_run_id,
        model_version="CASH_FORECAST_STORAGE_V1",
        reasons=tuple(REASONS.get(code, "结果的授权或数据校验未通过，暂不展示概率。") for code in result.reason_codes),
        message=messages[result.status],
        disclaimer="仅使用净值历史及现金分红再投口径，未包含持仓、新闻和公告分析。概率不是收益承诺，不构成投资建议。",
    )
