"""页面读取与研究发布检查隔离；旧ACTIVE记录和旧规则分数不能充当新版预测。"""

from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.watchlist_prediction import read_prediction_inputs
from app.schemas.watchlist_prediction import WatchlistPrediction
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
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        return build_prediction_status(*read_prediction_inputs(session, fund_code, today))
