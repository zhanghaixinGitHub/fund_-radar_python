"""日历确定应考日期；不访问数据库、行情价格、标签或模型。"""

from datetime import date, timedelta

from app.schemas.cash_exam_plan import CashExamPlan, CashExamWindowPlan
from app.services.cash_reinvestment_research import FUNDS
from app.services.cash_reinvestment_storage import cash_hash
from app.services.historical_nav_calibration import WINDOWS
from app.services.trading_calendar import load_calendar


def build_cash_exam_plan() -> CashExamPlan:
    """先列全交易日，再剔除终点必定越界的20日尾部；缺净值或晚公告不能再缩小分母。"""
    calendar = load_calendar()
    periods = [
        (w.window_id, w.role, w.calibration_end_date + timedelta(days=1), w.evaluation_end_date) for w in WINDOWS
    ]
    # V1独立测试年份本就固定为2025；不读今天的审批文件，保证旧快照仍可独立核验。
    periods.append(("INDEPENDENT_TEST_2025", "INDEPENDENT_TEST", date(2025, 1, 1), date(2025, 12, 31)))
    windows = []
    for name, role, start, end in periods:
        # 最后20个交易日的答案必然在本窗口之后，连日期都不需要向下一年猜测延长。
        sessions = tuple(day for day in calendar.sessions if start <= day <= end)
        windows.append(
            CashExamWindowPlan(
                window_id=name,
                role=role,
                start_date=start,
                end_date=end,
                planned_cutoffs=sessions[:-20],
                boundary_purged_cutoffs=sessions[-20:],
            )
        )
    result = CashExamPlan(
        fund_codes=FUNDS,
        calendar_version=calendar.definition.version,
        calendar_hash=calendar.content_hash,
        windows=tuple(windows),
        plan_hash="0" * 64,
    )
    return result.model_copy(update={"plan_hash": cash_hash(result.model_dump(mode="json", exclude={"plan_hash"}))})


def validate_cash_exam_plan(plan: CashExamPlan) -> None:
    """校验保存的清单必须等于独立生成的完整日历；重新计算指纹也不能掩盖删日期。"""
    if plan.model_dump(mode="json") != build_cash_exam_plan().model_dump(mode="json"):
        raise ValueError("cash exam plan differs from fixed calendar and windows")
