"""按固定日期计划检查已存研究资料覆盖；不训练，不把预览补盖成事前冻结证据。"""

from datetime import date
from decimal import Decimal, localcontext

from app.schemas.cash_exam_plan import CashExamCoverage, CashExamPlan, CashExamPreparation, CashExamPreparationRequest
from app.services.cash_exam_plan import build_cash_exam_plan, validate_cash_exam_plan
from app.services.cash_reinvestment_research import CashDataset, cash_window_rows, load_cash_dataset
from app.services.historical_nav_calibration import WINDOWS
from app.services.historical_nav_storage import HistoricalNavStorageError


def summarize_cash_exam_preparation(data: CashDataset, plan: CashExamPlan) -> CashExamPreparation:
    """只检查已验证的小矩阵的身份/可得日期，既不读取y也不计算模型分数。"""
    validate_cash_exam_plan(plan)
    seen = set()
    for row in data.rows:
        identity = row.fund_code, row.available_at
        if (
            identity in seen
            or row.fund_code not in plan.fund_codes
            or row.as_of_date != row.available_at
            or not date(2022, 1, 1) <= row.available_at < row.label_available_at <= date(2024, 12, 31)
        ):
            raise ValueError("exam preparation row scope/date/identity mismatch")
        seen.add(identity)
    groups = []
    with localcontext() as context:
        context.prec = 28
        for window in plan.windows:
            planned = set(window.planned_cutoffs)
            if window.role == "INDEPENDENT_TEST":
                groups.extend(
                    CashExamCoverage(
                        window_id=window.window_id,
                        fund_code=fund,
                        status="TEST_PERIOD_PROTECTED",
                        planned_count=len(planned),
                    )
                    for fund in plan.fund_codes
                )
                continue
            actual_window = next(w for w in WINDOWS if w.window_id == window.window_id)
            selected, _ = cash_window_rows(data, actual_window)  # 与研究器同一分段和跨界标签剔除规则。
            for fund in plan.fund_codes:
                usable = {r.available_at for r in selected["EXAM"] if r.fund_code == fund}
                if not usable.issubset(planned):
                    raise ValueError("prepared exam rows fall outside fixed eligible dates")
                late = sum(
                    r.fund_code == fund and r.available_at in planned and r.label_available_at > window.end_date
                    for r in data.rows
                )
                groups.append(
                    CashExamCoverage(
                        window_id=window.window_id,
                        fund_code=fund,
                        status="PREPARED",
                        planned_count=len(planned),
                        usable_count=len(usable),
                        coverage=Decimal(len(usable)) / len(planned),
                        missing_cutoffs=tuple(sorted(planned - usable)),
                        late_label_count=late,
                    )
                )
    return CashExamPreparation(plan=plan, preparation=data.report, coverage=tuple(groups))


def prepare_cash_exam_data(request: CashExamPreparationRequest) -> CashExamPreparation:
    plan = build_cash_exam_plan()  # 在开库和看数据之前生成；请求不能提交自选日期或覆盖率。
    if plan.plan_hash != request.expected_plan_hash:
        raise HistoricalNavStorageError("EXAM_PLAN_HASH_MISMATCH", "考试日期计划与核对的指纹不一致。", 409)
    data = load_cash_dataset(request)
    if data.report.dataset_hash != request.expected_dataset_hash:
        raise HistoricalNavStorageError("DATASET_HASH_MISMATCH", "资料指纹已变化，请重新核对准备报告。", 409)
    return summarize_cash_exam_preparation(data, plan)
