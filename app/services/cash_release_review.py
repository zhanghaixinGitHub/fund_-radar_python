"""已保存研究的候选规则审查；不冻结规则、不签发凭证、不把旧研究升级为正式模型。"""

from datetime import UTC, datetime
from decimal import Decimal, localcontext

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.cash_planned_research import find_planned_research
from app.schemas.calibration_diagnostic import WindowCalibrationDiagnostic
from app.schemas.cash_planned_research import CashPlannedResearch
from app.schemas.cash_policy_freeze import CashFrozenPolicy
from app.schemas.cash_prediction_check import CashPredictionCheckRequest
from app.schemas.cash_reinvestment_research import CashStoredResearch
from app.schemas.cash_release_review import (
    CashCoverageEvidence,
    CashReleaseCriterion,
    CashReleasePolicy,
    CashReleaseReview,
)
from app.schemas.historical_nav_calibration import ReliabilityReport
from app.schemas.historical_nav_evaluation import BaselineMetrics
from app.services.calibration_policy import calibration_diagnostic
from app.services.cash_exam_plan import validate_cash_exam_plan
from app.services.cash_planned_research import read_planned_in_session
from app.services.cash_policy_freeze import read_matching_policy_freeze, validate_policy_binding
from app.services.cash_prediction_check import inspect_cash_research, load_cash_prediction_research
from app.services.cash_reinvestment_storage import cash_hash
from app.services.cash_release_evidence import validate_reliability
from app.services.cash_release_policy import load_release_policy
from app.services.historical_nav_storage import HistoricalNavStorageError

EVIDENCE_MESSAGES = {
    "HISTORICAL_FIRST_VERSIONS_UNVERIFIED": "缺少历史首次公开版本的证据，不能把今天看到的修订值当成当时已知。",
    "DIVIDEND_COMPLETENESS_UNVERIFIED": "缺少现金分红完整性证明，数据库里没有记录不等于从未分红。",
    "NON_CASH_ADJUSTMENTS_UNVERIFIED": "缺少份额拆分等非现金调整的完整性证明。",
    "INDEPENDENT_TEST_NOT_EVALUATED": "2025最终独立测试尚未执行；此次审查不会读取其数值或答案。",
}


def _missing(
    code: str, message: str, *, window_id="GLOBAL", scope="ALL", required=None, operator="PRESENT"
) -> CashReleaseCriterion:
    return CashReleaseCriterion(
        code=code,
        window_id=window_id,
        scope=scope,
        status="MISSING",
        operator=operator,
        required=required,
        message=message,
    )


def _numeric(code: str, actual, required, operator: str, message: str, **identity) -> CashReleaseCriterion:
    actual, required = Decimal(str(actual)), Decimal(str(required))
    passed = {"GE": actual >= required, "LE": actual <= required, "GT": actual > required, "LT": actual < required}[
        operator
    ]
    return CashReleaseCriterion(
        code=code,
        actual=actual,
        required=required,
        operator=operator,
        status="PASS" if passed else "FAIL",
        message=message,
        **identity,
    )


def evaluate_annual_reliability(
    report: ReliabilityReport, metrics: BaselineMetrics, policy: CashReleasePolicy, *, window_id: str, scope: str
) -> tuple[CashReleaseCriterion, ...]:
    """年度考试才执行完整分档门槛；空档不凑数，少量孤立样本不自动合并到邻档。"""
    validate_reliability(report, metrics)
    identity = {"window_id": window_id, "scope": scope}
    populated = tuple((i, b) for i, b in enumerate(report.bins) if b.count)
    # 保存的ECE仅有8位小数；门槛比较使用分档重算值，避免略超限被四舍五入成恰好达标。
    weighted_ece = sum(b.absolute_gap * b.count / report.sample_count for _, b in populated)
    checks = [
        _numeric(
            "ANNUAL_POPULATED_BINS",
            len(populated),
            policy.minimum_populated_annual_bins,
            "GE",
            "年度至少两个概率档有样本，才能比较高低概率的实际表现。",
            **identity,
        ),
        _numeric(
            "ANNUAL_ECE",
            weighted_ece,
            policy.maximum_ece,
            "LE",
            "各档概率与实际上涨比例的差距按样本数加权；0.10表示10个百分点。",
            **identity,
        ),
    ]
    for index, bucket in populated:
        checks.extend(
            (
                _numeric(
                    "ANNUAL_BIN_COUNT",
                    bucket.count,
                    policy.minimum_bin_count,
                    "GE",
                    "本非空档至少有30条考试样本；数量通过也不表示这些题统计独立。",
                    bin_index=index,
                    **identity,
                ),
                _numeric(
                    "ANNUAL_BIN_GAP",
                    bucket.absolute_gap,
                    policy.maximum_bin_gap,
                    "LE",
                    "本档平均预测分数与实际上涨比例的绝对差；0.15表示15个百分点。",
                    bin_index=index,
                    **identity,
                ),
            )
        )
    monotone = all(
        left.observed_up_rate <= right.observed_up_rate
        for (_, left), (_, right) in zip(populated[:-1], populated[1:], strict=True)
    )
    checks.append(
        CashReleaseCriterion(
            code="ANNUAL_OBSERVED_RATE_ORDER",
            operator="NONDECREASING",
            **identity,
            status="MISSING" if len(populated) < 2 else "PASS" if monotone else "FAIL",
            message="按预测分数从低到高排列，非空档的实际上涨比例不得降低；少于两档无法比较。",
        )
    )
    return tuple(checks)


def review_cash_research(
    stored: CashStoredResearch,
    policy: CashReleasePolicy,
    *,
    fund_code: str,
    checked_at: datetime,
    frozen_policy: CashFrozenPolicy | None = None,
    planned_research: CashPlannedResearch | None = None,
) -> CashReleaseReview:
    """确定性核验已有报告，不改研究内容；所有三试点/固定窗口都检查，不按成绩择优。"""
    # 同原报告采用28位Decimal上下文，避免调用者改变精度后影响分档算术校验。
    with localcontext() as context:
        context.prec = 28
        return _review_cash_research(
            stored,
            policy,
            fund_code=fund_code,
            checked_at=checked_at,
            frozen_policy=frozen_policy,
            planned_research=planned_research,
        )


def _review_cash_research(stored, policy, *, fund_code, checked_at, frozen_policy, planned_research):
    policy = CashReleasePolicy.model_validate_json(policy.model_dump_json())
    if fund_code not in policy.fund_codes:
        raise ValueError("review fund outside frozen pilot scope")
    if frozen_policy is not None:
        validate_policy_binding(frozen_policy, policy)
    blockers, comparisons, _ = inspect_cash_research(stored)
    planned_groups = {}
    if planned_research is not None:
        # 入口只从同一只读事务中的真实关联还原此对象；不接受HTTP上传的计划或覆盖率。
        planned_research = CashPlannedResearch.model_validate_json(planned_research.model_dump_json())
        if (
            frozen_policy is None
            or planned_research.policy_freeze != frozen_policy
            or planned_research.research != stored
            or planned_research.preparation.preparation != stored.report.preparation
        ):
            raise ValueError("planned evidence belongs to a different policy or research")
        validate_cash_exam_plan(planned_research.preparation.plan)
        planned_groups = {(g.window_id, g.fund_code): g for g in planned_research.preparation.coverage}
    coverage_evidence = []
    calibration_diagnostics = []
    checks = [_missing(code, message) for code, message in EVIDENCE_MESSAGES.items()]
    checks.append(
        CashReleaseCriterion(
            code="POLICY_APPROVAL",
            window_id="GLOBAL",
            scope="ALL",
            operator="PRESENT",
            status="PASS" if policy.approval_state == "APPROVED" else "MISSING",
            message="业务确认须有可追溯记录；确认规则仍不等于规则已持久冻结或模型可发布。",
        )
    )
    checks.append(
        CashReleaseCriterion(
            code="POLICY_FREEZE",
            window_id="GLOBAL",
            scope="ALL",
            operator="PRESENT",
            status="PASS" if frozen_policy else "MISSING",
            message="已核验不可变规则快照与当前配置一致；仍不是模型发布许可。"
            if frozen_policy
            else "尚无匹配的不可变规则冻结凭证；文件内容指纹不能代替正式冻结记录。",
        )
    )
    for state in policy.required_market_states:
        checks.append(
            _missing(f"MARKET_STATE_{state}", f"缺少按事先固定方法划分的{state}行情考试证据；未分类不算覆盖。")
        )
    for window in stored.report.windows:
        window_id = window.window.window_id
        identity = {"window_id": window_id, "scope": "ALL"}
        checks.append(
            CashReleaseCriterion(
                code="WINDOW_EVALUATED",
                operator="PRESENT",
                **identity,
                status="PASS" if window.status == "EVALUATED" else "MISSING",
                message="固定窗口已完成考试。"
                if window.status == "EVALUATED"
                else "本固定窗口未完成考试，不能当作比较成绩通过。",
            )
        )
        minimums = {
            "FIT": policy.minimum_fit_per_fund,
            "CALIBRATION": policy.minimum_calibration_per_fund,
            "EXAM": policy.minimum_annual_exam_per_fund
            if window.window.role == "FIXED_VALIDATION"
            else policy.minimum_rolling_exam_per_fund,
        }
        for fund in window.funds:
            for stage, minimum in minimums.items():
                checks.append(
                    _numeric(
                        f"{stage}_SAMPLE_COUNT",
                        fund.counts[stage],
                        minimum,
                        "GE",
                        "本基金在本窗口剔除跨界答案后实际可用的题数，不用其他基金题数补齐。",
                        window_id=window_id,
                        scope=fund.fund_code,
                    )
                )
            if planned_research is not None and window.status == "EVALUATED":
                group = planned_groups[window_id, fund.fund_code]
                # inspect_cash_research已逐基金核验EXAM数量等于模型实际评分数，不能以待考题数冒充。
                if group.usable_count != fund.counts["EXAM"]:
                    raise ValueError("bound population differs from scored population")
                evidence = CashCoverageEvidence(
                    plan_version=planned_research.preparation.plan.version,
                    plan_hash=planned_research.preparation.plan.plan_hash,
                    dataset_hash=stored.report.preparation.dataset_hash,
                    window_id=window_id,
                    fund_code=fund.fund_code,
                    planned_count=group.planned_count,
                    scored_count=fund.counts["EXAM"],
                )
                coverage_evidence.append(evidence)
                checks.append(
                    _numeric(
                        "EXAM_COVERAGE",
                        Decimal(evidence.scored_count) / evidence.planned_count,
                        policy.minimum_coverage,
                        "GE",
                        "本窗口模型实际评分题数/评估前固定计划题数；覆盖率不是预测准确率。",
                        window_id=window_id,
                        scope=fund.fund_code,
                    )
                )
            else:
                checks.append(
                    _missing(
                        "EXAM_COVERAGE",
                        "已核验事前计划，但本窗口尚未完成评分，不能把可用资料当成模型考试成绩。"
                        if planned_research is not None
                        else "缺少事前应考截点计划，无法计算实际可评分题数/计划题数。",
                        window_id=window_id,
                        scope=fund.fund_code,
                        required=policy.minimum_coverage,
                        operator="GE",
                    )
                )
        if window.model is not None and policy.version == "CASH_RELEASE_POLICY_V1":
            checks.append(
                _numeric(
                    "CALIBRATION_SLOPE",
                    window.model.calibrator.slope,
                    0,
                    "GT",
                    "校准映射斜率必须为正；不允许无声反转基础模型的高低分顺序。",
                    **identity,
                )
            )
        elif window.model is not None:
            diagnostic = calibration_diagnostic(window.model.calibrator.slope, window.model.calibrator.intercept)
            calibration_diagnostics.append(WindowCalibrationDiagnostic(window_id=window_id, **diagnostic.model_dump()))
        if window.window.role == "FIXED_VALIDATION":
            if window.status != "EVALUATED":
                for scope in ("ALL", *policy.fund_codes):
                    checks.append(
                        _missing(
                            "ANNUAL_CALIBRATION_EVIDENCE",
                            "年度考试未完成，无法核验固定概率分档。",
                            window_id=window_id,
                            scope=scope,
                        )
                    )
            else:
                reports = {
                    "ALL": window.reliability_after,
                    **{f.fund_code: f.after for f in window.reliability_per_fund},
                }
                metrics = {"ALL": window.after.validation, **{f.fund_code: f.metrics for f in window.after.per_fund}}
                for scope in ("ALL", *policy.fund_codes):
                    checks.extend(
                        evaluate_annual_reliability(
                            reports[scope], metrics[scope], policy, window_id=window_id, scope=scope
                        )
                    )
    for comparison in comparisons:
        identity = {"window_id": comparison.window_id, "scope": comparison.scope, "baseline_id": comparison.baseline_id}
        checks.extend(
            (
                _numeric(
                    "BASELINE_ACCURACY_GAIN",
                    comparison.accuracy_delta,
                    0,
                    "GT",
                    "模型准确率减此对照，必须严格大于0；打平不算通过。",
                    **identity,
                ),
                _numeric(
                    "BASELINE_BRIER_GAIN",
                    comparison.brier_delta,
                    0,
                    "LT",
                    "模型平方误差减此对照，必须严格小于0；不能只凭方向准确率放行。",
                    **identity,
                ),
            )
        )
    blocking = list(blockers)
    if policy.approval_state != "APPROVED":
        blocking.append("POLICY_APPROVAL_MISSING")
    if frozen_policy is None:
        blocking.append("POLICY_NOT_FROZEN")
    if any(c.code == "EXAM_COVERAGE" and c.status == "MISSING" for c in checks):
        blocking.append("EX_ANTE_COVERAGE_EVIDENCE_MISSING")
    blocking.append("MARKET_STATE_EVIDENCE_MISSING")
    if any(c.status == "FAIL" for c in checks):
        blocking.append("RELEASE_RULES_NOT_MET")
    return CashReleaseReview(
        checked_at=checked_at,
        fund_code=fund_code,
        research_run_id=stored.run_id,
        report_hash=stored.report.report_hash,
        policy_hash=cash_hash(policy.model_dump(mode="json")),
        policy=policy,
        blocking_codes=tuple(dict.fromkeys(blocking)),
        checks=tuple(checks),
        calibration_diagnostics=tuple(calibration_diagnostics),
        check_counts={s: sum(c.status == s for c in checks) for s in ("PASS", "FAIL", "MISSING")},
        policy_persisted=frozen_policy is not None,
        policy_freeze_id=frozen_policy.freeze_id if frozen_policy else None,
        policy_frozen_at=frozen_policy.frozen_at if frozen_policy else None,
        policy_freeze_hash=frozen_policy.content_hash if frozen_policy else None,
        policy_binding_hash=frozen_policy.binding_hash if frozen_policy else None,
        ex_ante_plan_verified=planned_research is not None,
        planned_research_binding_id=planned_research.binding_id if planned_research else None,
        planned_research_binding_hash=planned_research.content_hash if planned_research else None,
        exam_plan_hash=planned_research.preparation.plan.plan_hash if planned_research else None,
        exam_coverage_evidence=tuple(coverage_evidence),
    )


def review_cash_release(request: CashPredictionCheckRequest) -> CashReleaseReview:
    """仅基金/来源/净值日期元数据和一份指定报告；一个只读快照，不加载X/y或预测结果。"""
    policy = load_release_policy()
    now = datetime.now(UTC)
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        stored = load_cash_prediction_research(session, request, now=now)
        frozen = read_matching_policy_freeze(session, policy)
        # 只有当前规则确实已批准、冻结才核验研究关联。草案或无匹配规则时不能借旧关联获准。
        binding = find_planned_research(session, research_run_id=stored.run_id) if frozen is not None else None
        planned = read_planned_in_session(session, binding) if binding is not None else None
        try:
            return review_cash_research(
                stored,
                policy,
                fund_code=request.fund_code,
                checked_at=now,
                frozen_policy=frozen,
                planned_research=planned,
            )
        except (ValueError, TypeError, KeyError, ArithmeticError) as error:
            raise HistoricalNavStorageError(
                "CASH_RESEARCH_CORRUPTED", "研究成绩或分档证据相互矛盾，停止规则审查。", 503
            ) from error
