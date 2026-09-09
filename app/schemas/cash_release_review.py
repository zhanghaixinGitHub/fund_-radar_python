"""发布规则草案与核验明细；规则审批/持久冻结、数值合格和正式发布是三件不同的事。"""

from datetime import date, datetime
from decimal import Decimal, localcontext
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.calibration_diagnostic import WindowCalibrationDiagnostic
from app.schemas.historical_nav_training import Hash


class CashReleasePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["CASH_RELEASE_POLICY_V1", "CASH_RELEASE_POLICY_V2"] = Field(
        description="V1保留正斜率硬检查，V2只记录方向风险；旧冻结记录不改"
    )
    approval_state: Literal["DRAFT", "APPROVED"] = Field(description="草案或业务已确认；APPROVED仍不是数据库已冻结")
    approval_reference: str | None = Field(max_length=200, description="业务确认的可追溯记录，不能由HTTP参数代填")
    fund_codes: tuple[str, ...] = Field(description="三只已确认股票型试点，不按成绩剔除基金")
    horizon_trading_days: Literal[20] = Field(description="固定现金再投总回报方向窗口")
    minimum_fit_per_fund: Literal[252] = Field(description="每基金净化后基础训练样本最低数量")
    minimum_calibration_per_fund: Literal[60] = Field(description="每基金与基础训练隔离的校准样本最低数量")
    minimum_rolling_exam_per_fund: Literal[40] = Field(description="季度研究窗口原有最低考试数量")
    minimum_annual_exam_per_fund: Literal[120] = Field(description="年度验证及最终测试最低考试数量")
    minimum_bin_count: Literal[30] = Field(description="非空概率档最低数量；低于门槛不能当作可靠校准证据")
    minimum_populated_annual_bins: Literal[2] = Field(description="至少两个有数据的年度概率档才可比较高低概率")
    maximum_ece: Decimal = Field(ge=0, le=1, description="年度分组加权平均绝对误差上限；0.10是10个百分点")
    maximum_bin_gap: Decimal = Field(ge=0, le=1, description="年度任一非空概率档绝对误差上限")
    minimum_coverage: Decimal = Field(gt=0, le=1, description="净化后考试数/事前计划可考试截点数，不用训练数作分母")
    require_strict_accuracy_gain: Literal[True] = Field(description="每窗口/基金相对全部四个对照准确率严格更高")
    require_strict_brier_gain: Literal[True] = Field(description="同时要求Brier严格更低，不能用准确率代替")
    require_monotone_observed_rates: Literal[True] = Field(description="年度相邻非空概率档的实际上涨率不逆序")
    independent_test_start: date = Field(description="固定保留测试起点，不表示已允许读取答案")
    independent_test_end: date = Field(description="固定保留测试终点，不根据成绩更换")
    required_market_states: tuple[str, ...] = Field(description="正式发布须覆盖的行情类型；未分类不是已覆盖")

    @model_validator(mode="after")
    def frozen_scope(self) -> Self:
        if self.fund_codes != ("001632", "006730", "008888"):
            raise ValueError("cash release pilot scope cannot change")
        if (self.independent_test_start, self.independent_test_end) != (date(2025, 1, 1), date(2025, 12, 31)):
            raise ValueError("independent test period cannot change")
        if self.required_market_states != ("UP", "DOWN", "HIGH_VOLATILITY", "SIDEWAYS"):
            raise ValueError("required market states cannot be omitted")
        if (self.approval_state == "APPROVED") != bool(self.approval_reference and self.approval_reference.strip()):
            raise ValueError("approval state requires an explicit review reference")
        if self.maximum_ece > self.maximum_bin_gap:
            raise ValueError("average error limit cannot exceed worst-bin limit")
        return self


class CashReleaseCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    code: str = Field(description="核验项编号，不是基金涨跌结论")
    window_id: str = Field(description="在哪个固定窗口检查；GLOBAL表示整个发布流程")
    scope: str = Field(description="ALL表示总体；基金代码表示逐基金，不混为一次通过")
    baseline_id: str | None = Field(default=None, description="比较哪条简单对照；非对照检查为null")
    bin_index: int | None = Field(default=None, ge=0, le=4, description="固定五个概率档的编号0到4；非分档检查为null")
    status: Literal["PASS", "FAIL", "MISSING"] = Field(description="符合候选规则、不符合或证据缺失；PASS不是发布许可")
    operator: Literal["GE", "LE", "GT", "LT", "PRESENT", "NONDECREASING"] = Field(
        description="依次为至少、至多、严格大于、严格小于、须有证据、不得逆序"
    )
    actual: Decimal | None = Field(default=None, description="实际核验数值；缺证据保持null，不填0")
    required: Decimal | None = Field(default=None, description="对应规则边界，比较含义由code/message说明")
    message: str = Field(description="通俗核验说明")

    @model_validator(mode="after")
    def numeric_result_matches_values(self) -> Self:
        if self.operator in {"GE", "LE", "GT", "LT"}:
            if self.required is None:
                raise ValueError("numeric check requires a threshold")
            if self.status == "MISSING":
                if self.actual is not None:
                    raise ValueError("missing numeric evidence cannot contain an actual value")
                return self
            if self.actual is None:
                raise ValueError("numeric result requires actual evidence")
            passed = {
                "GE": self.actual >= self.required,
                "LE": self.actual <= self.required,
                "GT": self.actual > self.required,
                "LT": self.actual < self.required,
            }[self.operator]
            if (self.status == "PASS") != passed:
                raise ValueError("numeric check status contradicts its values")
        return self


class CashCoverageEvidence(BaseModel):
    """服务端从已核验研究绑定及实际成绩提取的覆盖明细；HTTP不接受调用者填报。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    plan_version: Literal["CASH_EX_ANTE_EXAM_PLAN_V1"]
    plan_hash: Hash = Field(description="事前计划指纹；不能仅有一个自报通过标记")
    dataset_hash: Hash = Field(description="所核验的明确资料指纹")
    window_id: str = Field(description="固定考试窗口编号")
    fund_code: str = Field(description="明确基金，不使用被筛选后的总体数字代替")
    planned_count: int = Field(gt=0, le=1000, strict=True, description="按冻结窗口及日历事前确定的可考试截点数")
    scored_count: int = Field(
        ge=0, le=1000, strict=True, description="已完成窗口中实际被模型评分的数量，不是待考可用题数"
    )

    @model_validator(mode="after")
    def counts(self) -> Self:
        if self.scored_count > self.planned_count:
            raise ValueError("scored count cannot exceed plan")
        return self


class CashReleaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["READ_ONLY_RELEASE_RULE_REVIEW"] = "READ_ONLY_RELEASE_RULE_REVIEW"
    status: Literal["BLOCKED"] = "BLOCKED"
    checked_at: datetime = Field(description="此次核验时刻，不是模型训练或发布时间")
    fund_code: str = Field(description="请求所指基金；发布检查始终覆盖全部三试点，不因请求基金而择优")
    research_run_id: UUID = Field(description="实际重读和核验的研究编号")
    report_hash: Hash = Field(description="原研究指纹，不修改研究报告")
    policy_hash: Hash = Field(description="本次候选规则内容指纹，不能当作已冻结凭证")
    policy: CashReleasePolicy = Field(description="实际使用的服务器规则；草案同样可作诊断，但不能发布")
    blocking_codes: tuple[str, ...] = Field(description="未解除的证据、规则冻结和成绩问题")
    checks: tuple[CashReleaseCriterion, ...] = Field(description="逐窗口、逐基金的候选规则核验明细")
    calibration_diagnostics: tuple[WindowCalibrationDiagnostic, ...] = Field(
        default=(), description="各窗口校准方向说明；V2不以斜率符号判合格或失败，不计入检查通过数"
    )
    check_counts: dict[Literal["PASS", "FAIL", "MISSING"], int] = Field(
        description="明细中通过、不通过、缺证据各多少项"
    )
    policy_persisted: bool = Field(
        default=False, description="是否已读到并验证匹配的冻结快照；此只读接口自己不保存规则"
    )
    policy_freeze_id: UUID | None = Field(default=None, description="匹配的不可变规则快照编号；缺失时null")
    policy_frozen_at: datetime | None = Field(default=None, description="该快照的数据库冻结时间，不是本次检查时间")
    policy_freeze_hash: Hash | None = Field(default=None, description="匹配快照的完整内容指纹")
    policy_binding_hash: Hash | None = Field(default=None, description="已冻结比较约定的指纹")
    ex_ante_plan_verified: bool = Field(
        default=False, description="是否从数据库核验本次新研究在评估前选定了当前冻结计划"
    )
    planned_research_binding_id: UUID | None = Field(
        default=None, description="核验通过的研究绑定编号；旧报告或未核验时null"
    )
    planned_research_binding_hash: Hash | None = Field(
        default=None, description="绑定回执完整指纹，供追溯具体研究和计划"
    )
    exam_plan_hash: Hash | None = Field(
        default=None, description="绑定中固定考试日期计划的指纹，不是按可用数据倒推的日期"
    )
    exam_coverage_evidence: tuple[CashCoverageEvidence, ...] = Field(
        default=(), description="仅已完成考试窗口的逐基金分子/分母；未考试不以可用题数充当评分数"
    )
    publication_allowed: Literal[False] = Field(default=False, description="计划或单项成绩通过仍不是正式发布许可")
    inference_executed: Literal[False] = Field(default=False, description="只检查已有成绩，不执行预测公式")
    independent_test_read: Literal[False] = Field(default=False, description="不读取2025数值或答案")
    database_written: Literal[False] = Field(default=False, description="不保存审查回执、不修改报告、规则或模型")

    @model_validator(mode="after")
    def summary_matches_checks(self) -> Self:
        references = (self.policy_freeze_id, self.policy_frozen_at, self.policy_freeze_hash, self.policy_binding_hash)
        if self.policy_persisted:
            if any(v is None for v in references) or self.policy.approval_state != "APPROVED":
                raise ValueError("persisted policy requires an approved, complete snapshot reference")
        elif any(v is not None for v in references):
            raise ValueError("unfrozen policy cannot claim snapshot references")
        plan_references = (self.planned_research_binding_id, self.planned_research_binding_hash, self.exam_plan_hash)
        if self.ex_ante_plan_verified:
            if not self.policy_persisted or any(v is None for v in plan_references):
                raise ValueError("verified plan requires matching frozen policy and binding references")
        elif any(v is not None for v in plan_references) or self.exam_coverage_evidence:
            raise ValueError("unverified plan cannot claim binding references or coverage")
        evidence = {(e.window_id, e.fund_code): e for e in self.exam_coverage_evidence}
        measured = {
            (c.window_id, c.scope): c for c in self.checks if c.code == "EXAM_COVERAGE" and c.status != "MISSING"
        }
        if len(evidence) != len(self.exam_coverage_evidence) or evidence.keys() != measured.keys():
            raise ValueError("scored coverage evidence must match measured checks exactly")
        if len({item.dataset_hash for item in evidence.values()}) > 1:
            raise ValueError("one research cannot claim coverage from different datasets")
        for key, item in evidence.items():
            check = measured[key]
            with localcontext() as context:
                context.prec = 28
                ratio = Decimal(item.scored_count) / item.planned_count
            if (
                item.plan_hash != self.exam_plan_hash
                or item.fund_code not in self.policy.fund_codes
                or item.window_id not in {"DEV_2023_Q3", "DEV_2023_Q4", "VALIDATION_2024"}
                or check.actual != ratio
                or check.required != self.policy.minimum_coverage
                or check.operator != "GE"
            ):
                raise ValueError("coverage values, scope or plan contradict their evidence")
        if self.check_counts != {s: sum(c.status == s for c in self.checks) for s in ("PASS", "FAIL", "MISSING")}:
            raise ValueError("review counts do not match checks")
        if not self.blocking_codes or len(set(self.blocking_codes)) != len(self.blocking_codes):
            raise ValueError("blocked review requires distinct blocking reasons")
        return self
