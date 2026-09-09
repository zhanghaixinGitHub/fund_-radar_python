"""发布规则草案与核验明细；规则审批/持久冻结、数值合格和正式发布是三件不同的事。"""

from datetime import date
from decimal import Decimal
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.historical_nav_training import Hash


class CashReleasePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["CASH_RELEASE_POLICY_V1"] = Field(description="规则版本；改规则必须另立版本，不改旧冻结记录")
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
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str = Field(description="核验项编号，不是基金涨跌结论")
    window_id: str = Field(description="在哪个固定窗口检查；GLOBAL表示整个发布流程")
    scope: str = Field(description="ALL表示总体；基金代码表示逐基金，不混为一次通过")
    status: Literal["PASS", "FAIL", "MISSING"] = Field(description="符合候选规则、不符合或证据缺失；PASS不是发布许可")
    actual: Decimal | None = Field(default=None, description="实际核验数值；缺证据保持null，不填0")
    required: Decimal | None = Field(default=None, description="对应规则边界，比较含义由code/message说明")
    message: str = Field(description="通俗核验说明")


class CashCoverageEvidence(BaseModel):
    """未来事前日历计划生成器的服务内契约；当前HTTP不接受外部填报这些数。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    plan_version: Literal["CASH_EX_ANTE_EXAM_PLAN_V1"]
    plan_hash: Hash = Field(description="事前计划指纹；不能仅有一个自报通过标记")
    dataset_hash: Hash = Field(description="所核验的明确资料指纹")
    window_id: str = Field(description="固定考试窗口编号")
    fund_code: str = Field(description="明确基金，不使用被筛选后的总体数字代替")
    planned_count: int = Field(gt=0, le=1000, strict=True, description="按冻结窗口及日历事前确定的可考试截点数")
    scored_count: int = Field(ge=0, le=1000, strict=True, description="其中实际能完整评分的数量")

    @model_validator(mode="after")
    def counts(self) -> Self:
        if self.scored_count > self.planned_count:
            raise ValueError("scored count cannot exceed plan")
        return self


class CashReleaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["READ_ONLY_RELEASE_RULE_REVIEW"] = "READ_ONLY_RELEASE_RULE_REVIEW"
    status: Literal["BLOCKED"] = "BLOCKED"
    research_run_id: UUID = Field(description="实际重读和核验的研究编号")
    report_hash: Hash = Field(description="原研究指纹，不修改研究报告")
    policy_hash: Hash = Field(description="本次候选规则内容指纹，不能当作已冻结凭证")
    policy: CashReleasePolicy = Field(description="实际使用的服务器规则；草案同样可作诊断，但不能发布")
    blocking_codes: tuple[str, ...] = Field(description="未解除的证据、规则冻结和成绩问题")
    checks: tuple[CashReleaseCriterion, ...] = Field(description="逐窗口、逐基金的候选规则核验明细")
    policy_persisted: Literal[False] = False
    publication_allowed: Literal[False] = False
    inference_executed: Literal[False] = False
    independent_test_read: Literal[False] = False
    database_written: Literal[False] = False
