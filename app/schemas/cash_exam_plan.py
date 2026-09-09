"""先列考试日期再检查资料；计划、资料覆盖与真实考试成绩不能混为一件事。"""

from datetime import date
from decimal import Decimal, localcontext
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.cash_reinvestment_research import CashPreparation, CashPrepareRequest
from app.schemas.historical_nav_training import Hash


class CashExamWindowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    window_id: str = Field(description="固定季度、年度验证或独立考试编号")
    role: Literal["TRAIN_INTERNAL_ROLLING", "FIXED_VALIDATION", "INDEPENDENT_TEST"]
    start_date: date = Field(description="考试输入截止日的起点，含当天")
    end_date: date = Field(description="考试输入及答案必须可得的末日，含当天")
    planned_cutoffs: tuple[date, ...] = Field(
        min_length=1, max_length=260, description="仅由日历决定的应考日期，未来20交易日终点不越过本窗口"
    )
    boundary_purged_cutoffs: tuple[date, ...] = Field(
        min_length=20, max_length=20, description="窗口最后20个交易日，答案终点越界，事先排除而非按成绩删题"
    )

    @model_validator(mode="after")
    def ordered_dates(self) -> Self:
        dates = (*self.planned_cutoffs, *self.boundary_purged_cutoffs)
        if self.start_date > self.end_date or any(not self.start_date <= d <= self.end_date for d in dates):
            raise ValueError("exam plan dates outside window")
        if any(a >= b for a, b in zip(dates[:-1], dates[1:], strict=True)):
            raise ValueError("exam plan dates must be strictly ordered and disjoint")
        return self


class CashExamPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["CASH_EX_ANTE_EXAM_PLAN_V1"] = "CASH_EX_ANTE_EXAM_PLAN_V1"
    fund_codes: tuple[Literal["001632", "006730", "008888"], ...] = Field(
        min_length=3, max_length=3, description="全部三试点共用日期计划，不能按覆盖率剔除基金"
    )
    calendar_version: Literal["CN_A_SHARE_2021_2025_V1"]
    calendar_hash: Hash = Field(description="固定官方交易日历的内容指纹，不从净值记录倒推日历")
    horizon_trading_days: Literal[20] = 20
    denominator_rule: Literal["ALL_TRADING_CUTOFFS_WITH_20TD_END_WITHIN_WINDOW"] = (
        "ALL_TRADING_CUTOFFS_WITH_20TD_END_WITHIN_WINDOW"
    )
    windows: tuple[CashExamWindowPlan, ...] = Field(min_length=4, max_length=4)
    plan_hash: Hash = Field(description="上述固定计划内容指纹，不含批次、数据好坏、成绩或当前时间")


class CashExamPreparationRequest(CashPrepareRequest):
    expected_dataset_hash: Hash = Field(alias="expectedDatasetHash", description="原preparation返回的明确资料指纹")
    expected_plan_hash: Hash = Field(alias="expectedPlanHash", description="GET release-policy中待核对的日期计划指纹")


class CashExamCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    window_id: str = Field(description="固定考试窗口编号")
    fund_code: str = Field(description="本行对应基金，不用其他基金的题补足")
    status: Literal["PREPARED", "TEST_PERIOD_PROTECTED"]
    planned_count: int = Field(gt=0, le=260, description="事前日期计划中的应考题数，不随选取的批次减少")
    usable_count: int | None = Field(default=None, ge=0, le=260, description="有输入且答案在窗口内可得的题数；未考试")
    coverage: Decimal | None = Field(default=None, ge=0, le=1, description="可用题数/计划题数，不是准确率")
    missing_cutoffs: tuple[date, ...] | None = Field(
        default=None, max_length=260, description="计划中未能准备完整资料的日期；2025未检查，保持null而不是空列表"
    )
    late_label_count: int | None = Field(
        default=None, ge=0, le=260, description="缺口中已有研究输入与答案、但答案公告晚于本窗口末日的题数"
    )

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        values = (self.usable_count, self.coverage, self.missing_cutoffs, self.late_label_count)
        if self.status == "TEST_PERIOD_PROTECTED":
            if any(v is not None for v in values):
                raise ValueError("protected test cannot contain data coverage evidence")
        elif any(v is None for v in values):
            raise ValueError("prepared coverage requires counts and missing dates")
        else:
            if self.usable_count + len(self.missing_cutoffs) != self.planned_count:
                raise ValueError("coverage population mismatch")
            if self.late_label_count > len(self.missing_cutoffs):
                raise ValueError("late label count exceeds missing dates")
            if tuple(sorted(set(self.missing_cutoffs))) != self.missing_cutoffs:
                raise ValueError("missing dates must be unique and sorted")
            with localcontext() as context:
                context.prec = 28
                if abs(self.coverage * self.planned_count - self.usable_count) > Decimal("1e-20"):
                    raise ValueError("coverage ratio mismatch")
        return self


class CashExamPreparation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["READ_ONLY_EXAM_PREPARATION"] = "READ_ONLY_EXAM_PREPARATION"
    plan: CashExamPlan = Field(description="独立于所选资料生成的日历清单，不是审批或考试成绩")
    preparation: CashPreparation = Field(description="复用已核验的研究资料准备结果，不修改原版本或指纹")
    coverage: tuple[CashExamCoverage, ...] = Field(min_length=12, max_length=12)
    ex_ante_evidence: Literal[False] = Field(default=False, description="本只读预览不证明历史报告曾事先绑定本计划")
    publication_allowed: Literal[False] = False
    model_fitted: Literal[False] = False
    independent_test_read: Literal[False] = False
    database_written: Literal[False] = False

    @model_validator(mode="after")
    def coverage_matches_plan(self) -> Self:
        expected = {(w.window_id, fund) for w in self.plan.windows for fund in self.plan.fund_codes}
        if {(c.window_id, c.fund_code) for c in self.coverage} != expected:
            raise ValueError("coverage must contain every planned window and fund exactly once")
        for group in self.coverage:
            window = next(w for w in self.plan.windows if w.window_id == group.window_id)
            if group.planned_count != len(window.planned_cutoffs):
                raise ValueError("coverage denominator differs from plan")
            if (group.status == "TEST_PERIOD_PROTECTED") != (window.role == "INDEPENDENT_TEST"):
                raise ValueError("independent test must stay protected")
            if group.missing_cutoffs is not None and not set(group.missing_cutoffs).issubset(window.planned_cutoffs):
                raise ValueError("missing cutoff outside plan")
        return self
