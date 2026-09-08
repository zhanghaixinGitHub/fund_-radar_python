"""固定时间隔离校准和滚动研究验证的契约；不是发布或开放测试成绩。"""

from datetime import date
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from app.schemas.historical_nav_evaluation import BaselineComparison, HistoricalNavEvaluationResponse
from app.schemas.historical_nav_training import Hash, HistoricalNavTrainingRequest, LogisticModelArtifact


class HistoricalNavCalibrationRequest(HistoricalNavTrainingRequest):
    """复用候选请求，V1仅接受已经冻结的2022–2025时间协议，不接受自定义窗口。"""

    @model_validator(mode="after")
    def frozen_dates(self) -> Self:
        actual = (self.train_start_date, self.train_end_date, self.validation_end_date, self.test_end_date)
        if actual != (date(2022, 1, 1), date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)):
            raise ValueError("calibration V1 requires frozen 2022-2025 boundaries")
        return self


class CalibrationWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    window_id: str = Field(description="固定窗口编号，不根据成绩改日期")
    role: Literal["TRAIN_INTERNAL_ROLLING", "FIXED_VALIDATION"] = Field(description="训练内部滚动或2024固定验收")
    fit_end_date: date = Field(description="基础模型输入与答案可得截止；起点固定2022-01-01")
    calibration_end_date: date = Field(description="校准输入晚于fit_end_date，标签不晚于本日")
    evaluation_end_date: date = Field(description="考试输入晚于校准截止，标签不晚于本日")
    minimum_exam_per_fund: int = Field(description="季度40、2024全年120；不替代全局252/120/120门槛")


class CalibrationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["NAV_TEMPORAL_CALIBRATION_V1"] = Field(
        default="NAV_TEMPORAL_CALIBRATION_V1", description="冻结实验版本"
    )
    method: Literal["REGULARIZED_SIGMOID_L2"] = Field(
        default="REGULARIZED_SIGMOID_L2", description="sigmoid(a*z+b)，非完整Platt软标签算法"
    )
    minimum_fit_per_fund: int = Field(default=252, description="过滤后每基金最低基础拟合样本量")
    minimum_calibration_per_fund: int = Field(default=60, description="过滤后每基金最低独立校准样本量")
    bin_edges: tuple[float, ...] = Field(
        default=(0, 0.2, 0.4, 0.6, 0.8, 1), description="事先固定五个可靠性分数档，不按成绩分档"
    )
    minimum_bin_count: int = Field(default=30, description="每档不足30条则提示样本少，不自动填补或合并")
    windows: tuple[CalibrationWindow, ...] = Field(description="先固定的两个内部滚动窗口与一个2024验证窗口")


class SigmoidCalibrator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    method: Literal["REGULARIZED_SIGMOID_L2"] = Field(
        default="REGULARIZED_SIGMOID_L2", description="单变量L2逻辑回归映射"
    )
    slope: FiniteFloat = Field(description="基础线性得分z的权重a；非正斜率另报风险，不无声翻转")
    intercept: FiniteFloat = Field(description="校准基础偏移b，不是直接概率")
    c: Literal[1] = Field(default=1, description="正则强度倒数固定1，不搜索")
    iterations: int = Field(ge=1, le=1000, description="校准器实际求解迭代次数，不是发布门槛")
    start_date: date = Field(description="独立校准输入可得日期下限，含当天")
    end_date: date = Field(description="校准输入与答案可得日期上限")
    sample_count: int = Field(ge=1, description="用于拟合校准器的行数，不含考试行")
    counts_per_fund: dict[str, int] = Field(description="每基金校准样本量")
    weights_per_fund: dict[str, FiniteFloat] = Field(description="每基金每条校准行的权重，使基金总权重相同")
    class_counts: dict[str, int] = Field(description="仅校准段0/1答案数量")
    calibration_hash: Hash = Field(description="校准样本内容指纹，不含考试答案")
    base_model_hash: Hash = Field(description="只能配套使用的冻结基础模型哈希，不允许事后重训替换")

    @model_validator(mode="after")
    def validate_metadata(self) -> Self:
        if self.start_date > self.end_date or sum(self.counts_per_fund.values()) != self.sample_count:
            raise ValueError("calibration dates/counts mismatch")
        if not self.counts_per_fund or any(n <= 0 for n in self.counts_per_fund.values()):
            raise ValueError("calibration fund counts must be positive")
        if set(self.weights_per_fund) != set(self.counts_per_fund) or any(
            w <= 0 for w in self.weights_per_fund.values()
        ):
            raise ValueError("calibration weights mismatch")
        if set(self.class_counts) != {"0", "1"} or min(self.class_counts.values()) <= 0:
            raise ValueError("calibration requires both classes")
        if sum(self.class_counts.values()) != self.sample_count:
            raise ValueError("calibration class counts mismatch")
        return self


class CalibratedModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["NAV_CALIBRATED_MODEL_V1"] = Field(
        default="NAV_CALIBRATED_MODEL_V1", description="组合JSON模型版本"
    )
    purpose: Literal["LEARNING_ONLY"] = Field(default="LEARNING_ONLY", description="已校准仍仅研究，不保证概率可靠")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(default="MODEL_NOT_RELEASED", description="未发布")
    base_model: LogisticModelArtifact = Field(description="冻结的基础七指标模型，不再使用校准段重训")
    calibrator: SigmoidCalibrator = Field(description="仅由独立校准段拟合的a/b参数")
    model_hash: Hash = Field(description="完整组合模型内容指纹，不含考试数据/窗口成绩")


class ReliabilityBin(BaseModel):
    lower: Decimal = Field(description="该档下界，包含")
    upper: Decimal = Field(description="该档上界，只有最后一档包含上界1")
    count: int = Field(description="考试样本落入本档的数量")
    mean_score: Decimal | None = Field(description="本档平均预测分数；空档null")
    observed_up_rate: Decimal | None = Field(description="本档历史实际上涨比例；空档null，不伪造0")
    absolute_gap: Decimal | None = Field(description="平均分数与真实上涨比例差的绝对值")
    enough_samples: bool = Field(description="是否至少30条；仍不代表统计独立或概率已证实")


class ReliabilityReport(BaseModel):
    sample_count: int = Field(description="考试行数，不含拟合或校准训练行")
    ece: Decimal = Field(description="各档绝对偏差按数量加权，越小越好；受分箱选择和样本量影响")
    bins: tuple[ReliabilityBin, ...] = Field(description="固定五档，空档也保留")


class FundReliability(BaseModel):
    fund_code: str = Field(description="基金代码，用于分组，不是模型输入")
    before: ReliabilityReport = Field(description="同一基础模型校准前的可靠性分箱")
    after: ReliabilityReport = Field(description="同一基础模型校准后的可靠性分箱")


class WindowFundCounts(BaseModel):
    fund_code: str = Field(description="基金代码")
    counts: dict[str, int] = Field(description="FIT/CALIBRATION/EXAM可用数量")
    missing: dict[str, int] = Field(description="各段距离本窗口最低数量的缺口")
    purged: dict[str, int] = Field(description="全局过滤后本窗新增跨界剔除数；已有全局剔除见preparation，不重复计数")


class CalibrationPreview(BaseModel):
    fund_code: str = Field(description="历史考试基金")
    as_of_date: date = Field(description="历史考试净值业务日，不是今天的预测")
    before_score: Decimal = Field(description="基础模型未校准分数")
    after_score: Decimal = Field(description="校准后的研究分数，不能直接当产品概率")
    actual_up: Literal[0, 1] = Field(description="该考试段历史答案，不含2025答案")


class CalibrationWindowReport(BaseModel):
    window: CalibrationWindow = Field(description="固定时间窗口和角色")
    status: Literal["EVALUATED", "INSUFFICIENT_DATA", "REJECTED"] = Field(
        description="是否实际完成本窗；不足或异常不伪造成绩"
    )
    reason: str | None = Field(default=None, description="拒绝/不足原因码")
    funds: tuple[WindowFundCounts, ...] = Field(description="逐基金数量、剔除与缺口")
    model: CalibratedModelArtifact | None = Field(default=None, description="本窗基础模型和校准器JSON")
    before: BaselineComparison | None = Field(
        default=None, description="同一基础模型校准前考试成绩；validation字段指本窗考试"
    )
    after: BaselineComparison | None = Field(default=None, description="校准后考试成绩，总体及逐基金")
    baselines: tuple[BaselineComparison, ...] = Field(
        default=(), description="四种相同考试集合的朴素对照，频率只取考前历史"
    )
    reliability_before: ReliabilityReport | None = Field(default=None, description="总体校准前分箱与ECE")
    reliability_after: ReliabilityReport | None = Field(default=None, description="总体校准后分箱与ECE")
    reliability_per_fund: tuple[FundReliability, ...] = Field(default=(), description="逐基金校准前后可靠性")
    brier_delta: Decimal | None = Field(
        default=None, description="校准后减校准前Brier，负数更好，但不能单独证明校准改善"
    )
    ece_delta: Decimal | None = Field(default=None, description="校准后减校准前ECE，负数更好，不是发布门槛")
    warnings: tuple[str, ...] = Field(default=(), description="非正映射、样本少、固定验证已被观察等风险")
    prediction_preview: tuple[CalibrationPreview, ...] = Field(
        default=(), description="本窗最多previewSize条历史考试示例"
    )


class HistoricalNavCalibrationResponse(BaseModel):
    mode: Literal["OFFLINE_CALIBRATION_EVALUATION"] = Field(
        default="OFFLINE_CALIBRATION_EVALUATION", description="时间隔离研究验证"
    )
    status: Literal["CALIBRATION_EVALUATED", "PARTIAL_EVALUATION", "INSUFFICIENT_DATA", "NO_VALID_WINDOWS"] = Field(
        description="全部、部分、不足或所有窗口失败"
    )
    purpose: Literal["LEARNING_ONLY"] = Field(default="LEARNING_ONLY", description="研究用途")
    database_written: Literal[False] = Field(default=False, description="不写样本或模型表")
    artifact_persisted: Literal[False] = Field(default=False, description="HTTP不保存文件，CLI另写本机产物")
    training_eligible: Literal[False] = Field(default=False, description="严格历史数据准入尚未通过")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(default="MODEL_NOT_RELEASED", description="绝不自动激活")
    protocol: CalibrationProtocol = Field(description="事先固定的校准方法/分箱/窗口")
    preparation: HistoricalNavEvaluationResponse = Field(
        description="原全局数据校验及指纹；旧基线是全局2024结果，逐窗基线另列"
    )
    windows: tuple[CalibrationWindowReport, ...] = Field(description="固定三窗，不按结果筛选")
    evaluated_window_count: int = Field(description="实际完整考试的窗口数，不是重复训练次数")
    brier_improved_window_count: int = Field(description="校准后Brier严格变小的窗口数，不作为上线投票")
    limitations: tuple[str, ...] = Field(description="首次可得/交易日历/小样本/未发布等限制")
