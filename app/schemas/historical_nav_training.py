"""研究候选训练与可读模型产物；不接收调参、发布或任意文件路径。"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from app.schemas.historical_nav_evaluation import (
    BaselineComparison,
    HistoricalNavEvaluationRequest,
    HistoricalNavEvaluationResponse,
)

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SevenNumbers = Annotated[tuple[FiniteFloat, ...], Field(min_length=7, max_length=7)]


class HistoricalNavTrainingRequest(HistoricalNavEvaluationRequest):
    """沿用基线请求，再加已经确认的数据指纹，避免无意训练另一份数据。"""

    expected_dataset_hash: Hash = Field(alias="expectedDatasetHash", description="上一步基线报告中的dataset_hash")

    def evaluation_request(self) -> HistoricalNavEvaluationRequest:
        return HistoricalNavEvaluationRequest.model_validate(self.model_dump(exclude={"expected_dataset_hash"}))


class CandidateProtocol(BaseModel):
    """固定实验方案；不是允许客户端自由调整的训练参数。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    version: Literal["HISTORICAL_NAV_LOGISTIC_V1"] = Field(
        default="HISTORICAL_NAV_LOGISTIC_V1", description="算法及产物版本，改训练方案须升级"
    )
    algorithm: Literal["LOGISTIC_REGRESSION"] = Field(default="LOGISTIC_REGRESSION", description="逻辑回归二分类")
    regularization: Literal["L2"] = Field(default="L2", description="约束系数大小，减少对偶然历史的过度拟合")
    c: float = Field(default=1.0, description="正则强度的倒数，本轮固定1，不按验证成绩搜索")
    solver: Literal["lbfgs"] = Field(default="lbfgs", description="求解系数的方法")
    tolerance: float = Field(default=1e-8, description="求解器停止精度，不是准确率或发布门槛")
    max_iterations: int = Field(default=1000, description="最多迭代次数；未收敛则拒绝交付模型")
    random_seed: Literal[0] = Field(default=0, description="保留固定种子；当前lbfgs不使用随机种子")
    weighting: Literal["EQUAL_TOTAL_WEIGHT_PER_FUND"] = Field(
        default="EQUAL_TOTAL_WEIGHT_PER_FUND", description="每基金总训练权重相同，不是按涨跌类别均衡"
    )
    preprocessing: Literal["TRAIN_WEIGHTED_STANDARD_SCALER"] = Field(
        default="TRAIN_WEIGHTED_STANDARD_SCALER", description="只用训练行拟合加权均值和尺度，常量列尺度1"
    )
    missing_value_policy: Literal["REJECT"] = Field(default="REJECT", description="缺失/非有限数值拒绝，不填0")
    calibration: Literal["NONE"] = Field(default="NONE", description="本轮未做概率校准，输出只是未校准上涨分数")
    direction_threshold: float = Field(default=0.5, description="严格大于0.5算上涨，与四种对照一致")
    test_policy: Literal["HELD_OUT_NOT_SCORED"] = Field(default="HELD_OUT_NOT_SCORED", description="不输出测试成绩")


class LogisticModelArtifact(BaseModel):
    """纯JSON模型，无可执行对象；预测公式为sigmoid(截距+系数·标准化输入)。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    protocol: CandidateProtocol = Field(description="本模型固定训练方案")
    purpose: Literal["LEARNING_ONLY"] = Field(default="LEARNING_ONLY", description="只能用于离线研究")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(default="MODEL_NOT_RELEASED", description="未上线")
    feature_names: tuple[str, ...] = Field(min_length=7, max_length=7, description="七个输入指标的严格顺序")
    mean: SevenNumbers = Field(description="训练段各指标加权平均值；验证/测试不参与")
    scale: SevenNumbers = Field(description="训练段各指标尺度，标准化=(输入-mean)/scale；常量列为1")
    coefficients: SevenNumbers = Field(description="每个标准化指标的学习权重；正负不代表因果或独立重要性")
    intercept: FiniteFloat = Field(description="所有标准化指标都为0时的基础得分，不是直接的概率")
    classes: tuple[Literal[0], Literal[1]] = Field(default=(0, 1), description="0非上涨、1上涨；输出取类别1")
    iterations: int = Field(ge=1, le=1000, description="本次求解实际迭代次数")
    train_hash: Hash = Field(description="仅训练样本内容的指纹；不是完整训练/验证/测试数据指纹")
    train_start_date: date = Field(description="训练输入可得日期下限")
    train_end_date: date = Field(description="训练输入和答案可得日期上限")
    train_count: int = Field(ge=1, description="实际拟合的训练行数")
    train_counts_per_fund: dict[str, int] = Field(description="各基金参与训练的行数")
    sample_weight_per_fund: dict[str, FiniteFloat] = Field(description="各基金每一条训练行的权重，不是预测分数")
    train_class_counts: dict[str, int] = Field(description="训练段非上涨0/上涨1数量，不含验证或测试答案")
    versions: dict[str, str] = Field(description="输入特征、样本与标签规则版本")
    runtime_versions: dict[str, str] = Field(description="Python与数值依赖版本；跨平台仍可能有浮点差异")
    model_hash: Hash = Field(description="除本字段外完整模型JSON内容的SHA256，不是签名或发布凭证")

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if any(s <= 0 for s in self.scale):
            raise ValueError("scale must be positive")
        if self.train_start_date > self.train_end_date:
            raise ValueError("invalid training dates")
        if sum(self.train_counts_per_fund.values()) != self.train_count:
            raise ValueError("training counts mismatch")
        if any(n <= 0 for n in self.train_counts_per_fund.values()):
            raise ValueError("fund training counts must be positive")
        if set(self.sample_weight_per_fund) != set(self.train_counts_per_fund):
            raise ValueError("fund weight keys mismatch")
        if any(w <= 0 for w in self.sample_weight_per_fund.values()):
            raise ValueError("weights must be positive")
        if set(self.train_class_counts) != {"0", "1"} or min(self.train_class_counts.values()) <= 0:
            raise ValueError("require two training classes")
        if sum(self.train_class_counts.values()) != self.train_count:
            raise ValueError("class counts mismatch")
        return self


class CandidateBaselineDelta(BaseModel):
    baseline_id: str = Field(description="比较的朴素对照编号")
    accuracy_delta: Decimal = Field(description="候选准确率减基线准确率；正数更好，0.01代表1个百分点")
    balanced_accuracy_delta: Decimal | None = Field(description="候选平衡准确率减基线；单类验证时null")
    brier_delta: Decimal = Field(description="候选Brier减基线Brier，负数更好；不是发布结论")


class CandidatePredictionPreview(BaseModel):
    fund_code: str = Field(description="验证样本所属基金，不是模型输入")
    as_of_date: date = Field(description="历史验证样本净值业务日，不是今天的预测")
    up_score: Decimal = Field(description="未校准上涨分数，0到1，不是已证明有效的产品概率")
    predicted_up: Literal[0, 1] = Field(description="分数严格大于0.5为1，否则0")
    actual_up: Literal[0, 1] = Field(description="验证段历史真实答案；不输出测试答案")


class HistoricalNavTrainingResponse(BaseModel):
    mode: Literal["OFFLINE_CANDIDATE_TRAINING"] = Field(
        default="OFFLINE_CANDIDATE_TRAINING", description="离线研究训练"
    )
    status: Literal["INSUFFICIENT_DATA", "CANDIDATE_EVALUATED"] = Field(description="数量不足不训练；成功不等于上线")
    purpose: Literal["LEARNING_ONLY"] = Field(default="LEARNING_ONLY", description="仅供研究学习")
    database_written: Literal[False] = Field(default=False, description="没有写样本或模型数据库")
    artifact_persisted: Literal[False] = Field(default=False, description="服务只返回JSON；本机CLI另外保存文件")
    training_eligible: Literal[False] = Field(default=False, description="仍未取得正式历史数据训练资格")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(
        default="MODEL_NOT_RELEASED", description="未开放线上预测"
    )
    protocol: CandidateProtocol = Field(description="冻结的候选方案")
    preparation: HistoricalNavEvaluationResponse = Field(description="复用的数据准备及四种基线报告")
    model: LogisticModelArtifact | None = Field(default=None, description="可读模型JSON；数据不足为null")
    candidate: BaselineComparison | None = Field(default=None, description="候选总体及逐基金验证成绩；不含测试成绩")
    baseline_deltas: tuple[CandidateBaselineDelta, ...] = Field(
        default=(), description="相同验证集合上的差值，不选赢家"
    )
    prediction_preview: tuple[CandidatePredictionPreview, ...] = Field(default=(), description="最多20条验证预测示例")
    limitations: tuple[str, ...] = Field(description="概率校准、历史质量及发布限制")
