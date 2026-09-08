"""候选数据集与基线验证契约；不是训练模型、发布接口或全量数据导出。"""

from datetime import date
from decimal import Decimal
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HistoricalNavEvaluationRequest(BaseModel):
    """显式选择已存批次与统一时间边界，避免自动采用最新批次造成不可复现。"""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    batch_ids: tuple[UUID, ...] = Field(
        alias="batchIds", min_length=1, max_length=512, description="明确选择的批次编号，最多512个且不能重复"
    )
    train_start_date: date = Field(alias="trainStartDate", description="训练输入最早可得日，含当天；不是批次创建时间")
    train_end_date: date = Field(alias="trainEndDate", description="训练段截止，输入和答案均须在该日结束前可得")
    validation_end_date: date = Field(
        alias="validationEndDate", description="验证段截止；输入晚于训练截止，答案不晚于本日"
    )
    test_end_date: date = Field(alias="testEndDate", description="保留测试段截止；本轮不输出测试成绩或测试样本答案")
    preview_size: int = Field(
        alias="previewSize",
        default=5,
        ge=0,
        le=20,
        strict=True,
        description="最多展示多少条训练/验证样本，不影响数据指纹和评估",
    )

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if len(set(self.batch_ids)) != len(self.batch_ids):
            raise ValueError("batchIds must not contain duplicates")
        if not self.train_start_date <= self.train_end_date < self.validation_end_date < self.test_end_date:
            raise ValueError("require trainStartDate <= trainEndDate < validationEndDate < testEndDate")
        return self


class EvaluationProtocol(BaseModel):
    """随报告返回固定规则；客户端不能下调样本量或改变比较阈值。"""

    version: Literal["HISTORICAL_NAV_BASELINE_EVALUATION_V1"] = Field(
        default="HISTORICAL_NAV_BASELINE_EVALUATION_V1", description="候选数据与比较算法版本，规则变化须升级"
    )
    train_start_date: date = Field(description="训练段输入可得日期下限，包含")
    train_end_date: date = Field(description="训练段输入及标签可得日期上限，包含")
    validation_end_date: date = Field(description="验证段输入及标签可得日期上限，包含")
    test_end_date: date = Field(description="保留测试段输入及标签可得日期上限，包含")
    minimum_samples_per_fund: dict[str, int] = Field(
        default_factory=lambda: {"TRAIN": 252, "VALIDATION": 120, "TEST": 120},
        description="隔离、去重、质量过滤后每只基金至少需要的样本数，不是原始净值行数",
    )
    nav_value_basis: Literal["ACCUMULATED_NAV"] = Field(
        default="ACCUMULATED_NAV", description="本版候选集只接收累计净值口径，不混入单位净值回退样本"
    )
    split_clock: Literal["AVAILABLE_AT"] = Field(
        default="AVAILABLE_AT", description="按输入实际可得日期划分，不按批次保存时间或随机分组"
    )
    label_cutoff_rule: Literal["LABEL_AVAILABLE_AT_LTE_SPLIT_END"] = Field(
        default="LABEL_AVAILABLE_AT_LTE_SPLIT_END", description="答案跨越所属段截止日则剔除，不用20个自然日代替隔离"
    )
    direction_threshold: Decimal = Field(
        default=Decimal("0.5"), description="所有对照统一按分数严格大于0.5算上涨，等于0.5算非上涨"
    )
    test_policy: Literal["HELD_OUT_NOT_SCORED"] = Field(
        default="HELD_OUT_NOT_SCORED", description="本轮测试段仅准备和校验，不展示成绩、上涨比例或答案预览"
    )


class PreparedSamplePreview(BaseModel):
    """少量训练/验证样本；X和y分开，原始载荷与元数据不能整体喂给模型。"""

    split: Literal["TRAIN", "VALIDATION"] = Field(description="样本所属段，不输出测试段样本")
    batch_id: UUID = Field(description="保留样本所属批次，重叠相同样本固定选编号最小的批次")
    fund_code: str = Field(description="基金归属，仅用于分组追溯，不属于X")
    as_of_date: date = Field(description="净值业务日，仅用于追溯")
    available_at: date = Field(description="输入可得日期，也是时间划分依据")
    label_available_at: date = Field(description="历史答案最晚可得日期，不能进入X")
    x: tuple[Decimal, ...] = Field(description="与feature_names严格对应的7个数值，既不含基金编号，也不含答案")
    y: Literal[0, 1] = Field(description="历史真实上涨1/非上涨0，与X分开；不是预测值")


class FundPreparationSummary(BaseModel):
    """逐基金报告缺口，避免把多个基金的少量样本相加冒充单基金足够。"""

    fund_code: str = Field(description="基金代码")
    unique_sample_count: int = Field(description="跨批次去重后的样本数，含拒收项")
    split_counts: dict[str, int] = Field(description="TRAIN/VALIDATION/TEST最终可用样本数")
    missing_samples: dict[str, int] = Field(description="每段距离固定最低数量还缺多少条")
    excluded_reasons: dict[str, int] = Field(description="互斥剔除原因到数量，每个去重样本最多计一次")
    train_up_rate: Decimal | None = Field(description="仅用该基金训练段答案计算的历史上涨比例，不是未来预测概率")
    warnings: tuple[str, ...] = Field(description="如训练段只有一种答案；本轮基线可以比较，但不表示可训练正式模型")


class BaselineMetrics(BaseModel):
    """同一验证样本集合上的指标；小数以Decimal序列化，避免NaN。"""

    sample_count: int = Field(description="参加验证的样本数")
    correct_count: int = Field(description="方向判断正确数")
    actual_up_count: int = Field(description="真实上涨数")
    predicted_up_count: int = Field(description="对照判断上涨数")
    accuracy: Decimal = Field(description="正确数/验证样本数，越高越好")
    balanced_accuracy: Decimal | None = Field(
        description="上涨与非上涨各自召回率的平均；只出现一类答案时为null，不伪造满分"
    )
    brier_score: Decimal = Field(description="分数与真实0/1答案差的平方均值，越低越好；规则分数未经过概率校准")


class BaselineFundMetrics(BaseModel):
    fund_code: str = Field(description="该组基金代码")
    metrics: BaselineMetrics = Field(description="该基金的验证成绩")


class BaselineComparison(BaseModel):
    baseline_id: str = Field(description="固定对照方法编号，不代表训练模型版本")
    description: str = Field(description="这条对照规则具体怎样计算")
    validation: BaselineMetrics = Field(description="全部基金按样本汇总的验证指标")
    per_fund: tuple[BaselineFundMetrics, ...] = Field(description="逐基金验证指标，避免总体分数掩盖差异")


class HistoricalNavEvaluationResponse(BaseModel):
    """只读、确定性的学习报告；不生成已发布模型或数据库运行记录。"""

    mode: Literal["READ_ONLY_BASELINE_EVALUATION"] = Field(
        default="READ_ONLY_BASELINE_EVALUATION", description="只读学习评估，不保存样本或模型"
    )
    status: Literal["INSUFFICIENT_DATA", "BASELINE_EVALUATED"] = Field(description="不足时保留数据诊断、不返回基线成绩")
    purpose: Literal["LEARNING_ONLY"] = Field(default="LEARNING_ONLY", description="仅供学习研究，不是发布用途")
    persisted: Literal[False] = Field(default=False, description="本报告和候选矩阵未写入数据库；可保存请求/响应供复现")
    training_eligible: Literal[False] = Field(default=False, description="历史首次可得等限制未解决，不授予正式训练资格")
    publication_status: Literal["MODEL_NOT_RELEASED"] = Field(
        default="MODEL_NOT_RELEASED", description="没有发布模型，不输出产品预测"
    )
    protocol: EvaluationProtocol = Field(description="本次固定时间边界、数量和隔离规则")
    batch_ids: tuple[UUID, ...] = Field(description="排序后的输入批次清单；批次顺序不影响结果")
    source_code: str = Field(description="批次统一来源编码，不作为模型特征")
    source_sync_run_ids: tuple[UUID, ...] = Field(description="涉及的来源水位清单，不是首次公告证明")
    versions: dict[str, str] = Field(description="统一的特征、样本筛选、标签版本")
    feature_names: tuple[str, ...] = Field(description="X的固定7个指标顺序")
    dataset_hash: str = Field(description="请求时间规则和完整输入批次内容的指纹；不是落库报告编号")
    train_hash: str = Field(description="仅训练样本内容的指纹，用于验证后续答案变化没有改变训练输入")
    input_sample_count: int = Field(description="输入全部批次的样本总数，未去重")
    duplicate_sample_count: int = Field(description="跨批次内容完全相同而合并的样本数")
    unique_sample_count: int = Field(description="去重后样本数")
    included_sample_count: int = Field(description="进入三个时间段的样本总数，含保留测试样本")
    excluded_reasons: dict[str, int] = Field(description="去重后被排除的数量汇总，与纳入数量之和等于去重总数")
    funds: tuple[FundPreparationSummary, ...] = Field(description="逐基金各段数量、缺口和训练上涨频率")
    baselines: tuple[BaselineComparison, ...] = Field(
        description="仅在所有基金达到数量门槛时返回四种对照的验证成绩；不排名或自动选择赢家"
    )
    sample_preview: tuple[PreparedSamplePreview, ...] = Field(description="训练/验证少量X与y示例，不含测试段")
    limitations: tuple[str, ...] = Field(description="首次可得、交易日历、非发布资格等必须保留的解释")
