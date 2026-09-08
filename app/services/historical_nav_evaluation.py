"""已存学习样本 -> 候选X/y -> 验证段基线报告；不训练、不写库、不解除发布限制。"""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from time import perf_counter
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_sample_storage_engine
from app.repositories.historical_nav_evaluation import iter_batch_rows
from app.schemas.historical_nav_evaluation import (
    BaselineComparison,
    BaselineFundMetrics,
    BaselineMetrics,
    EvaluationProtocol,
    FundPreparationSummary,
    HistoricalNavEvaluationRequest,
    HistoricalNavEvaluationResponse,
    PreparedSamplePreview,
)
from app.schemas.historical_nav_storage import HistoricalNavStoredBatch
from app.services.historical_nav_samples import (
    HISTORICAL_NAV_FEATURE_VERSION,
    HISTORICAL_NAV_LABEL_VERSION,
    HISTORICAL_NAV_SAMPLE_RULE_VERSION,
    HistoricalNavSample,
)
from app.services.historical_nav_storage import HistoricalNavStorageError, restore_stored_batch
from app.services.historical_nav_storage_validation import validate_batch_samples
from app.services.momentum_baseline import fixed_momentum_score

# 固定X列顺序，不依赖JSON键顺序，也不将整个feature_payload转换成训练输入。
FEATURE_NAMES = (
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "max_drawdown_60d",
    "relative_position_60d",
    "consecutive_decline_days",
)
SPLITS = ("TRAIN", "VALIDATION", "TEST")
LIMITATIONS = (
    "LEARNING_ONLY：本报告不是正式训练许可或预测发布凭证。",
    "当前快照未恢复历史修订，来源水位不证明净值首次可得日期。",
    "20日标签沿用后20条有效净值区间；严格交易日历和分红复权仍待核验。",
    "本版只接收累计净值口径，单位净值回退项单独计为剔除。",
    "测试段只在内存准备及完整性校验，不输出成绩、类别比例或样本答案；尚无数据库级封存权限。",
    "本轮是固定单次时间划分的学习验证，不是滚动回测，也不选择赢家或调整发布阈值。",
    "规则分数和训练上涨频率未经概率校准，不能作为产品预测概率；重叠20日标签也不是独立样本。",
)


class HistoricalNavEvaluationError(RuntimeError):
    """稳定的可对外展示错误；不带SQL、Token或完整样本内容。"""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class PreparedRow:
    """内部候选矩阵的一行：x只装历史指标，y单独装答案，其他字段仅用于审计。"""

    batch_id: UUID  # 对应已存批次；不是模型特征。
    fund_code: str  # 分组评估与训练频率的归属；不是模型特征。
    as_of_date: date  # 原始净值日，配合批次可定位原样本。
    available_at: date  # 已知输入的公告截止，时间划分使用它。
    label_available_at: date  # 答案何时才可用于历史训练，不能放入x。
    x: tuple[Decimal, ...]  # 固定FEATURE_NAMES顺序的7个历史指标。
    y: int  # 历史真实方向：上涨1，否则0。
    content_hash: str  # 完整原样本的指纹，包含独立标签，用于复现而非训练。


@dataclass(frozen=True)
class PreparedDataset:
    """本次调用内的有界候选集；不保存为模型文件，test只交给后续正式评估流程。"""

    train: tuple[PreparedRow, ...]  # 仅这些行的y用于计算每基金历史上涨频率。
    validation: tuple[PreparedRow, ...]  # 本轮只对这一段计算基线成绩。
    test: tuple[PreparedRow, ...]  # 保留最终测试段，本轮不对外预览或评分。
    report: HistoricalNavEvaluationResponse  # 有界、可序列化的HTTP学习报告，不含全矩阵。


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, default=str, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _hash_parts(parts: Iterable[object]) -> str:
    """逐份散列而非拼接全量JSON；长度前缀保证相邻片段没有歧义。"""
    digest = hashlib.sha256()
    for part in parts:
        data = _canonical(part)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _check_deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise HistoricalNavEvaluationError("EVALUATION_TIMEOUT", "样本准备或评估超时，请减少批次范围后重试。", 503)


def evaluate_stored_historical_nav_batches(request: HistoricalNavEvaluationRequest) -> HistoricalNavEvaluationResponse:
    """一次只读一致性快照读完显式批次；离开数据库事务后进行纯内存计算。"""
    deadline = perf_counter() + 15
    batches = []
    with Session(get_nav_sample_storage_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        try:
            for batch, rows in iter_batch_rows(session, request.batch_ids):
                _check_deadline(deadline)
                batches.append(restore_stored_batch(batch, rows))
        except LookupError as error:
            raise HistoricalNavEvaluationError(
                "BATCH_NOT_FOUND", "一个或多个所选批次不存在；未返回部分评估。", 404
            ) from error
    result = prepare_historical_nav_dataset(request, batches, deadline=deadline)
    return result.report


def prepare_historical_nav_dataset(
    request: HistoricalNavEvaluationRequest,
    batches: Iterable[HistoricalNavStoredBatch],
    *,
    deadline: float | None = None,
) -> PreparedDataset:
    """纯计算入口：验证、跨批去重、时间隔离、生成X/y及验证成绩，不访问数据库。

    相同基金/净值日只有完整内容相同时才能合并；不根据标签好坏、最新水位或完成状态挑选。
    日期按available_at划段，再要求label_available_at不超过所属段截止，清除跨界答案。
    """
    deadline = perf_counter() + 15 if deadline is None else deadline
    ordered = sorted(batches, key=lambda b: b.batch_id)
    if len(ordered) != len(request.batch_ids) or {b.batch_id for b in ordered} != set(request.batch_ids):
        raise HistoricalNavEvaluationError("BATCH_SELECTION_MISMATCH", "实际批次与请求清单不一致，不能评估。")
    protocol = EvaluationProtocol(
        **request.model_dump(include={"train_start_date", "train_end_date", "validation_end_date", "test_end_date"})
    )
    versions = {
        "feature_version": HISTORICAL_NAV_FEATURE_VERSION,
        "sample_rule_version": HISTORICAL_NAV_SAMPLE_RULE_VERSION,
        "label_version": HISTORICAL_NAV_LABEL_VERSION,
    }
    source_code = ordered[0].source_code
    if len({batch.source_code for batch in ordered}) != 1:
        raise HistoricalNavEvaluationError("INCOMPATIBLE_BATCHES", "所选批次来源不一致，请选择同一来源。")
    unique = {}
    input_count = 0
    for batch in ordered:
        _check_deadline(deadline)
        if batch.source_code != source_code or any(getattr(batch, key) != value for key, value in versions.items()):
            raise HistoricalNavEvaluationError(
                "INCOMPATIBLE_BATCHES", "所选批次来源或规则版本不一致；请使用同一套来源与规则。"
            )
        if batch.source_sync_run_id is None:
            raise HistoricalNavEvaluationError("SOURCE_WATERMARK_MISSING", "所选批次缺少来源水位，不能准备候选数据。")
        try:
            validate_batch_samples(batch, batch.items)
        except (ValueError, ArithmeticError, TypeError, KeyError) as error:
            raise HistoricalNavStorageError(
                "STORED_BATCH_INCONSISTENT", "所选批次完整性校验失败，不自动重算或覆盖。", 503
            ) from error
        input_count += len(batch.items)
        for sample in batch.items:
            key = sample.fund_code, sample.as_of_date
            content_hash = _hash_parts([asdict(sample)])
            if key in unique:
                if unique[key][2] != content_hash:
                    raise HistoricalNavEvaluationError(
                        "CONFLICTING_SAMPLES", "同基金同一天在所选批次中内容不一致；请明确选择版本，不自动取最新。"
                    )
            else:
                unique[key] = (batch.batch_id, sample, content_hash)

    split_rows = {split: [] for split in SPLITS}
    excluded = defaultdict(Counter)
    fund_codes = sorted({b.fund_code for b in ordered})
    unique_counts = Counter(key[0] for key in unique)
    for batch_id, sample, content_hash in unique.values():
        split, reason = _sample_split(sample, protocol)
        if reason:
            excluded[sample.fund_code][reason] += 1
            continue
        metrics = sample.feature_payload["metrics"]
        label = sample.offline_label
        split_rows[split].append(
            PreparedRow(
                batch_id=batch_id,
                fund_code=sample.fund_code,
                as_of_date=sample.as_of_date,
                available_at=sample.available_at,
                label_available_at=label.label_available_at,
                x=tuple(Decimal(metrics[name]) for name in FEATURE_NAMES),
                y=label.label_up_20d,
                content_hash=content_hash,
            )
        )
    for rows in split_rows.values():
        rows.sort(key=lambda row: (row.available_at, row.fund_code, row.as_of_date))
    grouped = {split: _group_funds(rows) for split, rows in split_rows.items()}
    funds = []
    for fund_code in fund_codes:
        train = grouped["TRAIN"].get(fund_code, ())
        counts = {split: len(grouped[split].get(fund_code, ())) for split in SPLITS}
        funds.append(
            FundPreparationSummary(
                fund_code=fund_code,
                unique_sample_count=unique_counts[fund_code],
                split_counts=counts,
                missing_samples={s: max(0, protocol.minimum_samples_per_fund[s] - counts[s]) for s in SPLITS},
                excluded_reasons=dict(sorted(excluded[fund_code].items())),
                train_up_rate=Decimal(sum(r.y for r in train)) / len(train) if train else None,
                warnings=("TRAIN_SINGLE_CLASS",) if train and len({r.y for r in train}) == 1 else (),
            )
        )
    sufficient = all(not any(f.missing_samples.values()) for f in funds)
    _check_deadline(deadline)
    comparisons = _evaluate_baselines(grouped, funds, deadline) if sufficient else ()
    total_excluded = Counter()
    for reasons in excluded.values():
        total_excluded.update(reasons)
    preview = [
        PreparedSamplePreview(
            split=split,
            **{
                name: getattr(row, name)
                for name in ("batch_id", "fund_code", "as_of_date", "available_at", "label_available_at", "x", "y")
            },
        )
        for split in ("TRAIN", "VALIDATION")
        for row in split_rows[split][: request.preview_size]
    ][: request.preview_size]
    # 包含所有输入、标签、来源水位及规则，避免只有特征哈希而遗漏答案修订。
    # 不包含previewSize，也不依赖请求中的batchIds顺序。
    dataset_hash = _hash_parts(_dataset_hash_parts(protocol, ordered))
    train_hash = _hash_parts(
        {"fund_code": r.fund_code, "as_of_date": r.as_of_date, "content_hash": r.content_hash}
        for r in split_rows["TRAIN"]
    )
    report = HistoricalNavEvaluationResponse(
        status="BASELINE_EVALUATED" if sufficient else "INSUFFICIENT_DATA",
        protocol=protocol,
        batch_ids=tuple(b.batch_id for b in ordered),
        source_code=source_code,
        source_sync_run_ids=tuple(sorted({b.source_sync_run_id for b in ordered})),
        versions=versions,
        feature_names=FEATURE_NAMES,
        dataset_hash=dataset_hash,
        train_hash=train_hash,
        input_sample_count=input_count,
        duplicate_sample_count=input_count - len(unique),
        unique_sample_count=len(unique),
        included_sample_count=sum(map(len, split_rows.values())),
        excluded_reasons=dict(sorted(total_excluded.items())),
        funds=tuple(funds),
        baselines=comparisons,
        sample_preview=tuple(preview),
        limitations=LIMITATIONS,
    )
    _check_deadline(deadline)
    return PreparedDataset(
        tuple(split_rows["TRAIN"]), tuple(split_rows["VALIDATION"]), tuple(split_rows["TEST"]), report
    )


def _dataset_hash_parts(protocol: EvaluationProtocol, batches: list[HistoricalNavStoredBatch]) -> Iterable[object]:
    yield {"protocol": protocol.model_dump(mode="json"), "feature_names": FEATURE_NAMES}
    for batch in batches:
        yield batch.model_dump(mode="json")


def _sample_split(sample: HistoricalNavSample, protocol: EvaluationProtocol) -> tuple[str | None, str | None]:
    """按固定顺序给每个唯一样本一个去向；每项剔除只计数一次。"""
    if sample.eligibility_status != "SCORABLE":
        return None, f"{sample.eligibility_status}:{sample.unavailable_reason}"
    if sample.nav_value_basis != protocol.nav_value_basis:
        return None, "UNSUPPORTED_NAV_BASIS"
    if sample.available_at is None or sample.offline_label is None:
        raise ValueError("SCORABLE sample must have feature cutoff and answer")
    cutoff = sample.available_at
    if cutoff < protocol.train_start_date or cutoff > protocol.test_end_date:
        return None, "OUTSIDE_TIME_RANGE"
    if cutoff <= protocol.train_end_date:
        split, end = "TRAIN", protocol.train_end_date
    elif cutoff <= protocol.validation_end_date:
        split, end = "VALIDATION", protocol.validation_end_date
    else:
        split, end = "TEST", protocol.test_end_date
    if sample.offline_label.label_available_at > end:
        return None, f"{split}_LABEL_AFTER_CUTOFF"
    return split, None


def _group_funds(rows: Iterable[PreparedRow]) -> dict[str, list[PreparedRow]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.fund_code].append(row)
    return dict(grouped)


def calculate_baseline_metrics(labels: tuple[int, ...], scores: tuple[Decimal, ...]) -> BaselineMetrics:
    """二分类准确率、平衡准确率、Brier平方误差；0.5归为非上涨，不在验证段调阈值。"""
    if not labels or len(labels) != len(scores) or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("require paired non-empty binary labels and scores")
    if any(not score.is_finite() or not 0 <= score <= 1 for score in scores):
        raise ValueError("scores must be finite in [0, 1]")
    predictions = tuple(int(score > Decimal("0.5")) for score in scores)
    counts = Counter(labels)
    correct = Counter(y for y, predicted in zip(labels, predictions, strict=True) if y == predicted)
    balanced = ((Decimal(correct[0]) / counts[0] + Decimal(correct[1]) / counts[1]) / 2) if len(counts) == 2 else None
    return BaselineMetrics(
        sample_count=len(labels),
        correct_count=sum(correct.values()),
        actual_up_count=counts[1],
        predicted_up_count=sum(predictions),
        accuracy=_rounded(Decimal(sum(correct.values())) / len(labels)),
        balanced_accuracy=_rounded(balanced) if balanced is not None else None,
        brier_score=_rounded(sum((score - y) ** 2 for score, y in zip(scores, labels, strict=True)) / len(labels)),
    )


def _rounded(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)


def _evaluate_baselines(
    grouped: dict[str, dict[str, list[PreparedRow]]], funds: list[FundPreparationSummary], deadline: float
) -> tuple[BaselineComparison, ...]:
    """只从TRAIN计算频率，只在VALIDATION算分；既不引用TEST标签，也不选择最佳方法。"""
    descriptions = {
        "ALWAYS_UP": "永远预测上涨，分数固定1。",
        "TRAIN_UP_FREQUENCY": "每只基金固定使用其训练段上涨数/训练数；验证期间不更新。",
        "MOMENTUM_20D": "最近20日收益大于0给1，否则给0；仅使用历史输入。",
        "FIXED_MOMENTUM_SCORE": (
            "复用旧公式clip(0.5+2×20日收益,0.05,0.95)，四位小数；本次统一二分类，不复用旧三态阈值。"
        ),
    }
    collected = {name: [] for name in descriptions}
    overall_labels = []
    overall_scores = {name: [] for name in descriptions}
    for fund in funds:
        _check_deadline(deadline)
        rows = grouped["VALIDATION"][fund.fund_code]
        labels = tuple(row.y for row in rows)
        returns = tuple(row.x[FEATURE_NAMES.index("return_20d")] for row in rows)
        predictions = {
            "ALWAYS_UP": (Decimal(1),) * len(rows),
            "TRAIN_UP_FREQUENCY": (fund.train_up_rate,) * len(rows),
            "MOMENTUM_20D": tuple(Decimal(int(r > 0)) for r in returns),
            "FIXED_MOMENTUM_SCORE": tuple(fixed_momentum_score(r) for r in returns),
        }
        overall_labels.extend(labels)
        for name, scores in predictions.items():
            collected[name].append(
                BaselineFundMetrics(fund_code=fund.fund_code, metrics=calculate_baseline_metrics(labels, scores))
            )
            overall_scores[name].extend(scores)
    return tuple(
        BaselineComparison(
            baseline_id=name,
            description=description,
            validation=calculate_baseline_metrics(tuple(overall_labels), tuple(overall_scores[name])),
            per_fund=tuple(collected[name]),
        )
        for name, description in descriptions.items()
    )
