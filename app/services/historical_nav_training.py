"""学习用途逻辑回归：只拟合TRAIN，VALIDATION只打分，TEST绝不交给训练器。"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from importlib.metadata import version
from itertools import islice
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

from app.schemas.historical_nav_evaluation import BaselineComparison, BaselineFundMetrics
from app.schemas.historical_nav_training import (
    CandidateBaselineDelta,
    CandidatePredictionPreview,
    CandidateProtocol,
    HistoricalNavTrainingRequest,
    HistoricalNavTrainingResponse,
    LogisticModelArtifact,
)
from app.services.historical_nav_evaluation import (
    FEATURE_NAMES,
    PreparedDataset,
    PreparedRow,
    _hash_parts,
    calculate_baseline_metrics,
    load_historical_nav_dataset,
)

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

# 同一应用进程只做一个训练；多进程部署仍需各自计算上限，不能冒充全局锁。
_TRAINING_LOCK = Lock()
MAX_ROWS = 512 * 31
TRAINING_SECONDS = 30
PROTOCOL = CandidateProtocol()
NUMERICAL_PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "narwhals", "threadpoolctl", "cloudpickle")
LIMITATIONS = (
    "候选已拟合不等于正式训练数据准入或模型发布；仍为LEARNING_ONLY。",
    "输出为未经校准的上涨分数，不是投资建议或已验证的未来上涨概率。",
    "只做一次固定时间验证，不搜索参数、不做滚动回测、不展示保留测试成绩。",
    "共用模型按基金均衡训练，但验证总体按样本汇总；请同时看逐基金成绩。",
    "保留原报告的首次可得、历史修订、交易日历及重叠标签限制。",
    "同环境同输入可复现；不同CPU/操作系统或依赖版本可能产生微小浮点差异。",
)


class HistoricalNavTrainingError(RuntimeError):
    """受控错误，不携带原始数据、连接信息或Token。"""

    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.code, self.status_code = code, status_code


def artifact_hash(model: LogisticModelArtifact) -> str:
    """JSON键排序、禁止NaN；模型哈希不含报告/验证日期/完整数据指纹。"""
    encoded = json.dumps(
        model.model_dump(mode="json", exclude={"model_hash"}),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def restore_logistic_artifact(payload: str | bytes) -> LogisticModelArtifact:
    """只解析有界JSON和数值；不反序列化pickle，不执行产物中的代码。"""
    if len(payload.encode("utf-8") if isinstance(payload, str) else payload) > 65536:
        raise ValueError("artifact exceeds 64KiB")
    model = LogisticModelArtifact.model_validate_json(payload)
    if model.protocol != PROTOCOL or model.feature_names != FEATURE_NAMES or model.model_hash != artifact_hash(model):
        raise ValueError("artifact protocol, feature order or hash mismatch")
    return model


def predict_artifact_scores(
    model: LogisticModelArtifact, rows: tuple[tuple[Decimal | float | int, ...], ...]
) -> tuple[float, ...]:
    """离线数值推理，仅接收七列X，不接收标签/基金元数据；不挂到真实预测API。"""
    return tuple(sigmoid(z) for z in predict_artifact_logits(model, rows))


def sigmoid(value: float) -> float:
    """数值稳定地将线性得分映射到0–1；不是人为夹成0.05–0.95。"""
    if not math.isfinite(value):
        raise ValueError("non-finite linear score")
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def predict_artifact_logits(
    model: LogisticModelArtifact, rows: tuple[tuple[Decimal | float | int, ...], ...]
) -> tuple[float, ...]:
    """导出模型的原始线性得分；校准以它为输入，不对已经饱和的概率取logit。"""
    if len(rows) > MAX_ROWS:
        raise ValueError("too many prediction rows")
    # 对象也须经过JSON回读，避免model_copy绕过Pydantic验证或列顺序悄悄变化。
    model = restore_logistic_artifact(model.model_dump_json())
    scores = []
    for row in rows:
        if len(row) != 7 or any(isinstance(x, bool) or not math.isfinite(float(x)) for x in row):
            raise ValueError("require seven finite numeric features")
        z = model.intercept + sum(
            ((float(x) - mean) / scale) * weight
            for x, mean, scale, weight in zip(row, model.mean, model.scale, model.coefficients, strict=True)
        )
        if not math.isfinite(z):
            raise ValueError("non-finite linear score")
        scores.append(z)
    return tuple(scores)


@contextmanager
def training_slot() -> Iterator[None]:
    """候选训练和校准共用一个进程内计算槽；失败也释放，不代表跨进程分布式锁。"""
    if not _TRAINING_LOCK.acquire(blocking=False):
        raise HistoricalNavTrainingError("TRAINING_BUSY", "当前进程正在训练，请稍后重试。", 429)
    try:
        yield
    finally:
        _TRAINING_LOCK.release()


def train_stored_historical_nav_candidate(request: HistoricalNavTrainingRequest) -> HistoricalNavTrainingResponse:
    """HTTP与CLI共用：先只读快照，释放连接后训练；忙时立即拒绝，不排无限队列。"""
    with training_slot():
        prepared = load_historical_nav_dataset(request.evaluation_request())
        return train_prepared_candidate(request, prepared)


def _check_deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise HistoricalNavTrainingError("TRAINING_TIMEOUT", "候选训练超出阶段预算，本次不交付模型。", 503)


def _fit_training_rows(
    rows: tuple[PreparedRow, ...],
) -> tuple[StandardScaler, LogisticRegression, dict[str, int], dict[str, float]]:
    """训练器只接收TRAIN行；不可能从此参数取得验证/测试标签。依赖延迟导入不拖累旧只读入口。"""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    if not rows or len(rows) > MAX_ROWS:
        raise HistoricalNavTrainingError("INVALID_TRAINING_MATRIX", "训练矩阵为空或超出上限。")
    x = np.asarray([row.x for row in rows], dtype=np.float64)
    y = np.asarray([row.y for row in rows], dtype=np.int64)
    if (
        x.shape != (len(rows), 7)
        or not np.isfinite(x).all()
        or any(type(r.y) is not int or r.y not in (0, 1) for r in rows)
        or any(isinstance(value, bool) for row in rows for value in row.x)
    ):
        raise HistoricalNavTrainingError("INVALID_TRAINING_MATRIX", "训练输入须为七列有限数值和独立0/1答案。")
    if set(y.tolist()) != {0, 1}:
        raise HistoricalNavTrainingError("TRAIN_SINGLE_CLASS", "训练段必须同时有上涨和非上涨样本，不制造另一类。")
    counts = Counter(row.fund_code for row in rows)
    fund_weights = {fund: len(rows) / (len(counts) * n) for fund, n in sorted(counts.items())}
    weights = np.asarray([fund_weights[row.fund_code] for row in rows], dtype=np.float64)
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            scaler = StandardScaler().fit(x, sample_weight=weights)
            estimator = LogisticRegression(
                C=PROTOCOL.c,
                l1_ratio=0.0,
                solver=PROTOCOL.solver,
                tol=PROTOCOL.tolerance,
                max_iter=PROTOCOL.max_iterations,
                fit_intercept=True,
                class_weight=None,
                random_state=PROTOCOL.random_seed,
            ).fit(scaler.transform(x), y, sample_weight=weights)
    except ConvergenceWarning as error:
        raise HistoricalNavTrainingError("TRAINING_NOT_CONVERGED", "本次求解未收敛，不交付半成品模型。", 503) from error
    if not np.array_equal(estimator.classes_, [0, 1]) or int(estimator.n_iter_[0]) >= PROTOCOL.max_iterations:
        raise HistoricalNavTrainingError("TRAINING_NOT_CONVERGED", "类别或求解状态异常，不交付模型。", 503)
    return scaler, estimator, dict(sorted(counts.items())), fund_weights


def training_rows_hash(rows: tuple[PreparedRow, ...]) -> str:
    """沿用原训练内容指纹格式，不混入考试日期、预览大小或批次排列。"""
    return _hash_parts(
        {"fund_code": r.fund_code, "as_of_date": r.as_of_date, "content_hash": r.content_hash} for r in rows
    )


def fit_logistic_artifact(
    rows: tuple[PreparedRow, ...], *, start_date: date, end_date: date, versions: dict[str, str]
) -> LogisticModelArtifact:
    """纯拟合与导出；调用者负责确定时间段，函数也检查本段输入/标签的可得截止。"""
    if any(not start_date <= r.available_at <= end_date or r.label_available_at > end_date for r in rows):
        raise HistoricalNavTrainingError("FIT_TIME_BOUNDARY", "拟合样本越过本窗口的可得日期边界。")
    scaler, estimator, counts, weights = _fit_training_rows(rows)
    model = LogisticModelArtifact(
        protocol=PROTOCOL,
        feature_names=FEATURE_NAMES,
        mean=tuple(scaler.mean_),
        scale=tuple(scaler.scale_),
        coefficients=tuple(estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
        iterations=int(estimator.n_iter_[0]),
        train_hash=training_rows_hash(rows),
        train_start_date=start_date,
        train_end_date=end_date,
        train_count=len(rows),
        train_counts_per_fund=counts,
        sample_weight_per_fund=weights,
        train_class_counts={str(y): sum(row.y == y for row in rows) for y in (0, 1)},
        versions=versions,
        runtime_versions={"python": platform.python_version(), **{p: version(p) for p in NUMERICAL_PACKAGES}},
        model_hash="0" * 64,
    )
    model = model.model_copy(update={"model_hash": artifact_hash(model)})
    # 拟合完成就核对数值JSON，既不需要读取验证答案，也不把库对象序列化保存。
    replay_x = tuple(row.x for row in rows)
    raw_scores = predict_artifact_scores(model, replay_x)
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        library_scores = estimator.predict_proba(scaler.transform(replay_x))[:, 1]
    if any(abs(a - float(b)) > 1e-12 for a, b in zip(raw_scores, library_scores, strict=True)):
        raise HistoricalNavTrainingError("ARTIFACT_REPLAY_MISMATCH", "保存模型的重算结果不一致，不交付模型。", 503)
    return model


def train_prepared_candidate(
    request: HistoricalNavTrainingRequest, prepared: PreparedDataset
) -> HistoricalNavTrainingResponse:
    """有界纯计算；数据不足保留诊断，足够时只拟合一次并在相同验证段比较。"""
    deadline = perf_counter() + TRAINING_SECONDS
    report = prepared.report
    if report.dataset_hash != request.expected_dataset_hash:
        raise HistoricalNavTrainingError("DATASET_HASH_MISMATCH", "数据指纹与已确认报告不同，请先重新核对基线报告。")
    response = HistoricalNavTrainingResponse(
        status="INSUFFICIENT_DATA",
        protocol=PROTOCOL,
        preparation=report,
        limitations=LIMITATIONS,
    )
    if report.status == "INSUFFICIENT_DATA":
        return response
    if report.feature_names != FEATURE_NAMES or len(prepared.validation) > MAX_ROWS:
        raise HistoricalNavTrainingError("INVALID_TRAINING_MATRIX", "候选列顺序或验证数量不符合固定协议。")
    _check_deadline(deadline)
    model = fit_logistic_artifact(
        prepared.train,
        start_date=report.protocol.train_start_date,
        end_date=report.protocol.train_end_date,
        versions=report.versions,
    )
    if model.train_hash != report.train_hash:
        raise HistoricalNavTrainingError("TRAIN_HASH_MISMATCH", "拟合内容与准备报告不一致，不交付模型。")
    _check_deadline(deadline)
    raw_scores = predict_artifact_scores(model, tuple(row.x for row in prepared.validation))
    scores = tuple(Decimal(str(s)) for s in raw_scores)
    labels = tuple(row.y for row in prepared.validation)
    candidate = BaselineComparison(
        baseline_id=PROTOCOL.version,
        description="只用训练段拟合的基金均衡L2逻辑回归，未校准上涨分数。",
        validation=calculate_baseline_metrics(labels, scores),
        per_fund=tuple(
            BaselineFundMetrics(
                fund_code=fund,
                metrics=calculate_baseline_metrics(
                    tuple(row.y for row in prepared.validation if row.fund_code == fund),
                    tuple(
                        score for row, score in zip(prepared.validation, scores, strict=True) if row.fund_code == fund
                    ),
                ),
            )
            for fund in model.train_counts_per_fund
        ),
    )
    deltas = tuple(
        CandidateBaselineDelta(
            baseline_id=b.baseline_id,
            accuracy_delta=candidate.validation.accuracy - b.validation.accuracy,
            balanced_accuracy_delta=(candidate.validation.balanced_accuracy - b.validation.balanced_accuracy)
            if candidate.validation.balanced_accuracy is not None and b.validation.balanced_accuracy is not None
            else None,
            brier_delta=candidate.validation.brier_score - b.validation.brier_score,
        )
        for b in report.baselines
    )
    preview = tuple(
        CandidatePredictionPreview(
            fund_code=row.fund_code,
            as_of_date=row.as_of_date,
            up_score=score,
            predicted_up=int(score > Decimal("0.5")),
            actual_up=row.y,
        )
        for row, score in islice(zip(prepared.validation, scores, strict=True), request.preview_size)
    )
    _check_deadline(deadline)
    return response.model_copy(
        update={
            "status": "CANDIDATE_EVALUATED",
            "model": model,
            "candidate": candidate,
            "baseline_deltas": deltas,
            "prediction_preview": preview,
        }
    )
