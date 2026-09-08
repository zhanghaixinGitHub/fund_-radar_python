"""原TRAIN内部扩展窗口校准、DEV考试和独立2024验收；没有2025评分路径。"""

import hashlib
import json
import warnings
from collections import Counter
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from time import perf_counter

from app.schemas.historical_nav_calibration import (
    CalibratedModelArtifact,
    CalibrationPreview,
    CalibrationProtocol,
    CalibrationWindow,
    CalibrationWindowReport,
    FundReliability,
    HistoricalNavCalibrationRequest,
    HistoricalNavCalibrationResponse,
    ReliabilityBin,
    ReliabilityReport,
    SigmoidCalibrator,
    WindowFundCounts,
)
from app.schemas.historical_nav_evaluation import BaselineComparison, BaselineFundMetrics
from app.schemas.historical_nav_training import LogisticModelArtifact
from app.services.historical_nav_evaluation import (
    PreparedDataset,
    PreparedRow,
    calculate_baseline_metrics,
    evaluate_window_baselines,
    load_historical_nav_dataset,
)
from app.services.historical_nav_training import (
    MAX_ROWS,
    HistoricalNavTrainingError,
    fit_logistic_artifact,
    predict_artifact_logits,
    restore_logistic_artifact,
    sigmoid,
    training_rows_hash,
    training_slot,
)

START = date(2022, 1, 1)
WINDOWS = (
    CalibrationWindow(
        window_id="DEV_2023_Q3",
        role="TRAIN_INTERNAL_ROLLING",
        fit_end_date=date(2023, 2, 28),
        calibration_end_date=date(2023, 6, 30),
        evaluation_end_date=date(2023, 9, 30),
        minimum_exam_per_fund=40,
    ),
    CalibrationWindow(
        window_id="DEV_2023_Q4",
        role="TRAIN_INTERNAL_ROLLING",
        fit_end_date=date(2023, 5, 31),
        calibration_end_date=date(2023, 9, 30),
        evaluation_end_date=date(2023, 12, 31),
        minimum_exam_per_fund=40,
    ),
    CalibrationWindow(
        window_id="VALIDATION_2024",
        role="FIXED_VALIDATION",
        fit_end_date=date(2023, 6, 30),
        calibration_end_date=date(2023, 12, 31),
        evaluation_end_date=date(2024, 12, 31),
        minimum_exam_per_fund=120,
    ),
)
PROTOCOL = CalibrationProtocol(windows=WINDOWS)
COMPUTE_SECONDS = 60
LIMITATIONS = (
    "只在原TRAIN内部拟合基础模型与校准器，2024只验收，2025不评分；测试数据仍非数据库权限层封存。",
    "DEV是训练期内部两轮扩展窗口，不是覆盖各类市场行情的完整独立回测。",
    "每窗基础模型使用较短FIT段，不能将它与上一轮全TRAIN模型的差异都归功于校准。",
    "2024固定验证成绩此前已被观察，不是全新独立最终测试；不得据此反复换方案择优发布。",
    "Brier并非纯校准指标；ECE和分箱受样本量、边界和标签重叠影响，不单独证明概率可信。",
    "保留首次可得、历史修订、交易日历及分红复权限制，研究校准完成不解除正式准入或发布限制。",
)


def _deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise HistoricalNavTrainingError("CALIBRATION_TIMEOUT", "校准验证超出阶段预算，不返回部分成功报告。", 503)


def calibrated_model_hash(model: CalibratedModelArtifact) -> str:
    content = json.dumps(
        model.model_dump(mode="json", exclude={"model_hash"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def restore_calibrated_artifact(payload: bytes | str) -> CalibratedModelArtifact:
    """只回读数值JSON；配套基础模型不能更换，哈希只是完整性校验而非签名。"""
    if len(payload.encode("utf-8") if isinstance(payload, str) else payload) > 131072:
        raise ValueError("calibrated artifact exceeds 128KiB")
    model = CalibratedModelArtifact.model_validate_json(payload)
    restore_logistic_artifact(model.base_model.model_dump_json())
    if (
        model.model_hash != calibrated_model_hash(model)
        or model.calibrator.base_model_hash != model.base_model.model_hash
        or model.calibrator.start_date <= model.base_model.train_end_date
    ):
        raise ValueError("calibrated model linkage/hash/time mismatch")
    return model


def predict_calibrated_scores(
    model: CalibratedModelArtifact, x: tuple[tuple[Decimal | float | int, ...], ...]
) -> tuple[float, ...]:
    """推理只接受历史七列X；不接受y，基础模型与校准器始终绑定。"""
    model = restore_calibrated_artifact(model.model_dump_json())
    z = predict_artifact_logits(model.base_model, x)
    return tuple(sigmoid(model.calibrator.slope * value + model.calibrator.intercept) for value in z)


def fit_calibrator(
    base: LogisticModelArtifact, rows: tuple[PreparedRow, ...], *, end_date: date
) -> CalibratedModelArtifact:
    """输入只有独立CAL段，基础模型不再fit；固定一维L2校准，不搜索方法或强度。"""
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    if (
        not rows
        or len(rows) > MAX_ROWS
        or any(not base.train_end_date < r.available_at <= end_date or r.label_available_at > end_date for r in rows)
    ):
        raise HistoricalNavTrainingError("CALIBRATION_TIME_BOUNDARY", "校准样本为空、超量或越过独立时间边界。")
    if any(type(r.y) is not int or r.y not in (0, 1) for r in rows) or {r.y for r in rows} != {0, 1}:
        raise HistoricalNavTrainingError("CALIBRATION_SINGLE_CLASS", "校准段必须同时有上涨和非上涨答案。")
    counts = dict(sorted(Counter(r.fund_code for r in rows).items()))
    weights = {f: len(rows) / (len(counts) * n) for f, n in counts.items()}
    logits = np.asarray(predict_artifact_logits(base, tuple(r.x for r in rows)), dtype=np.float64).reshape(-1, 1)
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            estimator = LogisticRegression(
                C=1.0,
                l1_ratio=0.0,
                solver="lbfgs",
                tol=1e-8,
                max_iter=1000,
                fit_intercept=True,
                class_weight=None,
                random_state=0,
            ).fit(logits, [r.y for r in rows], sample_weight=[weights[r.fund_code] for r in rows])
            reference = estimator.predict_proba(logits)[:, 1]
    except ConvergenceWarning as error:
        raise HistoricalNavTrainingError("CALIBRATION_NOT_CONVERGED", "校准器没有收敛，本窗不交付模型。") from error
    if int(estimator.n_iter_[0]) >= 1000 or estimator.classes_.tolist() != [0, 1]:
        raise HistoricalNavTrainingError("CALIBRATION_NOT_CONVERGED", "校准器迭代或类别异常。")
    calibrator = SigmoidCalibrator(
        slope=float(estimator.coef_[0, 0]),
        intercept=float(estimator.intercept_[0]),
        iterations=int(estimator.n_iter_[0]),
        start_date=base.train_end_date + timedelta(days=1),
        end_date=end_date,
        sample_count=len(rows),
        counts_per_fund=counts,
        weights_per_fund=weights,
        class_counts={str(y): sum(r.y == y for r in rows) for y in (0, 1)},
        calibration_hash=training_rows_hash(rows),
        base_model_hash=base.model_hash,
    )
    model = CalibratedModelArtifact(base_model=base, calibrator=calibrator, model_hash="0" * 64)
    model = model.model_copy(update={"model_hash": calibrated_model_hash(model)})
    replay = predict_calibrated_scores(model, tuple(r.x for r in rows))
    if any(abs(a - float(b)) > 1e-12 for a, b in zip(replay, reference, strict=True)):
        raise HistoricalNavTrainingError("CALIBRATION_REPLAY_MISMATCH", "校准JSON重算不一致，不交付产物。", 503)
    return model


def reliability_report(labels: tuple[int, ...], scores: tuple[Decimal, ...]) -> ReliabilityReport:
    """固定五档，不按答案自适应分箱；空档保持null，不把缺数据当0%上涨。"""
    calculate_baseline_metrics(labels, scores)  # 复用成对、有限、0/1及[0,1]校验。
    bins, ece = [], Decimal(0)
    for index in range(5):
        lower, upper = Decimal(index) / 5, Decimal(index + 1) / 5
        pairs = [(y, s) for y, s in zip(labels, scores, strict=True) if lower <= s < upper or index == 4 and s == 1]
        count = len(pairs)
        mean = sum(s for _, s in pairs) / count if count else None
        rate = Decimal(sum(y for y, _ in pairs)) / count if count else None
        gap = abs(mean - rate) if count else None
        if count:
            ece += gap * count / len(labels)
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_score=mean,
                observed_up_rate=rate,
                absolute_gap=gap,
                enough_samples=count >= PROTOCOL.minimum_bin_count,
            )
        )
    return ReliabilityReport(
        sample_count=len(labels), ece=ece.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP), bins=tuple(bins)
    )


def _window_rows(
    data: PreparedDataset, window: CalibrationWindow
) -> tuple[dict[str, tuple[PreparedRow, ...]], tuple[WindowFundCounts, ...]]:
    """只从原TRAIN拆学习/校准，原VALIDATION仅作固定考试；从不引用data.test。"""
    rows = {}
    periods = (
        ("FIT", data.train, START - timedelta(days=1), window.fit_end_date),
        ("CALIBRATION", data.train, window.fit_end_date, window.calibration_end_date),
        (
            "EXAM",
            data.train if window.role == "TRAIN_INTERNAL_ROLLING" else data.validation,
            window.calibration_end_date,
            window.evaluation_end_date,
        ),
    )
    purged = {stage: Counter() for stage, *_ in periods}
    for stage, source, lower, upper in periods:
        selected = tuple(r for r in source if lower < r.available_at <= upper)
        rows[stage] = tuple(r for r in selected if r.label_available_at <= upper)
        purged[stage].update(r.fund_code for r in selected if r.label_available_at > upper)
    minimum = {
        "FIT": PROTOCOL.minimum_fit_per_fund,
        "CALIBRATION": PROTOCOL.minimum_calibration_per_fund,
        "EXAM": window.minimum_exam_per_fund,
    }
    summaries = []
    for fund in data.report.funds:
        counts = {stage: sum(r.fund_code == fund.fund_code for r in subset) for stage, subset in rows.items()}
        summaries.append(
            WindowFundCounts(
                fund_code=fund.fund_code,
                counts=counts,
                missing={s: max(0, minimum[s] - n) for s, n in counts.items()},
                purged={s: purged[s][fund.fund_code] for s in rows},
            )
        )
    return rows, tuple(summaries)


def _comparison(name: str, rows: tuple[PreparedRow, ...], scores: tuple[Decimal, ...]) -> BaselineComparison:
    return BaselineComparison(
        baseline_id=name,
        description="同一冻结基础模型的校准前/后考试成绩，不是2025测试。",
        validation=calculate_baseline_metrics(tuple(r.y for r in rows), scores),
        per_fund=tuple(
            BaselineFundMetrics(
                fund_code=f,
                metrics=calculate_baseline_metrics(
                    tuple(r.y for r in rows if r.fund_code == f),
                    tuple(s for r, s in zip(rows, scores, strict=True) if r.fund_code == f),
                ),
            )
            for f in sorted({r.fund_code for r in rows})
        ),
    )


def _evaluate_window(
    request: HistoricalNavCalibrationRequest,
    data: PreparedDataset,
    window: CalibrationWindow,
    *,
    deadline: float,
) -> CalibrationWindowReport:
    _deadline(deadline)
    rows, counts = _window_rows(data, window)
    result = CalibrationWindowReport(window=window, status="INSUFFICIENT_DATA", funds=counts)
    if data.report.status == "INSUFFICIENT_DATA":
        return result.model_copy(update={"reason": "GLOBAL_DATA_INSUFFICIENT"})
    if any(any(f.missing.values()) for f in counts):
        return result.model_copy(update={"reason": "WINDOW_DATA_INSUFFICIENT"})
    try:
        base = fit_logistic_artifact(
            rows["FIT"], start_date=START, end_date=window.fit_end_date, versions=data.report.versions
        )
        _deadline(deadline)
        model = fit_calibrator(base, rows["CALIBRATION"], end_date=window.calibration_end_date)
    except HistoricalNavTrainingError as error:
        if error.code not in {
            "TRAIN_SINGLE_CLASS",
            "CALIBRATION_SINGLE_CLASS",
            "TRAINING_NOT_CONVERGED",
            "CALIBRATION_NOT_CONVERGED",
        }:
            raise
        return result.model_copy(update={"status": "REJECTED", "reason": error.code})
    _deadline(deadline)
    exam = rows["EXAM"]
    before_scores = tuple(Decimal(str(sigmoid(z))) for z in predict_artifact_logits(base, tuple(r.x for r in exam)))
    after_scores = tuple(Decimal(str(s)) for s in predict_calibrated_scores(model, tuple(r.x for r in exam)))
    before, after = (
        _comparison("UNCALIBRATED_SAME_BASE", exam, before_scores),
        _comparison("CALIBRATED_SAME_BASE", exam, after_scores),
    )
    labels = tuple(r.y for r in exam)
    before_reliability, after_reliability = (
        reliability_report(labels, before_scores),
        reliability_report(labels, after_scores),
    )
    per_fund = tuple(
        FundReliability(
            fund_code=f.fund_code,
            before=reliability_report(
                tuple(r.y for r in exam if r.fund_code == f.fund_code),
                tuple(s for r, s in zip(exam, before_scores, strict=True) if r.fund_code == f.fund_code),
            ),
            after=reliability_report(
                tuple(r.y for r in exam if r.fund_code == f.fund_code),
                tuple(s for r, s in zip(exam, after_scores, strict=True) if r.fund_code == f.fund_code),
            ),
        )
        for f in counts
    )
    history = tuple(
        r
        for r in data.train
        if r.available_at <= window.calibration_end_date and r.label_available_at <= window.calibration_end_date
    )
    warnings_ = ["OVERLAPPING_LABELS_NOT_INDEPENDENT"]
    if model.calibrator.slope <= 0:
        warnings_.append("NON_POSITIVE_CALIBRATION_SLOPE")
    if any(0 < b.count < 30 for p in (before_reliability, after_reliability) for b in p.bins):
        warnings_.append("SMALL_RELIABILITY_BINS")
    if window.role == "FIXED_VALIDATION":
        warnings_.append("PREVIOUSLY_OBSERVED_VALIDATION_NOT_FINAL_TEST")
    return result.model_copy(
        update={
            "status": "EVALUATED",
            "model": model,
            "before": before,
            "after": after,
            "baselines": evaluate_window_baselines(history, exam, deadline=deadline),
            "reliability_before": before_reliability,
            "reliability_after": after_reliability,
            "reliability_per_fund": per_fund,
            "warnings": tuple(warnings_),
            "brier_delta": after.validation.brier_score - before.validation.brier_score,
            "ece_delta": after_reliability.ece - before_reliability.ece,
            "prediction_preview": tuple(
                CalibrationPreview(
                    fund_code=r.fund_code, as_of_date=r.as_of_date, before_score=b, after_score=a, actual_up=r.y
                )
                for r, b, a in zip(
                    exam[: request.preview_size],
                    before_scores[: request.preview_size],
                    after_scores[: request.preview_size],
                    strict=True,
                )
            ),
        }
    )


def evaluate_prepared_calibration(
    request: HistoricalNavCalibrationRequest, data: PreparedDataset
) -> HistoricalNavCalibrationResponse:
    """固定三窗；不自动跳过不佳成绩，不聚合成一个跨时期总命中率。"""
    deadline = perf_counter() + COMPUTE_SECONDS
    if data.report.dataset_hash != request.expected_dataset_hash:
        raise HistoricalNavTrainingError("DATASET_HASH_MISMATCH", "数据指纹与已确认报告不同，请先核对。")
    windows = tuple(_evaluate_window(request, data, window, deadline=deadline) for window in WINDOWS)
    _deadline(deadline)
    evaluated = sum(w.status == "EVALUATED" for w in windows)
    status = (
        "CALIBRATION_EVALUATED"
        if evaluated == len(WINDOWS)
        else "PARTIAL_EVALUATION"
        if evaluated
        else "NO_VALID_WINDOWS"
    )
    if not evaluated and all(w.status == "INSUFFICIENT_DATA" for w in windows):
        status = "INSUFFICIENT_DATA"
    return HistoricalNavCalibrationResponse(
        status=status,
        protocol=PROTOCOL,
        preparation=data.report,
        windows=windows,
        evaluated_window_count=evaluated,
        brier_improved_window_count=sum(w.brier_delta is not None and w.brier_delta < 0 for w in windows),
        limitations=LIMITATIONS,
    )


def evaluate_stored_calibration(request: HistoricalNavCalibrationRequest) -> HistoricalNavCalibrationResponse:
    """与候选训练共享有界计算槽，只读快照退出后才拟合，不写数据库。"""
    with training_slot():
        return evaluate_prepared_calibration(request, load_historical_nav_dataset(request.evaluation_request()))
