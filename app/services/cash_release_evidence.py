"""只检查冻结报告中的数学与绑定关系，不重训、不读取原始价格或独立测试答案。"""

from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from app.schemas.historical_nav_calibration import CalibrationWindowReport, ReliabilityReport
from app.schemas.historical_nav_evaluation import BaselineComparison, BaselineMetrics
from app.services.cash_reinvestment_research import FUNDS

ROUND_UNIT = Decimal("0.00000001")  # 原成绩统一保留8位；汇总已舍入的逐基金成绩最多产生一个末位误差。
ARITHMETIC_TOLERANCE = Decimal("1e-20")  # 分档均值未舍入，仅容纳Decimal除法的末位误差。


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _metrics_check(metrics: BaselineMetrics) -> None:
    """沿用原生成前检查的基础成绩边界；共用到分档审查，不下调任何条件。"""
    if (
        metrics.sample_count <= 0
        or not 0 <= metrics.correct_count <= metrics.sample_count
        or not 0 <= metrics.actual_up_count <= metrics.sample_count
        or not 0 <= metrics.predicted_up_count <= metrics.sample_count
        or any(not value.is_finite() or not 0 <= value <= 1 for value in (metrics.accuracy, metrics.brier_score))
        or abs(metrics.accuracy - Decimal(metrics.correct_count) / metrics.sample_count) > ROUND_UNIT
    ):
        raise ValueError("invalid comparison metrics")


def validate_comparison(comparison: BaselineComparison) -> dict[str, BaselineMetrics]:
    """同一考试的总体数必须等于各基金之和，不能拿另一组样本的好成绩替换总体。"""
    groups = {f.fund_code: f.metrics for f in comparison.per_fund}
    _require(len(comparison.per_fund) == len(FUNDS) and set(groups) == set(FUNDS), "metric fund scope mismatch")
    groups = {"ALL": comparison.validation, **groups}
    for metrics in groups.values():
        _metrics_check(metrics)
        n, actual, predicted = metrics.sample_count, metrics.actual_up_count, metrics.predicted_up_count
        # 从总正确数、真实上涨数、预测上涨数反推四格计数；负数/半个人都说明报告矛盾。
        twice_true_positive = metrics.correct_count + actual + predicted - n
        _require(twice_true_positive % 2 == 0, "classification counts cannot form a confusion matrix")
        true_positive = twice_true_positive // 2
        true_negative = metrics.correct_count - true_positive
        _require(0 <= true_positive <= min(actual, predicted), "invalid true positive count")
        _require(0 <= true_negative <= min(n - actual, n - predicted), "invalid true negative count")
        if actual in (0, n):
            _require(metrics.balanced_accuracy is None, "single-class balanced accuracy must be absent")
        else:
            balanced = (Decimal(true_positive) / actual + Decimal(true_negative) / (n - actual)) / 2
            _require(
                metrics.balanced_accuracy is not None
                and metrics.balanced_accuracy.is_finite()
                and abs(metrics.balanced_accuracy - balanced) <= ROUND_UNIT,
                "balanced accuracy mismatch",
            )
    total = comparison.validation
    for field in ("sample_count", "correct_count", "actual_up_count", "predicted_up_count"):
        _require(getattr(total, field) == sum(getattr(groups[f], field) for f in FUNDS), "metric total mismatch")
    weighted_brier = sum(groups[f].brier_score * groups[f].sample_count for f in FUNDS) / total.sample_count
    _require(abs(total.brier_score - weighted_brier) <= ROUND_UNIT, "weighted Brier mismatch")
    return groups


def validate_reliability(report: ReliabilityReport, metrics: BaselineMetrics) -> None:
    """核验固定五档、空档、真实上涨数、误差及加权ECE；缺字段不填默认值。"""
    _require(report.sample_count == metrics.sample_count and report.sample_count > 0, "reliability sample mismatch")
    _require(len(report.bins) == 5 and report.ece.is_finite() and 0 <= report.ece <= 1, "reliability shape invalid")
    count, up_count, weighted_gap = 0, 0, Decimal(0)
    for index, bucket in enumerate(report.bins):
        lower, upper = Decimal(index) / 5, Decimal(index + 1) / 5
        _require((bucket.lower, bucket.upper) == (lower, upper), "fixed bin edges mismatch")
        _require(type(bucket.count) is int and 0 <= bucket.count <= report.sample_count, "invalid bin count")
        _require(bucket.enough_samples == (bucket.count >= 30), "bin sufficiency flag mismatch")
        values = (bucket.mean_score, bucket.observed_up_rate, bucket.absolute_gap)
        if bucket.count == 0:
            _require(all(value is None for value in values), "empty bin must retain null values")
            continue
        _require(
            all(value is not None and value.is_finite() and 0 <= value <= 1 for value in values), "invalid bin value"
        )
        mean, rate, gap = values
        _require(lower <= mean < upper or index == 4 and mean == 1, "mean outside fixed bin")
        _require(abs(gap - abs(mean - rate)) <= ARITHMETIC_TOLERANCE, "bin absolute gap mismatch")
        positives = rate * bucket.count
        rounded = positives.to_integral_value()
        _require(abs(positives - rounded) <= ARITHMETIC_TOLERANCE, "bin positive count is not integral")
        up_count += int(rounded)
        count += bucket.count
        weighted_gap += gap * bucket.count / report.sample_count
    _require((count, up_count) == (metrics.sample_count, metrics.actual_up_count), "bin population mismatch")
    _require(report.ece == weighted_gap.quantize(ROUND_UNIT, rounding=ROUND_HALF_UP), "weighted ECE mismatch")


def _validate_bin_totals(total: ReliabilityReport, groups: tuple[ReliabilityReport, ...]) -> None:
    for index, bucket in enumerate(total.bins):
        members = tuple(group.bins[index] for group in groups)
        _require(bucket.count == sum(member.count for member in members), "aggregate bin count mismatch")
        if not bucket.count:
            continue
        for field in ("mean_score", "observed_up_rate"):
            weighted = sum(getattr(member, field) * member.count for member in members if member.count) / bucket.count
            _require(abs(getattr(bucket, field) - weighted) <= ARITHMETIC_TOLERANCE, "aggregate bin value mismatch")


def validate_window_evidence(window: CalibrationWindowReport) -> None:
    """不足/拒绝窗不能夹带成绩；完整窗须与明确模型及逐基金考试数一致。"""
    funds = {f.fund_code: f for f in window.funds}
    _require(len(window.funds) == len(FUNDS) and set(funds) == set(FUNDS), "window fund scope mismatch")
    minimums = {"FIT": 252, "CALIBRATION": 60, "EXAM": window.window.minimum_exam_per_fund}
    for group in funds.values():
        for collection in (group.counts, group.missing, group.purged):
            _require(set(collection) == set(minimums), "window count stages mismatch")
            _require(all(type(n) is int and n >= 0 for n in collection.values()), "negative/noninteger window count")
        _require(
            group.missing == {s: max(0, m - group.counts[s]) for s, m in minimums.items()}, "window deficit mismatch"
        )
    if window.status != "EVALUATED":
        _require(
            all(
                value is None
                for value in (
                    window.model,
                    window.before,
                    window.after,
                    window.reliability_before,
                    window.reliability_after,
                    window.brier_delta,
                    window.ece_delta,
                )
            )
            and not (window.baselines or window.reliability_per_fund or window.prediction_preview),
            "unevaluated window contains numerical evidence",
        )
        return
    _require(
        window.model is not None and window.before is not None and window.after is not None, "model/scores missing"
    )
    _require(not any(any(f.missing.values()) for f in funds.values()), "evaluated window below original sample minima")
    model = window.model
    _require(
        model.base_model.train_counts_per_fund == {f: funds[f].counts["FIT"] for f in FUNDS}, "fit model/count mismatch"
    )
    _require(
        model.calibrator.counts_per_fund == {f: funds[f].counts["CALIBRATION"] for f in FUNDS},
        "calibrator/count mismatch",
    )
    _require(
        (
            model.base_model.train_start_date,
            model.base_model.train_end_date,
            model.calibrator.start_date,
            model.calibrator.end_date,
        )
        == (
            date(2022, 1, 1),
            window.window.fit_end_date,
            window.window.fit_end_date + timedelta(days=1),
            window.window.calibration_end_date,
        ),
        "model/window dates mismatch",
    )
    _require(window.before.baseline_id == "UNCALIBRATED_SAME_BASE", "before score identity mismatch")
    _require(window.after.baseline_id == "CALIBRATED_SAME_BASE", "after score identity mismatch")
    before, after = validate_comparison(window.before), validate_comparison(window.after)
    for comparison in (window.before, window.after, *window.baselines):
        groups = validate_comparison(comparison)
        for fund in FUNDS:
            _require(groups[fund].sample_count == funds[fund].counts["EXAM"], "exam score/count mismatch")
            _require(groups[fund].actual_up_count == after[fund].actual_up_count, "exam label population mismatch")
    _require(window.reliability_before is not None and window.reliability_after is not None, "reliability missing")
    validate_reliability(window.reliability_before, before["ALL"])
    validate_reliability(window.reliability_after, after["ALL"])
    reliability = {f.fund_code: f for f in window.reliability_per_fund}
    _require(
        len(window.reliability_per_fund) == len(FUNDS) and set(reliability) == set(FUNDS), "reliability funds mismatch"
    )
    for fund, group in reliability.items():
        validate_reliability(group.before, before[fund])
        validate_reliability(group.after, after[fund])
    _validate_bin_totals(window.reliability_before, tuple(reliability[f].before for f in FUNDS))
    _validate_bin_totals(window.reliability_after, tuple(reliability[f].after for f in FUNDS))
    _require(window.brier_delta == after["ALL"].brier_score - before["ALL"].brier_score, "Brier delta mismatch")
    _require(window.ece_delta == window.reliability_after.ece - window.reliability_before.ece, "ECE delta mismatch")
