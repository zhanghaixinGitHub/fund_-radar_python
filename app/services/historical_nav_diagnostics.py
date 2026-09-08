"""复盘已保存模型：只重算分数和描述统计，不fit、不选特征、不改变任何训练方案。"""

from collections import Counter
from datetime import date
from decimal import Decimal
from statistics import fmean

from app.schemas.historical_nav_calibration import CalibratedModelArtifact, HistoricalNavCalibrationResponse
from app.schemas.historical_nav_training import LogisticModelArtifact
from app.services.historical_nav_calibration import PROTOCOL, _window_rows, predict_calibrated_scores
from app.services.historical_nav_evaluation import (
    FEATURE_NAMES,
    PreparedDataset,
    PreparedRow,
    calculate_baseline_metrics,
)
from app.services.historical_nav_samples import (
    HistoricalNavPoint,
    HistoricalNavSampleInput,
    build_historical_nav_samples,
)
from app.services.historical_nav_training import predict_artifact_logits, sigmoid, training_rows_hash


def _number(value: float) -> float:
    return round(value, 10)


def _statistics(rows: tuple[PreparedRow, ...], model: CalibratedModelArtifact) -> dict:
    """一组历史行的描述统计；FIT和CAL成绩是样本内成绩，不能当独立考试。"""
    if not rows:
        return {"sample_count": 0, "status": "EMPTY_GROUP"}
    x, y = tuple(r.x for r in rows), tuple(r.y for r in rows)
    logits = predict_artifact_logits(model.base_model, x)
    before = tuple(Decimal(str(sigmoid(z))) for z in logits)
    after = tuple(Decimal(str(p)) for p in predict_calibrated_scores(model, x))
    up_logits = [z for z, label in zip(logits, y, strict=True) if label == 1]
    down_logits = [z for z, label in zip(logits, y, strict=True) if label == 0]
    return {
        "sample_count": len(rows),
        "actual_up_rate": _number(sum(y) / len(y)),  # 本段真正上涨的比例，不是预测概率。
        "mean_score_before": _number(fmean(before)),
        "mean_score_after": _number(fmean(after)),
        # 正数：上涨样本平均基础得分较高；负数：此段平均关系反了。不做因果或显著性推断。
        "up_minus_down_mean_logit": _number(fmean(up_logits) - fmean(down_logits))
        if up_logits and down_logits
        else None,
        "before": calculate_baseline_metrics(y, before).model_dump(mode="json"),
        "after": calculate_baseline_metrics(y, after).model_dump(mode="json"),
    }


def _feature_statistics(
    rows: tuple[PreparedRow, ...], fit_rows: tuple[PreparedRow, ...], base: LogisticModelArtifact
) -> list[dict]:
    """固定七列全部报告；比较均值和历史范围，不按考试答案筛特征或生成新模型。"""
    result = []
    for index, name in enumerate(FEATURE_NAMES):
        values = [float(r.x[index]) for r in rows]
        fit_values = [float(r.x[index]) for r in fit_rows]
        low, high = min(fit_values), max(fit_values)
        result.append(
            {
                "feature": name,
                "mean": _number(fmean(values)),
                "fit_mean": base.mean[index],
                "fit_scale": base.scale[index],
                # 采用已冻结的全基金FIT尺度，不再用考试数据拟合标准化。
                "mean_shift_in_fit_scale": _number((fmean(values) - base.mean[index]) / base.scale[index]),
                "outside_fit_range_rate": _number(sum(v < low or v > high for v in values) / len(values)),
                "coefficient": base.coefficients[index],
                "mean_absolute_linear_contribution": _number(
                    fmean(abs((v - base.mean[index]) / base.scale[index] * base.coefficients[index]) for v in values)
                ),  # 对本模型线性得分的数值贡献；相关指标不能据此作因果排名。
            }
        )
    return result


def diagnose_prepared(data: PreparedDataset, saved: HistoricalNavCalibrationResponse) -> dict:
    """验证旧产物确实对应当前批次后，仅诊断原TRAIN和VALIDATION，从不引用data.test。"""
    if data.report.dataset_hash != saved.preparation.dataset_hash or saved.protocol != PROTOCOL:
        raise ValueError("diagnostic dataset/protocol differs from saved experiment")
    if saved.status != "CALIBRATION_EVALUATED" or tuple(w.window for w in saved.windows) != PROTOCOL.windows:
        raise ValueError("diagnostics requires all three completed frozen windows")
    windows = []
    for old in saved.windows:
        if old.model is None or old.status != "EVALUATED":
            raise ValueError("missing saved model")
        rows, counts = _window_rows(data, old.window)
        model = old.model
        if (
            counts != old.funds
            or training_rows_hash(rows["FIT"]) != model.base_model.train_hash
            or training_rows_hash(rows["CALIBRATION"]) != model.calibrator.calibration_hash
            or model.base_model.versions != data.report.versions
        ):
            raise ValueError("saved model does not match diagnostic input")
        stages = {}
        for stage, subset in rows.items():
            stages[stage] = {
                "use": "OUT_OF_FIT_EXAM" if stage == "EXAM" else "IN_SAMPLE_DESCRIPTIVE_ONLY",
                "overall": _statistics(subset, model),
                "per_fund": {
                    f.fund_code: _statistics(tuple(r for r in subset if r.fund_code == f.fund_code), model)
                    for f in counts
                },
                "features": _feature_statistics(subset, rows["FIT"], model.base_model),
            }
        # 先重现旧报告，防止诊断悄悄对另一批行出分。
        exam = stages["EXAM"]["overall"]
        if exam["before"] != old.before.validation.model_dump(mode="json") or exam[
            "after"
        ] != old.after.validation.model_dump(mode="json"):
            raise ValueError("saved exam metrics do not replay")
        quarters = {}
        if old.window.role == "FIXED_VALIDATION":
            for quarter in range(1, 5):
                subset = tuple(r for r in rows["EXAM"] if (r.available_at.month - 1) // 3 + 1 == quarter)
                quarters[f"2024_Q{quarter}"] = {
                    "overall": _statistics(subset, model),
                    "per_fund": {
                        f.fund_code: _statistics(tuple(r for r in subset if r.fund_code == f.fund_code), model)
                        for f in counts
                    },
                }
        windows.append(
            {
                "window_id": old.window.window_id,
                "model_hash": model.model_hash,
                "calibration_slope": model.calibrator.slope,
                "calibration_intercept": model.calibrator.intercept,
                "stages": stages,
                "exam_quarters": quarters,  # 原考试集合按输入公告季度拆分，不新增季度训练/标签剔除。
                "same_exam_baselines": [b.model_dump(mode="json") for b in old.baselines],
            }
        )
    return {
        "version": "NAV_FAILURE_DIAGNOSTICS_V1",
        "purpose": "LEARNING_ONLY",
        "publication_status": "MODEL_NOT_RELEASED",
        "training_eligible": False,
        "model_fitted": False,
        "test_scored": False,
        "dataset_hash": data.report.dataset_hash,
        "windows": windows,
        "limitations": [
            "这是看过既有结果后的探索性复盘，不是新的独立测试或根因证明。",
            "FIT和CAL统计只描述样本内表现；EXAM才是对应模型未拟合的历史。",
            "不使用2025评分；既有准备层仍读取2025作完整性和数据指纹校验。",
            "20条净值标签重叠，样本数不是独立试验数；不输出显著性或收益承诺。",
            "季度按available_at分组，答案可跨季度但不跨原考试截止；不另训季度模型。",
            "特征均值变化和线性贡献仅为关联线索，不证明某指标导致失败。",
        ],
    }


def audit_nav_snapshot(data: PreparedDataset, nav_rows: list[dict]) -> list[dict]:
    """比较当前源快照与已存2022–2024可用样本；不恢复历史版本、不修改标签口径。"""
    if len(nav_rows) > 4500 or any(not date(2021, 1, 1) <= r["nav_date"] <= date(2024, 12, 31) for r in nav_rows):
        raise ValueError("raw audit exceeds fixed pre-test scope")
    allowed = {f.fund_code for f in data.report.funds}
    if any(r["fund_code"] not in allowed for r in nav_rows):
        raise ValueError("raw audit contains unexpected fund")
    output = []
    for fund in data.report.funds:
        raw = [r for r in nav_rows if r["fund_code"] == fund.fund_code]
        dates = [r["nav_date"] for r in raw]
        if dates != sorted(set(dates)) or len({r["source_id"] for r in raw}) > 1:
            raise ValueError("raw audit is unordered/duplicated/mixed source")
        selected = tuple(r for r in (*data.train, *data.validation) if r.fund_code == fund.fund_code)
        # 只读重放当前累计净值构建器，2021数据仅用来给2022起点提供历史。
        rebuilt = (
            {
                s.as_of_date: s
                for s in build_historical_nav_samples(
                    HistoricalNavSampleInput(
                        fund_code=fund.fund_code,
                        fund_type="STOCK",
                        source_code=data.report.source_code,
                        source_sync_run_id=None,
                        nav_points=tuple(
                            HistoricalNavPoint(r["nav_date"], r["ann_date"], r["unit_nav"], r["accumulated_nav"])
                            for r in raw
                        ),
                    )
                )
            }
            if raw
            else {}
        )
        indices = {d: i for i, d in enumerate(dates)}
        failures, examples, adjusted_gaps = Counter(), [], []
        sign_changes, compared, weekend_windows = 0, 0, 0
        for row in selected:
            sample = rebuilt.get(row.as_of_date)
            reason = None
            if sample is None or sample.eligibility_status != "SCORABLE":
                reason = "CURRENT_SAMPLE_MISSING_OR_UNSCORABLE"
            elif (
                sample.available_at != row.available_at
                or tuple(Decimal(sample.feature_payload["metrics"][f]) for f in FEATURE_NAMES) != row.x
                or sample.offline_label.label_up_20d != row.y
                or sample.offline_label.label_available_at != row.label_available_at
            ):
                reason = "CURRENT_INPUT_OR_LABEL_DIFFERS"
            if reason:
                failures[reason] += 1
                if len(examples) < 5:
                    examples.append({"as_of_date": row.as_of_date.isoformat(), "reason": reason})
            index = indices.get(row.as_of_date)
            if index is None or index + 20 >= len(raw):
                continue
            window = raw[index : index + 21]
            weekend_windows += any(r["nav_date"].weekday() >= 5 for r in window[1:])
            start, end = window[0], window[-1]
            values = [r[k] for r in (start, end) for k in ("accumulated_nav", "adjusted_nav")]
            if any(v is None or not v.is_finite() or v <= 0 for v in values):
                continue
            accumulated_return = end["accumulated_nav"] / start["accumulated_nav"] - 1
            adjusted_return = end["adjusted_nav"] / start["adjusted_nav"] - 1
            compared += 1
            sign_changes += (accumulated_return > 0) != (adjusted_return > 0)
            adjusted_gaps.append(abs(adjusted_return - accumulated_return))
        weekends = [d.isoformat() for d in dates if d.weekday() >= 5]
        output.append(
            {
                "fund_code": fund.fund_code,
                "raw_row_count": len(raw),
                "raw_start": dates[0].isoformat() if dates else None,
                "raw_end": dates[-1].isoformat() if dates else None,
                "missing_or_invalid_ann_date": sum(r["ann_date"] is None or r["ann_date"] < r["nav_date"] for r in raw),
                "source_published_at_missing": sum(r["source_published_at"] is None for r in raw),
                "earliest_local_created_at": str(min(r["created_at"] for r in raw)) if raw else None,
                "weekend_nav_dates": weekends,  # 周末业务日是日历核验线索；不据此直接删除来源记录。
                "evaluated_windows_containing_weekend_future_nav": weekend_windows,
                "saved_sample_count_checked": len(selected),
                "current_replay_mismatches": dict(failures),
                "mismatch_examples": examples,
                "basis_comparable_count": compared,
                "accumulated_vs_adjusted_direction_difference_count": sign_changes,
                "mean_absolute_return_gap_bps": _number(fmean(adjusted_gaps) * 10000) if adjusted_gaps else None,
                "max_absolute_return_gap_bps": _number(float(max(adjusted_gaps) * 10000)) if adjusted_gaps else None,
                "note": "复权列只作同端点敏感性比较，不改现有累计净值标签，不证明复权版本正确或分红处理完整。",
            }
        )
    return output
