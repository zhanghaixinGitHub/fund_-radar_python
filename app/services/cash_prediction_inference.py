"""现金模型的纯数值内核；不查库、不训练，不自行授予发布资格或暴露HTTP。"""

from decimal import ROUND_HALF_UP, Decimal

from app.schemas.cash_prediction_features import CashPredictionFeature
from app.schemas.cash_prediction_inference import CashInferenceValue
from app.schemas.historical_nav_calibration import CalibratedModelArtifact
from app.services.calibration_policy import calibration_diagnostic
from app.services.cash_reinvestment_research import VERSIONS
from app.services.cash_reinvestment_samples import _hash
from app.services.historical_nav_calibration import predict_calibrated_scores, restore_calibrated_artifact
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.trading_calendar import load_prediction_calendar


def calculate_cash_inference(
    feature: CashPredictionFeature,
    artifact: CalibratedModelArtifact,
    *,
    expected_model_hash: str,
) -> CashInferenceValue:
    """只在服务内使用；上线生成流程须在调用前完成正式发布校验，不能用本函数绕过闸门。"""
    # model_copy不重新校验，内部调用同样重新读JSON，拒绝被改动的模型/输入结构。
    feature = CashPredictionFeature.model_validate_json(feature.model_dump_json())
    model = restore_calibrated_artifact(artifact.model_dump_json())
    payload = feature.feature_payload
    if model.model_hash != expected_model_hash or model.base_model.versions != VERSIONS:
        raise ValueError("cash inference model hash or data versions mismatch")
    if model.base_model.feature_names != FEATURE_NAMES:
        raise ValueError("cash inference feature order invalid")
    diagnostic = calibration_diagnostic(model.calibrator.slope, model.calibrator.intercept)
    if (
        feature.status != "INPUT_READY"
        or payload is None
        or feature.input_issues
        or feature.anchor_lag_sessions not in (0, 1)
        or feature.feature_hash != _hash(payload)
    ):
        raise ValueError("cash inference requires complete, intact history input")
    if (
        feature.fund_code != payload.fund_code
        or feature.cutoff_date != payload.cutoff_date
        or feature.anchor_nav_date != payload.anchor_nav_date
        or payload.available_at > feature.cutoff_date
        or payload.source_code != "TUSHARE_PRO_FUND"
        or feature.fund_code not in model.base_model.train_counts_per_fund
        or feature.fund_code not in model.calibrator.counts_per_fund
        or feature.cutoff_date <= model.calibrator.end_date
        or feature.cutoff_date.year == 2025
    ):
        raise ValueError("cash inference fund, source, time boundary or protected test period invalid")

    calendar = load_prediction_calendar(feature.cutoff_date)
    base_index = calendar.at_or_before_index(feature.cutoff_date)
    anchor_index = calendar.at_or_before_index(payload.anchor_nav_date)
    if (
        (payload.calendar_version, payload.calendar_hash) != (calendar.definition.version, calendar.content_hash)
        or anchor_index < 60
        or calendar.sessions[anchor_index] != payload.anchor_nav_date
        or base_index - anchor_index != feature.anchor_lag_sessions
        or feature.history_dates != calendar.sessions[anchor_index - 60 : anchor_index + 1]
        or tuple(p.nav_date for p in payload.history_series) != feature.history_dates
        or any(p.available_at > feature.cutoff_date for p in payload.history_series)
        or max(p.available_at for p in payload.history_series) != payload.available_at
        or set(payload.metrics) != set(FEATURE_NAMES)
    ):
        raise ValueError("cash inference calendar, known history or metrics mismatch")
    future_dates = calendar.future_sessions(feature.cutoff_date, 20)
    # 跨年安排在cutoff当时可能尚未公布；知道今天的日历不等于当年已经知道。
    target_years = {calendar.sessions[base_index].year, *(d.year for d in future_dates)}
    if any(n.announced_on > feature.cutoff_date for n in calendar.definition.years if n.year in target_years):
        raise ValueError("cash inference target calendar was not available at cutoff")
    x = tuple(Decimal(payload.metrics[name]) for name in FEATURE_NAMES)
    if any(not n.is_finite() for n in x):
        raise ValueError("cash inference input contains nonfinite values")
    score = Decimal(str(predict_calibrated_scores(model, (x,))[0]))
    if not score.is_finite() or not 0 <= score <= 1:
        raise ValueError("cash inference produced invalid score")
    score = score.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
    return CashInferenceValue(
        fund_code=feature.fund_code,
        cutoff_date=feature.cutoff_date,
        anchor_nav_date=payload.anchor_nav_date,
        target_base_date=calendar.sessions[base_index],
        target_end_date=future_dates[-1],
        feature_hash=feature.feature_hash,
        model_hash=model.model_hash,
        up_score=score,
        predicted_up=score > Decimal("0.5"),
        calibration=diagnostic,
    )
