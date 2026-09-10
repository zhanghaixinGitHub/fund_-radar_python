"""用封存模型计算当前输入；无训练、选模、正式发布或数据库写入。"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import localcontext
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.repositories.cash_forecast import current_cash_source_revision
from app.repositories.direction_experiment import read_experiment_history
from app.repositories.historical_nav import HistoricalNavPreviewReadError, read_historical_nav_source
from app.schemas.direction_experiment import DirectionExperiment, ExperimentModelScore
from app.services.cash_forecast import utc_now
from app.services.direction_linear_models import predict_model, restore
from app.services.direction_nav_data import cash_series
from app.services.direction_training_artifacts import digest
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.historical_nav_samples import _build_metrics
from app.services.trading_calendar import CalendarCoverageError, load_prediction_calendar

FUNDS = ("001632", "006730", "008888")
MODEL_FILE = (
    Path(__file__).resolve().parents[2]
    / ".local-runs/direction-training-0c0e06a9-725e-4b68-b813-de6ff5124b29"
    / "linear-models-DEV_2024_Q4_FULL_V1.json"
)
# 固定最后时间窗口，不按基金、季度表现或当前分数选择。指纹同时防止静默换模型。
MODEL_HASHES = {
    "DROP_60D_GROUP_L2": "b08efdc7bc8fc04833d0c3d6ed1edcbd7c3caff080e4787db837cef04f17660f",
    "REFERENCE": "0146d0ff0ac0504d23a428c0f607d16636e457110ac67c6e60d1b7b7dea19789",
}


@dataclass(frozen=True)
class LiveModelInput:
    """仅给数值推理传入基金和七特征；不扩大离线DirectionInput的年份范围。"""

    fund: str
    x: tuple[float, ...]


def load_experiment_models():
    with MODEL_FILE.open("rb") as handle:
        content = handle.read(65537)
    if len(content) > 65536:
        raise ValueError("EXPERIMENT_MODEL_FILE_TOO_LARGE")
    payload = json.loads(content)
    models = []
    for branch, expected in MODEL_HASHES.items():
        model = restore(payload[branch]["models"]["POOLED"])
        if model["hash"] != expected or model["branch"] != branch or model["fit_end"] != "2024-03-31":
            raise ValueError("EXPERIMENT_MODEL_NOT_PINNED")
        models.append(model)
    return models


def build_experiment_scores(fund, cutoff, dates, nav, events, calendar, models):
    """复用已验收现金再投及七特征公式，使用2026日历，不执行任何fit。"""
    series, audit, issues = cash_series(dates, {p.nav_date: p for p in nav}, events, cutoff, calendar=calendar)
    if issues:
        return (), None, issues
    with localcontext() as context:
        context.prec = 40
        metrics = _build_metrics(series)
    if metrics is None:
        return (), None, ["FLAT_HISTORY_POSITION_UNDEFINED"]
    item = LiveModelInput(fund, tuple(float(metrics[name]) for name in FEATURE_NAMES))
    scores = []
    for model in models:
        score = predict_model(model, [item])[0]
        scores.append(
            ExperimentModelScore(
                branch=model["branch"],
                score=score,
                direction="UP" if score > 0.5 else "NON_UP",
                model_hash=model["hash"],
                fit_end=model["fit_end"],
            )
        )
    audit["original_ann_dates"] = [str(p.ann_date) if p.ann_date else None for p in nav]
    audit["calendar_hash"] = calendar.content_hash
    return tuple(scores), audit, []


def get_direction_experiment(fund_code: str) -> DirectionExperiment:
    """当前日期由服务端决定；只读一致性事务，固定61日输入和两个数值模型。"""
    now = utc_now()
    fields = {"fund_code": fund_code, "read_at": now}

    def unavailable(status, code, message):
        return DirectionExperiment(**fields, status=status, reason_codes=(code,), message=message)

    if fund_code not in FUNDS:
        return unavailable(
            "NOT_APPLICABLE", "OUTSIDE_EXPERIMENT_SCOPE", "目前仅支持001632、006730、008888三只研究基金。"
        )
    try:
        models = load_experiment_models()
    except (OSError, ValueError, KeyError, TypeError):
        return unavailable("UNAVAILABLE", "EXPERIMENT_MODEL_UNAVAILABLE", "本机实验模型文件缺失或校验失败。")
    try:
        closed_day = now.astimezone(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)
        calendar = load_prediction_calendar(closed_day)
        index = calendar.at_or_before_index(closed_day)
        if index < 61:
            raise CalendarCoverageError("历史日历不足")
        cutoff = calendar.sessions[index]
        dates = calendar.sessions[index - 61 : index]
        # 只使用当前年度前向试用，不开放历史任意日期查询，也不读2025数值。
        if dates[0] < date(2026, 1, 1):
            raise CalendarCoverageError("当前年度完整历史不足")
        end = calendar.future_sessions(cutoff)[-1]
    except CalendarCoverageError:
        return unavailable(
            "DATA_INSUFFICIENT", "CALENDAR_COVERAGE_INSUFFICIENT", "已核验日历无法覆盖完整历史和未来20交易日。"
        )
    fields.update(cutoff_date=cutoff, target_base_date=cutoff, target_end_date=end)
    with Session(get_nav_preview_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        try:
            source = read_historical_nav_source(session, fund_code=fund_code)
        except HistoricalNavPreviewReadError as error:
            return unavailable("DATA_INSUFFICIENT", error.code, str(error))
        revision = current_cash_source_revision(session, fund_code=fund_code)
        if revision is None:
            return unavailable(
                "DATA_INSUFFICIENT", "SOURCE_UPDATE_NOT_READY", "来源最近同步未完成或失败，暂不计算实验结果。"
            )
        nav, events = read_experiment_history(
            session, fund_code=fund_code, source_id=source.source_id, dates=dates, cutoff=cutoff
        )
        fields["latest_nav_date"] = max((p.nav_date for p in nav), default=None)
        scores, audit, issues = build_experiment_scores(fund_code, cutoff, dates, nav, events, calendar, models)
        if issues:
            return unavailable(
                "DATA_INSUFFICIENT", issues[0], "完整61交易日净值或现金分红校验未通过，暂不输出方向和分数。"
            )
        return DirectionExperiment(
            **fields,
            status="EXPERIMENTAL",
            models=scores,
            source_revision_id=revision,
            input_hash=digest({"fund": fund_code, "cutoff": str(cutoff), "source": str(revision), "input": audit}),
            message="已用本地净值运行实验模型；两套模型均未证明稳定优势，分数未经独立校准。",
        )
