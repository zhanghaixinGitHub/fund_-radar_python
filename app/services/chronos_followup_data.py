"""2026盲测专用只读资料；原2022—2024样本和2025保护规则保持独立。"""

from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Self
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.repositories.cash_prediction_features import read_cash_history_inputs
from app.repositories.historical_nav import read_historical_nav_source
from app.schemas.cash_prediction_features import CashPredictionFeatureRequest
from app.schemas.model_comparison import ComparisonInput
from app.services.cash_prediction_features import read_cash_prediction_feature_in_session
from app.services.cash_reinvestment_samples import _number, build_cash_return_series
from app.services.historical_nav_evaluation import FEATURE_NAMES
from app.services.historical_nav_samples import _build_metrics
from app.services.model_comparison_artifacts import file_hash, fingerprint, read_jsonl, write_json, write_jsonl
from app.services.model_comparison_protocol import local_engine
from app.services.trading_calendar import load_current_calendar


class FollowupInput(ComparisonInput):
    """复用输入形状，使用独立的2026日期契约；不放宽旧ComparisonInput。"""

    cutoff_date: date = Field(ge=date(2026, 1, 1), le=date(2026, 12, 31))

    @model_validator(mode="after")
    def check_values(self) -> Self:
        calendar = load_current_calendar()
        index = calendar.at_or_before_index(self.cutoff_date)
        anchor = index - self.anchor_lag_sessions
        if (
            self.label_base_date != calendar.sessions[index]
            or self.label_end_date != calendar.future_sessions(self.cutoff_date)[-1]
            or self.anchor_nav_date != calendar.sessions[anchor]
            or self.history_dates != calendar.sessions[anchor - 60 : anchor + 1]
        ):
            raise ValueError("2026 forecast/calendar alignment mismatch")
        if any(d > self.cutoff_date or d.year != 2026 for d in self.history_available_at):
            raise ValueError("history unavailable or intersects protected year")
        if any(not x.is_finite() or x <= 0 for x in self.history_values) or any(not x.is_finite() for x in self.x):
            raise ValueError("nonfinite or nonpositive history")
        with localcontext() as ctx:
            ctx.prec, ctx.rounding = 40, ROUND_HALF_UP
            metrics = _build_metrics(self.history_values)
        if metrics is None or tuple(Decimal(metrics[k]) for k in FEATURE_NAMES) != self.x:
            raise ValueError("seven features differ from shared history")
        if self.input_hash != "0" * 64 and self.input_hash != fingerprint(
            self.model_dump(mode="json", exclude={"input_hash"})
        ):
            raise ValueError("input hash mismatch")
        return self


def feature_to_input(feature, run_id: UUID) -> FollowupInput:
    payload = feature.feature_payload
    if payload is None or payload.calendar_hash != load_current_calendar().content_hash:
        raise ValueError("missing input or calendar mismatch")
    calendar = load_current_calendar()
    item = FollowupInput(
        fund_code=feature.fund_code,
        cutoff_date=feature.cutoff_date,
        batch_id=run_id,
        anchor_nav_date=feature.anchor_nav_date,
        anchor_lag_sessions=feature.anchor_lag_sessions,
        label_base_date=calendar.sessions[calendar.at_or_before_index(feature.cutoff_date)],
        label_end_date=calendar.future_sessions(feature.cutoff_date)[-1],
        history_dates=feature.history_dates,
        history_values=tuple(Decimal(p.growth_index) for p in payload.history_series),
        history_available_at=tuple(p.available_at for p in payload.history_series),
        x=tuple(Decimal(payload.metrics[k]) for k in FEATURE_NAMES),
        source_feature_hash=feature.feature_hash,
        input_hash="0" * 64,
    )
    return item.model_copy(update={"input_hash": fingerprint(item.model_dump(mode="json", exclude={"input_hash"}))})


def export_followup_inputs(folder: Path, protocol: dict) -> dict:
    from time import perf_counter

    if date.fromisoformat(protocol["observation_date"]) >= datetime.now(ZoneInfo("Asia/Shanghai")).date():
        raise ValueError("observation day must have closed")
    items, failures = [], []
    with Session(local_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        if session.scalar(text("SHOW transaction_read_only")) != "on":
            raise ValueError("database must be read only")
        for fund in protocol["funds"]:
            for day in protocol["planned_cutoffs"]:
                request = CashPredictionFeatureRequest(fundCode=fund, cutoffDate=day)
                feature = read_cash_prediction_feature_in_session(session, request, deadline=perf_counter() + 15)
                if feature.feature_payload:
                    item = feature_to_input(feature, UUID(protocol["run_id"]))
                    items.append(item.model_dump(mode="json"))
                else:
                    failures.append(
                        {"fund_code": fund, "cutoff_date": day, "reasons": [i.code for i in feature.input_issues]}
                    )
    write_jsonl(folder / "inputs.jsonl", items)
    manifest = {
        "protocol_hash": protocol["protocol_hash"],
        "input_count": len(items),
        "failures": failures,
        "planned_count": len(protocol["planned_cutoffs"]) * len(protocol["funds"]),
        "database_read_only": True,
        "answers_read": False,
        "input_sha256": file_hash(folder / "inputs.jsonl"),
    }
    write_json(folder / "inputs_manifest.json", manifest)
    return manifest


def load_followup_inputs(folder: Path) -> dict[str, FollowupInput]:
    result = {}
    for raw in read_jsonl(folder / "inputs.jsonl"):
        if raw.get("input_hash") == "0" * 64:
            raise ValueError("unsealed input")
        item = FollowupInput.model_validate(raw)
        if item.key in result:
            raise ValueError("duplicate input")
        result[item.key] = item
    return result


def answer_from_series(item, nav, events, *, available_by: date):
    """标签独立计算，沿用原现金再投公式与12位零收益判定。"""
    calendar = load_current_calendar()
    dates = (item.label_base_date, *calendar.future_sessions(item.cutoff_date))
    if dates[-1] > available_by or available_by.year != 2026:
        return {"key": item.key, "y": None, "reason": "ANSWER_NOT_MATURE"}
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = 40, ROUND_HALF_UP
        series, issues = build_cash_return_series(
            dates, {p.nav_date: p for p in nav}, events, latest_available=available_by
        )
        if issues:
            return {"key": item.key, "y": None, "reason": ",".join(sorted({i.code for i in issues}))}
        value = _number(Decimal(series[-1].growth_index) / 100 - 1)
    return {
        "key": item.key,
        "input_hash": item.input_hash,
        "y": int(Decimal(value) > 0),
        "future_return_20d": value,
        "label_available_at": str(series[-1].available_at),
        "label_series": [p.model_dump(mode="json") for p in series],
        "reason": None,
    }


def export_followup_answers(folder: Path, protocol: dict, inputs: dict) -> list[dict]:
    """调用者必须先核验预测冻结回执；本函数也在开库前核验文件。"""
    from app.services.model_comparison_artifacts import read_json

    receipt = read_json(folder / "predictions_frozen.json")
    if (
        receipt["predictions_sha256"] != file_hash(folder / "predictions.jsonl")
        or receipt["protocol_hash"] != protocol["protocol_hash"]
    ):
        raise ValueError("predictions not frozen")
    available = date.fromisoformat(protocol["observation_date"])
    answers = []
    with Session(local_engine()) as session, session.begin():
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        if session.scalar(text("SHOW transaction_read_only")) != "on":
            raise ValueError("database must be read only")
        for fund in protocol["funds"]:
            source = read_historical_nav_source(session, fund_code=fund)
            for item in inputs.values():
                if item.fund_code != fund:
                    continue
                nav, events = read_cash_history_inputs(
                    session, fund_code=fund, source_id=source.source_id, start=item.label_base_date, cutoff=available
                )
                answers.append(answer_from_series(item, nav, events, available_by=available))
    write_jsonl(folder / "answers.jsonl", answers)
    return answers
