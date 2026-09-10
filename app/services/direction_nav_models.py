"""固定七特征逻辑回归；只接收本窗 FIT 标签，EXAM 仅包含输入。"""

from datetime import date

from app.schemas.direction_training import DirectionInput
from app.services.direction_nav_protocol import VERSION
from app.services.direction_training_models import prepared
from app.services.historical_nav_training import (
    fit_logistic_artifact,
    predict_artifact_scores,
    restore_logistic_artifact,
)


def execute_job(payload):
    if set(payload) != {"version", "window", "branch", "fit", "exam"} or payload["version"] != VERSION:
        raise ValueError("JOB_FIELDS_OR_VERSION")
    window = payload["window"]
    if set(window) != {"name", "fit_end", "cal_end", "exam_end"}:
        raise ValueError("WINDOW_FIELDS")
    fit_end, cal_end, exam_end = (date.fromisoformat(window[k]) for k in ("fit_end", "cal_end", "exam_end"))
    if not date(2021, 1, 1) <= fit_end < cal_end < exam_end <= date(2024, 12, 31):
        raise ValueError("JOB_TIME_BOUNDARY")
    if payload["branch"] not in ("LEGACY", "COMPLETE", "TOLERANT"):
        raise ValueError("JOB_BRANCH")
    fit = prepared(payload["fit"], date(2020, 12, 31), fit_end)
    if any(sum(r.fund_code == f for r in fit) < 252 for f in ("001632", "006730", "008888")):
        raise ValueError("FIT_INSUFFICIENT")
    variants = payload["exam"]
    expected = {"CLEAN"} if payload["branch"] == "LEGACY" else {"CLEAN", "DROP1", "DROP2", "NATURAL"}
    if set(variants) != expected:
        raise ValueError("EXAM_VARIANTS")
    parsed = {k: [DirectionInput.model_validate(row) for row in rows] for k, rows in variants.items()}
    for items in parsed.values():
        if len(items) > 10000 or len({i.key for i in items}) != len(items):
            raise ValueError("EXAM_BUDGET_OR_DUPLICATE")
        if any(not cal_end < i.cutoff <= exam_end for i in items):
            raise ValueError("EXAM_TIME_BOUNDARY")
    model = fit_logistic_artifact(
        fit,
        start_date=date(2021, 1, 1),
        end_date=fit_end,
        versions={"feature": VERSION, "label": "CASH_REINVESTMENT_FORWARD_20TD_V2", "research": VERSION},
    )
    artifact = model.model_dump(mode="json")
    restored = restore_logistic_artifact(model.model_dump_json())
    scores = {
        k: list(predict_artifact_scores(restored, tuple(i.x for i in items))) if items else []
        for k, items in parsed.items()
    }
    return {"status": "PREDICTED", "scores": scores, "artifact": artifact}
