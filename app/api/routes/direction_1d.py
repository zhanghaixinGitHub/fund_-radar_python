"""Java专用1日实验API；浏览器不可直连，读取不训练，写入字段严格白名单。"""

from datetime import datetime, time, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from app.api.dependencies import require_service_token
from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.schemas.direction_1d import AssessmentAck, ForecastRequest, Scope, TrainingRequest
from app.services.direction_1d_data import inventory
from app.services.direction_1d_inference import labels
from app.services.direction_1d_jobs import submit, submit_forecast
from app.services.direction_1d_protocol import ZONE, calendar, window

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.get("/status")
def status():
    now = repo.clock()
    with get_engine().connect() as c:
        latest = (
            c.execute(text("""SELECT kind,state,finished_at,result->>'status' AS result_status,
                result->>'reason' AS reason FROM direction_1d_job ORDER BY created_at DESC LIMIT 5"""))
            .mappings()
            .all()
        )
    return {
        "server_time": datetime.now(ZONE).isoformat(),
        "database_time": now.isoformat(),
        "window": window(now),
        "model_released": False,
        "up_probability": None,
        "recent_jobs": [dict(r) for r in latest],
        "training_policy_note": (
            "每周日12点检查一次；无新增已核对样本则跳过。2025受保护，2026只纳入启动后已核对样本，"
            "近504交易日不足252个基准日时沿用旧模型。"
        ),
    }


@router.post("/coverage")
def coverage(request: Scope):
    return inventory(request.fund_codes)


@router.post("/forecast-jobs", status_code=202)
def forecast(request: ForecastRequest):
    try:
        return submit_forecast(request.fund_code)
    except ValueError as error:
        raise HTTPException(429 if str(error) == "QUEUE_FULL" else 409, str(error)) from error


@router.get("/forecast-jobs/{job_id}")
@router.get("/training-jobs/{job_id}")
def job(job_id: UUID):
    result = repo.get_job(job_id)
    if not result:
        raise HTTPException(404, "JOB_NOT_FOUND")
    return result


@router.get("/labels/{job_id}")
def read_label(job_id: UUID):
    return labels(job_id)


@router.post("/labels/{job_id}/assessed")
def assessed(job_id: UUID, request: AssessmentAck):
    """仅在Java提交核对之后调用；以后训练的成熟时间取本次接纳时刻，不能由客户端倒填。"""
    job = repo.get_job(job_id)
    if not job or job["kind"] != "FORECAST" or job["state"] != "SUCCEEDED":
        raise HTTPException(404, "FORECAST_NOT_FOUND")
    import json

    task_key = json.loads(job["result"]["payload_json"])["task_key"]
    with get_engine().begin() as c:
        exists = c.execute(
            text(
                "SELECT 1 FROM direction_1d_snapshot WHERE task_key=:key AND kind='LABEL' "
                "AND content_hash=:hash AND payload->>'training_eligible'='true'"
            ),
            {"key": task_key, "hash": request.label_hash},
        ).first()
        if not exists:
            raise HTTPException(409, "LABEL_HASH_MISMATCH")
        c.execute(
            text("""INSERT INTO direction_1d_assessment_ack(task_key,label_hash,assessed_at)
          VALUES(:key,:hash,clock_timestamp()) ON CONFLICT DO NOTHING"""),
            {"key": task_key, "hash": request.label_hash},
        )
    return {"status": "ACKNOWLEDGED"}


@router.post("/training-jobs", status_code=202)
def training(_: TrainingRequest):
    now = repo.clock().astimezone(ZONE)
    sunday = now.date() - timedelta(days=(now.weekday() + 1) % 7)
    opened = datetime.combine(sunday, time(12), ZONE)
    days, _ = calendar()
    next_day = next((d for d in days if d > sunday), None)
    if not next_day or not opened <= now < datetime.combine(next_day, time(12), ZONE):
        return {"state": "NOT_DUE", "reason": "WEEKLY_WINDOW_CLOSED"}
    return submit("WEEKLY:" + str(sunday), "TRAINING", {})
