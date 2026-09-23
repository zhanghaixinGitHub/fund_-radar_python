"""多周期内部API：只能由Java服务调用，LIVE生成时间及模型路由均由服务器决定。"""

from datetime import date, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import require_service_token
from app.repositories.prediction_store import history
from app.services.prediction_contract import PredictionFailure
from app.services.prediction_generation import (
    create_task,
    current_predictions,
    recover_tasks,
    resolve_pending_targets,
    retry_failed,
    task_status,
    verify_outcomes,
)
from app.services.prediction_models import model_status
from app.services.prediction_research import create_research, recover_research, research_status

router = APIRouter(dependencies=[Depends(require_service_token)])
FundCode = Annotated[str, Path(pattern=r"^[0-9]{6}$")]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fundCodes: list[Annotated[str, Field(pattern=r"^[0-9]{6}$")]] = Field(max_length=500)
    requestKey: UUID | None = None


@router.post("/batches", status_code=202)
def generate(request: BatchRequest):
    try:
        return create_task(request.fundCodes, request_key=str(request.requestKey) if request.requestKey else None)
    except PredictionFailure as error:
        raise HTTPException(422, error.payload) from error


@router.get("/batches/{task_id}")
def read_task(task_id: UUID):
    try:
        return task_status(task_id)
    except PredictionFailure as error:
        raise HTTPException(404, error.payload) from error


@router.post("/batches/{task_id}/retry", status_code=202)
def retry(task_id: UUID):
    return retry_failed(task_id)


@router.get("/funds/{fund_code}")
def current(fund_code: FundCode):
    return current_predictions(fund_code)


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictionIds: list[UUID] = Field(max_length=100)


@router.post("/funds/{fund_code}/outcomes")
def outcomes(fund_code: FundCode, request: OutcomeRequest):
    """批量读取公共预测核验，不接收个人报告、仓位或用户身份。"""
    from app.db.session import get_engine
    from app.repositories.prediction_store import rows

    with get_engine().connect() as c:
        result = rows(
            c,
            """SELECT p.prediction_id,p.horizon_id,
          (SELECT jsonb_agg(o.payload ORDER BY o.checked_at) FROM prediction_outcome o
             WHERE o.prediction_id=p.prediction_id) outcomes,
          (SELECT s.payload FROM prediction_check_state s WHERE s.prediction_id=p.prediction_id) check_state
          FROM fund_prediction_record p WHERE p.fund_code=:code AND p.prediction_id=ANY(:ids)""",
            code=fund_code,
            ids=request.predictionIds,
        )
    return {str(r["prediction_id"]): r for r in result}


@router.get("/funds/{fund_code}/history")
def read_history(
    fund_code: FundCode,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    before: datetime | None = None,
    beforeId: UUID | None = None,
):
    return {"items": history(fund_code, limit=limit, before=before, before_id=beforeId)}


@router.get("/funds/{fund_code}/replay-inputs")
def replay_input(fund_code: FundCode, start: date, end: date):
    from app.services.prediction_replay_inputs import replay_inputs

    try:
        return replay_inputs(fund_code, start, end)
    except PredictionFailure as error:
        raise HTTPException(422, error.payload) from error


@router.get("/models")
def models():
    return model_status()


@router.post("/maintenance")
def maintenance():
    return {
        "recoveredTasks": recover_tasks(),
        "recoveredResearch": recover_research(),
        "resolvedTargets": resolve_pending_targets(),
        "outcomes": verify_outcomes(),
    }


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fundCodes: list[Annotated[str, Field(pattern=r"^[0-9]{6}$")]] = Field(min_length=1, max_length=50)
    horizonIds: list[str] = Field(min_length=1, max_length=3)
    trainStart: str
    trainEnd: str
    validationEnd: str
    selectionEnd: str
    stride: int = Field(default=5, ge=5, le=20)
    evidenceLevel: str = "DEVELOPMENT_ONLY"


@router.post("/research", status_code=202)
def start_research(request: ResearchRequest):
    try:
        return create_research(request.model_dump())
    except PredictionFailure as error:
        raise HTTPException(422, error.payload) from error


@router.get("/research/{run_id}")
def read_research(run_id: UUID):
    return research_status(run_id)


@router.post("/research/{run_id}/cancel")
def cancel_research(run_id: UUID):
    from sqlalchemy import text

    from app.db.session import get_engine

    with get_engine().begin() as c:
        c.execute(text("UPDATE prediction_research_run SET cancel_requested=true WHERE run_id=:id"), {"id": run_id})
    return research_status(run_id)


@router.post("/research/{run_id}/resume")
def resume_research(run_id: UUID):
    from app.services.prediction_research import _executor, run_research

    value = research_status(run_id)
    if value["status"] not in {"FAILED", "INTERRUPTED", "CANCELLED"}:
        raise HTTPException(409, "只有已中断或失败的研究可以恢复")
    from sqlalchemy import text

    from app.db.session import get_engine

    with get_engine().begin() as c:
        c.execute(
            text("UPDATE prediction_research_run SET cancel_requested=false,status='QUEUED' WHERE run_id=:id"),
            {"id": run_id},
        )
    _executor.submit(run_research, run_id)
    return research_status(run_id)
