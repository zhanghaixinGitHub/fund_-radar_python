"""多周期内部API：只能由Java服务调用，LIVE生成时间及模型路由均由服务器决定。"""

from datetime import date, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.routing import APIRoute
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


class PredictionRoute(APIRoute):
    """领域错误保留原因码与重试边界，不能变成无上下文500。"""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def checked(request):
            try:
                return await original(request)
            except PredictionFailure as error:
                code = error.payload["code"]
                status = 404 if code.endswith("NOT_FOUND") else 409 if "LEASE" in code or "CONFLICT" in code else 422
                raise HTTPException(status, error.payload) from error

        return checked


router = APIRouter(dependencies=[Depends(require_service_token)], route_class=PredictionRoute)
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


@router.get("/funds/{fund_code}/ledger-inputs")
def ledger_input(fund_code: FundCode, start: date, end: date):
    from app.services.prediction_replay_inputs import replay_inputs

    return replay_inputs(fund_code, start, end, frozen_bundles=[])


@router.get("/auto/policy")
def automatic_policy():
    from app.services.auto_model_contract import auto_policy

    return auto_policy()


@router.post("/maintenance")
def maintenance():
    from app.services.auto_model_selection import recover_auto

    return {
        "recoveredAuto": recover_auto(),
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


class AutoCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fundCodes: list[Annotated[str, Field(pattern=r"^[0-9]{6}$")]] = Field(max_length=10000)
    source: str = Field(default="MAINTENANCE", pattern="^(MAINTENANCE|SYNC_COMPLETED)$")


@router.post("/auto/check", status_code=202)
def auto_check(request: AutoCheckRequest):
    from app.services.auto_model_selection import check_auto

    return check_auto(request.fundCodes, request.source)


@router.post("/auto/summary")
def auto_summary(request: AutoCheckRequest):
    from app.services.auto_model_summary import operating_summary

    return operating_summary(request.fundCodes)


@router.post("/auto/effects")
def auto_effects(request: AutoCheckRequest):
    from app.services.auto_model_summary import effect_summary

    return effect_summary(request.fundCodes)


@router.get("/auto/cycles")
def auto_cycles(
    limit: Annotated[int, Query(ge=1, le=50)] = 20, before: datetime | None = None, beforeId: UUID | None = None
):
    from app.services.auto_model_summary import technical_cycles

    return technical_cycles(limit, before, beforeId)


@router.get("/auto/cycles/{cycle_id}")
def auto_detail(cycle_id: UUID):
    from app.services.auto_model_store import cycle_state

    value = cycle_state(cycle_id)
    # 技术详情也不整批返回每日frames，按单基金回放接口读取。
    return {key: value[key] for key in ("cycle_id", "status", "spec", "checkpoint", "result", "error", "updated_at")}


@router.post("/auto/replay/claim")
def auto_claim():
    from app.services.auto_model_selection import claim_replay

    return claim_replay()


@router.post("/auto/cycles/{cycle_id}/cancel")
def auto_cancel(cycle_id: UUID):
    from sqlalchemy import text

    from app.db.session import get_engine
    from app.services.auto_model_store import cycle_state, event

    with get_engine().begin() as c:
        changed = c.execute(
            text("""UPDATE prediction_auto_cycle SET cancel_requested=true,status='CANCELLED',
          updated_at=clock_timestamp(),finished_at=clock_timestamp() WHERE cycle_id=:id
          AND status NOT IN ('COMPLETED','VERIFYING_ADOPTION') RETURNING cycle_id"""),
            {"id": cycle_id},
        ).scalar()
        if changed:
            event(c, cycle_id, "CANCELLED", {"source": "ADMIN"})
    return cycle_state(cycle_id)


@router.post("/auto/cycles/{cycle_id}/resume")
def auto_resume(cycle_id: UUID):
    from sqlalchemy import text

    from app.db.session import get_engine
    from app.services.auto_model_selection import recover_auto
    from app.services.auto_model_store import cycle_state

    with get_engine().begin() as c:
        c.execute(
            text("""UPDATE prediction_auto_cycle SET cancel_requested=false,
          status=CASE WHEN checkpoint?'bundles' THEN 'REPLAYING' ELSE 'INTERRUPTED' END,
          lease_until=NULL,lease_owner=NULL,next_attempt_at=NULL,updated_at=clock_timestamp()
          WHERE cycle_id=:id AND status IN ('CANCELLED','FAILED','INTERRUPTED')"""),
            {"id": cycle_id},
        )
    recover_auto()
    return cycle_state(cycle_id)


@router.get("/auto/cycles/{cycle_id}/replay/{fund_code}")
def auto_replay_input(cycle_id: UUID, fund_code: FundCode, owner: UUID):
    from app.services.auto_model_selection import replay_input

    return replay_input(cycle_id, owner, fund_code)


class ReplayResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    leaseOwner: UUID
    result: dict


@router.post("/auto/cycles/{cycle_id}/replay/{fund_code}")
def auto_replay_result(cycle_id: UUID, fund_code: FundCode, request: ReplayResultRequest):
    from app.services.auto_model_selection import save_replay

    return save_replay(cycle_id, request.leaseOwner, fund_code, request.result)


class AdoptionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictionIds: list[UUID] = Field(min_length=3, max_length=100)
    apiReadback: bool
    adviceReferenced: bool
    referenceHash: str = Field(pattern=r"^[a-f0-9]{64}$")


@router.post("/auto/releases/{release_id}/receipt")
def auto_receipt(release_id: UUID, receipt: AdoptionReceipt):
    from app.services.auto_model_store import actual_use_receipt

    return actual_use_receipt(release_id, receipt.model_dump())


@router.get("/auto/releases/pending")
def pending_adoption():
    from app.db.session import get_engine
    from app.repositories.prediction_store import rows

    with get_engine().connect() as c:
        return rows(
            c,
            """SELECT r.release_id FROM prediction_model_release r
          JOIN prediction_auto_cycle a ON a.cycle_id=r.cycle_id WHERE a.status='VERIFYING_ADOPTION'
          ORDER BY r.created_at LIMIT 100""",
        )


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
