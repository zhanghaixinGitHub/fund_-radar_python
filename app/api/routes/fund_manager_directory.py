"""基金经理内部API：读接口不采集，更新由管理员通过Java提交到原任务中心。"""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.exc import SQLAlchemyError

from app.api.dependencies import require_service_token
from app.api.routes.funds import _to_internal_sync_job_status
from app.core.logging import get_logger
from app.schemas.fund import InternalSyncJobStatus
from app.schemas.fund_manager_directory import (
    ManagerDetail,
    ManagerFund,
    ManagerFundPage,
    ManagerPage,
    ManagerSyncRequest,
)
from app.services import fund_manager_directory_read as read
from app.services.sync_jobs import SyncJobInProgressError, get_sync_job_manager

router = APIRouter(dependencies=[Depends(require_service_token)])
logger = get_logger(__name__)


def execute(reader, *args, **kwargs):
    try:
        return reader(*args, **kwargs)
    except read.ManagerSnapshotExpired as error:
        raise HTTPException(409, {"code": "SNAPSHOT_EXPIRED", "message": str(error)}) from None
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except (SQLAlchemyError, OSError):
        logger.exception("fund_manager_directory.read   >>> manager data unavailable")
        raise HTTPException(503, "基金经理资料暂不可用，请稍后重试。") from None


@router.get("", response_model=ManagerPage)
def directory(
    response: Response,
    keyword: Annotated[str, Query(max_length=128)] = "",
    company: Annotated[str, Query(max_length=128)] = "",
    fund_type: Annotated[
        str, Query(alias="fundType", pattern=r"^(BOND|STOCK|MIXED|INDEX|MONEY|QDII|FOF|OTHER)?$")
    ] = "",
    state: Literal["all", "current", "history"] = "all",
    page: Annotated[int, Query(ge=1, le=10000)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=50)] = 20,
    snapshot_id: Annotated[UUID | None, Query(alias="snapshotId")] = None,
):
    response.headers["Cache-Control"] = "private, no-store"
    return execute(read.list_managers, keyword, company, fund_type, state, page, page_size, snapshot_id)


@router.post("/sync", response_model=InternalSyncJobStatus, status_code=202)
def synchronize(body: ManagerSyncRequest):
    try:
        if body.managerIds:
            execute(read.validate_ids, body.managerIds)
        value = get_sync_job_manager().start_fund_managers(tuple(body.managerIds), str(body.requestId))
        return _to_internal_sync_job_status(value)
    except SyncJobInProgressError:
        raise HTTPException(409, "已有同步任务正在执行，请稍后重试。") from None
    except LookupError as error:
        raise HTTPException(404, str(error)) from None
    except ValueError as error:
        raise HTTPException(409, str(error)) from None


@router.get("/sync/latest", response_model=InternalSyncJobStatus | None)
def latest():
    value = get_sync_job_manager().get_latest_job("FUND_MANAGERS")
    return _to_internal_sync_job_status(value) if value else None


@router.get("/links", response_model=dict[str, UUID])
def manager_links(fund_code: Annotated[str, Query(alias="fundCode", pattern=r"^[0-9]{6}$")]):
    return execute(read.links, fund_code)


@router.get("/{manager_id}", response_model=ManagerDetail)
def manager(manager_id: UUID, snapshot_id: Annotated[UUID | None, Query(alias="snapshotId")] = None):
    return execute(read.detail, manager_id, snapshot_id)


@router.get("/{manager_id}/funds", response_model=ManagerFundPage)
def funds(
    manager_id: UUID,
    state: Literal["all", "current", "history", "future", "pending"] = "current",
    page: Annotated[int, Query(ge=1, le=10000)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=50)] = 20,
    snapshot_id: Annotated[UUID | None, Query(alias="snapshotId")] = None,
):
    return execute(read.funds, manager_id, state, page, page_size, snapshot_id)


@router.get("/{manager_id}/profile", response_model=ManagerDetail)
def profile(manager_id: UUID):
    return execute(read.detail, manager_id)


@router.get("/{manager_id}/performance", response_model=ManagerFund)
def performance(
    manager_id: UUID,
    related_fund_id: Annotated[UUID, Query(alias="relatedFundId")],
    tenure_version_id: Annotated[UUID, Query(alias="tenureVersionId")],
    snapshot_id: Annotated[UUID | None, Query(alias="snapshotId")] = None,
):
    detail = execute(read.detail, manager_id, snapshot_id)
    selected = next(
        (
            row
            for row in detail.funds
            if row.relatedFundId == related_fund_id and row.tenureVersionId == tenure_version_id
        ),
        None,
    )
    if selected is None:
        raise HTTPException(404, "所选任期不属于该经理或该基金。")
    return selected


@router.get("/{manager_id}/disclosures")
def disclosures(manager_id: UUID):
    return execute(read.detail, manager_id).disclosures


@router.get("/{manager_id}/funds/{related_fund_id}")
def related_fund(
    manager_id: UUID, related_fund_id: UUID, snapshot_id: Annotated[UUID | None, Query(alias="snapshotId")] = None
):
    return execute(read.related_fund, manager_id, related_fund_id, snapshot_id)
