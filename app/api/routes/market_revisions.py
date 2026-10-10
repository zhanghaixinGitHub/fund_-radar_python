"""服务间只读批量版本查询；POST 仅承载有界查询条件，不创建任何作业。"""

from fastapi import APIRouter, Depends

from app.api.dependencies import require_service_token
from app.services.market_revisions import RevisionRequest, read_revisions

router = APIRouter(dependencies=[Depends(require_service_token)])


@router.post("")
def revisions(request: RevisionRequest):
    return read_revisions(request)
