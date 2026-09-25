"""已采集基金资料的内部只读契约，浏览器经 Java 用户鉴权后访问。"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.api.dependencies import require_service_token
from app.services import fund_materials
from app.services.fund_catalog_read import get_fund

router = APIRouter(dependencies=[Depends(require_service_token)])
Code = Annotated[str, Path(pattern=r"^\d{6}$")]


def require_fund(code: str):
    if get_fund(code) is None:
        raise HTTPException(404, "基金不存在。")


@router.get("/{fund_code}/materials")
def get_materials(
    fund_code: Code,
    report_id: Annotated[str | None, Query(alias="reportId", pattern=r"^[a-f0-9]{64}$")] = None,
    stock_code: Annotated[str | None, Query(alias="stockCode", pattern=r"^\d{6}\.(SH|SZ|BJ)$")] = None,
):
    require_fund(fund_code)
    try:
        return fund_materials.overview(fund_code, report_id, stock_code)
    except ValueError as exc:
        raise HTTPException(422, "所选资料暂不可用。") from exc


@router.get("/{fund_code}/materials/documents")
def get_documents(
    fund_code: Code,
    page: Annotated[int, Query(ge=1, le=10000)] = 1,
    page_size: Annotated[int, Query(alias="pageSize", ge=1, le=50)] = 20,
    kind: Literal["all", "fund", "company", "news"] = "all",
    stock_code: Annotated[str | None, Query(alias="stockCode", pattern=r"^\d{6}\.(SH|SZ|BJ)$")] = None,
    keyword: Annotated[str, Query(max_length=80)] = "",
    latest_only: Annotated[bool, Query(alias="latestOnly")] = False,
):
    require_fund(fund_code)
    return fund_materials.documents(
        fund_code,
        page=page,
        size=page_size,
        kind=kind,
        stock_code=stock_code,
        keyword=keyword.strip(),
        latest_only=latest_only,
    )
