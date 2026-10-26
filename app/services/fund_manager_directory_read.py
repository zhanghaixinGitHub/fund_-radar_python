"""公共经理资料只读服务；读取不可变快照，GET绝不触发采集或重新计算。"""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db.session import get_nav_preview_engine
from app.models.fund_manager_directory import FundManager, FundManagerCoverageSnapshot
from app.schemas.fund_manager_directory import ManagerDetail, ManagerFundPage, ManagerPage, ManagerSummary


class ManagerSnapshotExpired(ValueError):
    """客户端引用不存在或不匹配的分页快照，须重新读取第一页。"""


@contextmanager
def reading():
    """有界、可重复读的数据库查询，不修改任何资料状态。"""
    with Session(get_nav_preview_engine()) as session:
        session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        yield session


def list_managers(
    keyword: str = "",
    company: str = "",
    fund_type: str = "",
    state: str = "all",
    page: int = 1,
    page_size: int = 20,
    snapshot_id: UUID | None = None,
) -> ManagerPage:
    """目录快照最多五千人员；筛选只在已发布公共摘要中执行，不形成逐人员查询。"""
    with reading() as session:
        query = select(FundManagerCoverageSnapshot).where(FundManagerCoverageSnapshot.scope_kind == "DIRECTORY")
        if snapshot_id:
            query = query.where(FundManagerCoverageSnapshot.snapshot_id == snapshot_id)
        snapshot = session.scalars(query.order_by(FundManagerCoverageSnapshot.published_at.desc()).limit(1)).first()
        if snapshot is None:
            if snapshot_id:
                raise ManagerSnapshotExpired("所选资料版本已不可用，请重新查看列表。")
            return ManagerPage(items=[], page=page, pageSize=page_size, totalCount=0, totalPages=0)
        rows = snapshot.payload["items"]
        keyword, company = keyword.strip().casefold(), company.strip().casefold()
        filtered = [
            row
            for row in rows
            if (not keyword or keyword in (row["managerName"] + " ".join(row["companies"])).casefold())
            and (not company or any(company in value.casefold() for value in row["companies"]))
            and (not fund_type or fund_type in row.get("fundTypes", []))
            and (state != "current" or (row.get("shareCount") or 0) > 0)
            and (state != "history" or (row.get("historyCount") or 0) > 0)
        ]
        start = (page - 1) * page_size
        return ManagerPage(
            items=[ManagerSummary.model_validate(row) for row in filtered[start : start + page_size]],
            page=page,
            pageSize=page_size,
            totalCount=len(filtered),
            totalPages=(len(filtered) + page_size - 1) // page_size,
            snapshotId=snapshot.snapshot_id,
        )


def detail(manager_id: UUID, snapshot_id: UUID | None = None) -> ManagerDetail:
    """只允许读取该人员自己的快照；旧编号最多重定向一次，循环或跨人快照拒绝。"""
    with reading() as session:
        manager = session.get(FundManager, manager_id)
        if manager is None:
            raise LookupError("暂未收录该基金经理。")
        if manager.redirect_to:
            manager = session.get(FundManager, manager.redirect_to)
            if manager is None or manager.redirect_to:
                raise LookupError("经理资料待核对。")
        snapshot = (
            session.get(FundManagerCoverageSnapshot, snapshot_id or manager.current_snapshot_id)
            if (snapshot_id or manager.current_snapshot_id)
            else None
        )
        if snapshot is None or snapshot.manager_id != manager.manager_id or snapshot.scope_kind != "MANAGER":
            if snapshot_id:
                raise ManagerSnapshotExpired("资料版本已变化，请重新查看。")
            raise LookupError("经理资料尚未整理完成。")
        result = ManagerDetail.model_validate(snapshot.payload)
        # 过期只说明最后核对时间，不通过修改本机时间把旧履历伪装成新资料。
        checked = result.checkedAt
        result.stale = checked is None or datetime.now(UTC) - checked > timedelta(days=2)
        return result


def funds(
    manager_id: UUID, state: str = "current", page: int = 1, page_size: int = 20, snapshot_id: UUID | None = None
) -> ManagerFundPage:
    value = detail(manager_id, snapshot_id)
    rows = [row for row in value.funds if state == "all" or row.state.casefold() == state]
    start = (page - 1) * page_size
    return ManagerFundPage(
        items=rows[start : start + page_size],
        page=page,
        pageSize=page_size,
        totalCount=len(rows),
        totalPages=(len(rows) + page_size - 1) // page_size,
        snapshotId=value.snapshotId,
    )


def related_fund(manager_id: UUID, related_fund_id: UUID, snapshot_id: UUID | None = None):
    value = detail(manager_id, snapshot_id)
    rows = [row for row in value.funds if row.relatedFundId == related_fund_id]
    if not rows:
        raise LookupError("该经理的已收录资料中没有这只基金。")
    return {
        "manager": value.model_dump(mode="json", exclude={"funds", "disclosures"}),
        "fund": rows[0].model_dump(mode="json"),
        "tenures": [row.model_dump(mode="json") for row in rows],
    }


def links(fund_code: str) -> dict[str, UUID]:
    """单基金一次读取经理锚点；姓名只用于本基金卡片匹配，跳转始终使用稳定人员编号。"""
    with reading() as session:
        rows = session.scalars(
            select(FundManager).where(
                FundManager.anchor_key.like(f"%:{fund_code}:%"), FundManager.current_snapshot_id.is_not(None)
            )
        )
        grouped: dict[str, set[UUID]] = {}
        for manager in rows:
            grouped.setdefault(manager.display_name, set()).add(manager.redirect_to or manager.manager_id)
        return {name: next(iter(ids)) for name, ids in grouped.items() if len(ids) == 1}


def validate_ids(manager_ids: list[UUID]) -> None:
    with reading() as session:
        found = set(session.scalars(select(FundManager.manager_id).where(FundManager.manager_id.in_(manager_ids))))
        if found != set(manager_ids):
            raise LookupError("部分基金经理尚未收录。")
