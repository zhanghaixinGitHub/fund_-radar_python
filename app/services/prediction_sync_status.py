"""手动预测批次的状态归并；原预测不重写，异步作业必须按原编号核对。"""

from copy import deepcopy
from datetime import UTC, datetime

from sqlalchemy import text

from app.db.session import get_nav_preview_engine


def pending_daily(item: dict) -> bool:
    """兼容旧摘要丢失作业编号的记录；普通净值不足不触发任务查询。"""
    return (
        item.get("horizonId") == "T1"
        and item.get("state") == "WAITING"
        and (
            item.get("pending") is True
            or item.get("reasonCode") in {"QUEUED", "RUNNING", "STATUS_UNAVAILABLE"}
            or "尚未确认留档" in (item.get("reason") or "")
        )
    )


def summary(items: list[dict], previous: dict | None = None) -> dict:
    """基金周期项互斥计数；正在计算的任务不归为等待行情资料。"""
    return {
        **(previous or {}),
        "items": items,
        "counts": {
            state: sum(i.get("state") == state for i in items)
            for state in ("COMPLETED", "WAITING", "UNSUPPORTED", "ERROR")
        },
        "pendingDailyCount": sum(pending_daily(i) for i in items),
    }


def legacy_job_id(item: dict, started_at: datetime, finished_at: datetime) -> str | None:
    """旧批次只在原执行时间、基金、目标日内唯一匹配时恢复编号；有歧义绝不选最新任务。"""
    with get_nav_preview_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        rows = (
            connection.execute(
                text("""
            SELECT job_id FROM direction_1d_job WHERE kind='FORECAST'
              AND payload->>'fund_code'=:code AND payload->>'target_nav_date'=:target
              AND created_at BETWEEN :start AND :end ORDER BY created_at LIMIT 2
        """),
                {"code": item["fundCode"], "target": item["targetDate"], "start": started_at, "end": finished_at},
            )
            .scalars()
            .all()
        )
    return str(rows[0]) if len(rows) == 1 else None


def reconcile(snapshot, service, *, offset: int = 0, recover_legacy: bool = False) -> dict:
    """每次最多核对四项，网络有界；轮换偏移避免前面的慢任务阻挡后续作业。"""
    previous = snapshot.result_summary or {}
    items = deepcopy(previous.get("items", []))
    indices = [index for index, item in enumerate(items) if pending_daily(item)]
    if not indices:
        return {}
    shift = offset % len(indices)
    indices = (indices[shift:] + indices[:shift])[:4]
    created, reused = snapshot.created_count, snapshot.updated_count
    for index in indices:
        item = items[index]
        if not item.get("sourceJobId") and recover_legacy and snapshot.started_at and snapshot.finished_at:
            job_id = legacy_job_id(item, snapshot.started_at, snapshot.finished_at)
            if job_id:
                item["sourceJobId"] = job_id
        if not item.get("sourceJobId"):
            continue
        updated = service.reconcile_item(item)
        items[index] = updated
        if updated["state"] == "COMPLETED":
            created += int(not updated["reused"])
            reused += int(updated["reused"])
    combined = summary(items, previous)
    if created + reused > snapshot.created_count + snapshot.updated_count and "savedResults" in previous:
        saved = service.saved_results()
        if saved is None:
            combined.pop("savedResults", None)
        else:
            combined["savedResults"] = saved
    if combined == previous:
        return {}
    counts, pending = combined["counts"], combined["pendingDailyCount"]
    issues = [
        f"{i['fundCode']}/{i['horizonId']}：{i.get('reason') or '结果尚未确认'}"
        for i in items
        if i.get("state") in {"WAITING", "ERROR"}
    ]
    followups = combined.get("followupIssues", [])
    incomplete = issues + followups
    status = "SUCCEEDED" if not incomplete else "PARTIAL_SUCCESS" if counts["COMPLETED"] else "FAILED"
    return {
        "result_summary": combined,
        "status": status,
        "created_count": created,
        "updated_count": reused,
        "skipped_count": counts["WAITING"] + counts["ERROR"] + counts["UNSUPPORTED"],
        "progress_message": (
            f"已检查 {len(items)} 个基金周期项，新生成 {created}，已有 {reused}，暂不支持 {counts['UNSUPPORTED']}，"
            f"正在生成 {pending}，等待资料 {counts['WAITING'] - pending}，生成失败 {counts['ERROR']}"
        ),
        "error_code": "PREDICTION_INCOMPLETE" if incomplete else None,
        "error_message": "；".join(incomplete[:100]) if incomplete else None,
        "finished_at": snapshot.finished_at if pending else datetime.now(UTC),
    }
