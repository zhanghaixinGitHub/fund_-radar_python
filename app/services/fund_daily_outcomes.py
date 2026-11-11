"""本人当前模拟持仓的一日到期核对；有界分页、实际读回，不开启旧训练和定时器。"""

from app.services.fund_exposure_common import now
from app.services.fund_materials_store import versioned_save

COUNTS = {"total", "checked", "completed", "created", "pending", "failed", "remaining"}


def reconcile(backend, *, all_archived=False):
    from app.commands.fund_prediction_followthrough import owner_scope

    due = str(now().date())
    base = backend.settings.core_service_base_url + f"/internal/v1/direction-1d/sync/users/{backend.owner}/outcomes"
    if all_archived:
        base += "/archive"
    get_scope = (
        (lambda: backend.get(base + "/fund-codes"))
        if all_archived
        else lambda: owner_scope(backend.settings, backend.owner)
    )
    scope = get_scope()
    receipt_name = "one-day-archive" if all_archived else "one-day"
    summary = dict.fromkeys(COUNTS, 0)
    items, cursors = [], set()
    after = None
    for batch in range(100):
        response = backend.client.post(
            base + "/reconcile", json={"dueBefore": due, **({"after": after} if after else {})}, timeout=180
        )
        response.raise_for_status()
        page = response.json()
        counts = [page[k] for k in ("checked", "completed", "created", "pending", "failed")]
        if (
            page.get("dueBefore") != due
            or any(type(n) is not int or n < 0 for n in counts)
            or page["checked"] != page["completed"] + page["pending"] + page["failed"]
            or page["created"] > page["completed"]
            or len(page["items"]) != page["checked"]
            or any(r["fundCode"] not in scope for r in page["items"])
        ):
            raise ValueError("ONE_DAY_OUTCOME_PAGE_INVALID")
        versioned_save(backend.directory / f"{receipt_name}-outcome-batches" / f"{due}-{batch:03d}.json", page)
        for key in ("checked", "completed", "created", "pending", "failed"):
            summary[key] += page[key]
        items.extend(page["items"])
        after = page.get("nextCursor")
        if not after:
            break
        if after in cursors:
            raise ValueError("ONE_DAY_OUTCOME_CURSOR_REPEATED")
        cursors.add(after)
    else:
        raise ValueError("ONE_DAY_OUTCOME_BATCH_BUDGET_REACHED")
    live = backend.get(base + "/status?dueBefore=" + due)
    if scope != get_scope():
        raise ValueError("ONE_DAY_OUTCOME_HOLDINGS_CHANGED")
    summary.update(total=live["total"], remaining=live["remaining"])
    # 读回后仍缺结果、漏页或保存回执不一致都不能报成功。
    complete = (
        summary["checked"] == summary["total"]
        and not any(summary[k] for k in ("pending", "failed", "remaining"))
        and live["completed"] == summary["completed"]
        and not live["pending"]
        and not live["failed"]
    )
    value = {
        "status": "SUCCEEDED" if complete else "PARTIAL_SUCCESS",
        "owner_user_id": backend.owner,
        "scope": scope,
        "scope_kind": "OWNER_ARCHIVE" if all_archived else "SIMULATED_HOLDINGS",
        "dueBefore": due,
        "checked_at": now().isoformat(),
        "summary": summary,
        "readback": live,
        "items": items,
    }
    versioned_save(backend.directory / f"{receipt_name}-outcomes.json", value)
    return value
