"""每日新进程在原资料批次终态后恢复到期核对；保留原子任务摘要，不重提资料作业。"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import text

from app.db.session import get_engine
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save
from app.services.prediction_generation import verify_outcomes


def pending_ids(due_before, generated_before):
    """仅选择本批之前已保存且实际到期、仍等待数据的原预测，最多处理五百项。"""
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        scope = """FROM fund_prediction_record p
            JOIN prediction_check_state s USING(prediction_id)
            WHERE p.mode='LIVE' AND s.status='PENDING_DATA' AND p.generated_at<=:before
            AND coalesce(p.payload->>'endDate',p.payload->>'nominalEndDate')<:due"""
        params = {"due": due_before, "before": generated_before}
        total = c.execute(text("SELECT count(*) " + scope), params).scalar_one()
        ids = [str(x) for x in c.execute(text("SELECT p.prediction_id " + scope +
               " ORDER BY s.last_attempt_at,p.prediction_id LIMIT 500"), params).scalars()]
        return {"total": total, "ids": ids}


def recover(output: Path, native, child):
    """同批有界续查一次；没有新原预测、不改原结果、不把未加载新代码当作核对成功。"""
    if not native.get("summary", {}).get("pending"):
        return None
    if output.exists():
        previous = read(output)
        if previous.get("original_job_id") == child["job_id"]:
            return previous
    due = native["summary"].get("dueBefore") or str(
        datetime.fromisoformat(child["started_at"]).astimezone(ZoneInfo("Asia/Shanghai")).date()
    )
    scope = pending_ids(due, child["started_at"])
    selected = scope["ids"]
    result = verify_outcomes(limit=len(selected), prediction_ids=selected) if selected else {
        "checked": 0, "completed": 0, "created": 0, "pending": 0, "failed": 0,
    }
    value = {"at": now().isoformat(), "original_job_id": child["job_id"], "original_summary": native,
             "selected_prediction_ids": selected, "result": result,
             "remaining_beyond_budget": max(0, scope["total"]-len(selected)),
             "original_subtask_not_modified": True}
    versioned_save(output, value)
    return value
