"""四周期同步：批次独立收尾，等待超时不取消后台任务，旧成果按日期展示。"""

import logging
import time
from datetime import date

from app.services.direction_1d_sync import Direction1dSyncResult, Direction1dSyncService
from app.services.prediction_contract import prediction_policy
from app.services.prediction_generation import create_task, task_status, verify_outcomes

logger = logging.getLogger(__name__)
WAIT_SECONDS = 600
WAITING_CODES = {
    "NAV_CURRENT_NOT_READY",
    "NAV_LATEST_NOT_READY",
    "NAV_GAP",
    "NAV_HISTORY_SHORTAGE",
    "NAV_BASE_MISSING",
    "DATA_INSUFFICIENT",
    "WINDOW_CHANGED",
    "BASE_NAV_MISMATCH",
}


class MultiPredictionSyncService(Direction1dSyncService):
    def sync(self, *, progress_reporter):
        """每个基金周期只有一个状态；归档/核验问题单独记录，不重复计算失败基金数。"""
        codes = self.read_fund_codes()
        horizons = [h["horizon_id"] for h in prediction_policy()["horizons"]]
        planned = len(codes) * (1 + len(horizons))
        total, created, reused = 0, 0, 0
        issues, unsupported, items, followups = [], [], [], []
        target = date.today()  # 旧接口兼容日期；逐项目标日未知时返回 null，不能伪造预测日期。
        progress_reporter(0, planned, None, f"{len(codes)}只基金，正在检查四个周期的预测")
        if codes:
            try:
                target = self.read_target_date()
                daily = self.sync_codes(
                    codes,
                    target=target,
                    progress_reporter=lambda current, _total, code, message: progress_reporter(
                        current,
                        planned,
                        code,
                        f"一日预测：{message}",
                    ),
                )
                created, reused = daily.created, daily.existing
                issues.extend(f"一日预测/{issue}" for issue in daily.issues)
                unsupported.extend(f"一日预测/{item}" for item in daily.unsupported)
                items.extend(daily.items)
            except Exception:
                logger.exception("multi_prediction_sync.sync >>> 一日预测检查未完成，fund_count=%s", len(codes))
                issues.extend(f"一日预测/{code}：一日预测检查未完成，请稍后重试" for code in codes)
                items.extend({"fundCode": c, "horizonId": "T1", "targetDate": None, "state": "ERROR"} for c in codes)
            total = len(codes)
            progress_reporter(total, planned, None, "一日预测检查结束，继续检查其他周期")
        for start in range(0, len(codes), 100):
            batch = codes[start : start + 100]
            task, wait_failed, blocked = None, False, False
            blocked_pairs = set()
            try:
                task = create_task(batch)
                blocked = task.get("blockedByPreviousTask", False)
                blocked_pairs = {(i["fundCode"], i["horizonId"]) for i in task.get("blockedItems", [])}
                deadline = time.monotonic() + WAIT_SECONDS
                while not blocked and task["status"] in {"QUEUED", "RUNNING", "INTERRUPTED"}:
                    progress_reporter(
                        total + task["plannedItems"] - task["pendingItems"],
                        planned,
                        None,
                        f"当前批次还有 {task['pendingItems']} 项预测待完成",
                    )
                    if time.monotonic() >= deadline:
                        raise TimeoutError("MULTI_TASK_WAIT_TIMEOUT")
                    time.sleep(0.5)
                    task = task_status(task["taskId"])
            except Exception:
                wait_failed = task is not None
                logger.exception("multi_prediction_sync.sync >>> 批次创建或等候未完成，funds=%s", batch)
            indexed = {} if blocked else {(i["fundCode"], i["horizonId"]): i for i in (task or {}).get("items", [])}
            for code in batch:
                for horizon in horizons:
                    item_blocked = (code, horizon) in blocked_pairs or blocked
                    item = {} if item_blocked else indexed.get((code, horizon), {})
                    result = item.get("result") or {}
                    error = result.get("error") or {}
                    state, reason = "ERROR", error.get("summary") or "预测未确认完成"
                    if item.get("status") in {"CREATED", "REUSED"}:
                        state = "COMPLETED"
                        created += int(item["status"] == "CREATED")
                        reused += int(item["status"] == "REUSED")
                    elif error.get("code") == "CALENDAR_POLICY_MISSING":
                        state = "UNSUPPORTED"
                        unsupported.append(f"{code}/{horizon}：{reason}")
                    else:
                        if (
                            item_blocked
                            or error.get("code") in WAITING_CODES
                            or (wait_failed and item.get("status") not in {"FAILED", "CANCELLED"})
                        ):
                            state = "WAITING"
                            if item_blocked:
                                reason = "上一期预测尚未结束，本期等待原任务完成后重试"
                            elif wait_failed:
                                reason = "预测仍待后台确认，重试会先检查原任务"
                        issues.append(f"{code}/{horizon}：{reason}")
                    items.append(
                        {
                            "fundCode": code,
                            "horizonId": horizon,
                            "state": state,
                            "targetDate": result.get("startDate"),
                            "reason": reason if state != "COMPLETED" else None,
                        }
                    )
            total += len(batch) * len(horizons)
            # 公共任务持有自己的执行锁；等候者超时不能假取消、释放其锁或归档半成品。
            if task and task.get("taskId") and not blocked and not wait_failed and not task.get("pendingItems"):
                try:
                    response = self._client.post(
                        "/internal/v1/multi-predictions/sync/finalize",
                        json={"taskId": task["taskId"]},
                    )
                    response.raise_for_status()
                    if response.json().get("failed", 0):
                        raise RuntimeError("DECISION_ARCHIVE_INCOMPLETE")
                except Exception:
                    followups.append(f"{batch[0]}—{batch[-1]}：预测已保存，部分关注关联或建议尚未完成")
                    logger.exception("multi_prediction_sync.sync >>> 归档未完成 task=%s", task["taskId"])
            progress_reporter(total, planned, None, f"已检查 {total}/{planned} 个基金周期项")
        try:
            verify_outcomes()
        except Exception:
            followups.append("到期结果核验暂未完成，已保存预测保留")
            logger.exception("multi_prediction_sync.sync >>> 到期核验未完成")
        saved = ()
        try:
            response = self._client.get("/internal/v1/multi-predictions/sync/saved-results")
            response.raise_for_status()
            saved = tuple(response.json())
        except Exception:
            logger.warning("multi_prediction_sync.sync >>> 已保存日期摘要暂不可读")
        return Direction1dSyncResult(
            target, total, created, reused, tuple(issues), tuple(unsupported), tuple(items), tuple(followups), saved
        )
