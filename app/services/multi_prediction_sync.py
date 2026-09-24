"""全部关注四周期预测；一日沿用独立窗口和留档，多周期接回个人建议后核验到期。"""

import logging
import time
from datetime import date

from app.services.direction_1d_sync import Direction1dSyncResult, Direction1dSyncService
from app.services.prediction_contract import prediction_policy
from app.services.prediction_generation import create_task, task_status, verify_outcomes

logger = logging.getLogger(__name__)


class MultiPredictionSyncService(Direction1dSyncService):
    def sync(self, *, progress_reporter):
        """优先尝试有截止时间的一日预测；窗口外或该阶段异常不阻断其他周期。"""
        codes = self.read_fund_codes()
        planned = len(codes) * (1 + len(prediction_policy()["horizons"]))
        total, created, reused, issues = 0, 0, 0, []
        progress_reporter(0, planned, None, f"{len(codes)}只基金，正在检查一日、五日、二十日和半年预测")
        if codes:
            try:
                daily = self.sync_codes(
                    codes,
                    target=self.read_target_date(),
                    progress_reporter=lambda current, _total, code, message: progress_reporter(
                        current,
                        planned,
                        code,
                        f"一日预测：{message}",
                    ),
                )
                created, reused = daily.created, daily.existing
                issues.extend(f"一日预测/{issue}" for issue in daily.issues)
            except Exception:
                # 例如一日窗口服务不可用；不能把未确认结果记为成功，也不能阻塞其他周期。
                logger.exception("multi_prediction_sync.sync >>> 一日预测阶段未完成，fund_count=%s", len(codes))
                issues.extend(f"一日预测/{code}：一日预测检查未完成，请稍后重试" for code in codes)
            total = len(codes)
            progress_reporter(total, planned, None, "一日预测检查结束，继续检查五日、二十日和半年预测")
        # 公共批次最多500只，基金范围分页，跨用户关注去重；统计单位明确为基金周期项。
        for start in range(0, len(codes), 100):
            task = create_task(codes[start : start + 100])
            deadline = time.monotonic() + 600
            while task["status"] in {"QUEUED", "RUNNING", "INTERRUPTED"}:
                progress_reporter(
                    total + task["plannedItems"] - task["pendingItems"],
                    planned,
                    None,
                    f"{len(codes)}只基金；当前持久预测任务 {task['taskId']}，待处理{task['pendingItems']}项",
                )
                if time.monotonic() > deadline:
                    raise TimeoutError("MULTI_TASK_WAIT_TIMEOUT: " + task["taskId"])
                time.sleep(0.5)
                task = task_status(task["taskId"])
            total += task["plannedItems"]
            created += task["createdItems"]
            reused += task["reusedItems"]
            issues.extend(
                f"{item['fundCode']}/{item['horizonId']}：{item['result']['error']['summary']}"
                for item in task["items"]
                if item["status"] == "FAILED"
            )
            # 公共原文已提交后再生成私人建议；Java不接收Python传来的用户编号或金额。
            response = self._client.post(
                "/internal/v1/multi-predictions/sync/finalize", json={"taskId": task["taskId"]}
            )
            response.raise_for_status()
            if response.json().get("failed", 0):
                raise RuntimeError("DECISION_ARCHIVE_INCOMPLETE: " + task["taskId"])
            progress_reporter(total, planned, None, f"已检查 {total}/{planned} 个基金周期项")
        verify_outcomes()
        return Direction1dSyncResult(date.today(), total, created, reused, tuple(issues))
