"""全部关注公共预测流水线；费用阶段先于此阶段，Java接回个人建议，最后核验到期。"""

import re
import time
from datetime import date

from app.services.direction_1d_sync import Direction1dSyncResult, Direction1dSyncService
from app.services.prediction_generation import create_task, task_status, verify_outcomes


class MultiPredictionSyncService(Direction1dSyncService):
    def sync(self, *, progress_reporter):
        after, codes = "", []
        while True:
            response = self._client.get("/internal/v1/direction-1d/sync/fund-codes", params={"after": after})
            response.raise_for_status()
            page = response.json()
            if (
                not isinstance(page, list)
                or len(page) > 100
                or page != sorted(set(page))
                or any(
                    not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code) or code <= after for code in page
                )
            ):
                raise ValueError("MULTI_SCOPE_INVALID")
            if not page:
                break
            codes.extend(page)
            after = page[-1]
        total, created, reused, issues = 0, 0, 0, []
        # 公共批次最多500只，基金范围分页，跨用户关注去重；统计单位明确为基金周期项。
        for start in range(0, len(codes), 100):
            task = create_task(codes[start : start + 100])
            deadline = time.monotonic() + 600
            while task["status"] in {"QUEUED", "RUNNING", "INTERRUPTED"}:
                progress_reporter(
                    total + task["plannedItems"] - task["pendingItems"],
                    len(codes) * 3,
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
        verify_outcomes()
        return Direction1dSyncResult(date.today(), total, created, reused, tuple(issues))
