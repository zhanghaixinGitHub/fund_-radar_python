"""每天关注预测的持久续办器；同步、审阅、保存和本人读回各有独立完成条件。

本模块不计算方向，也不调用付费模型。聊天提供绑定真实输入的审阅后，才沿用
正式导入方法。运行目录只绑定一个目标窗口；没有外部证据的行动不能自动关闭。
"""

import hashlib
from pathlib import Path

import httpx

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save

TERMINAL = {"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED"}


class DailyPredictionRun:
    """用可替换业务适配器执行同一份台账，方便测试实际状态转换和异常边界。"""

    def __init__(self, directory: Path, backend):
        self.directory = directory.resolve()
        self.backend = backend
        self.path = self.directory / "run.json"

    def advance(self, *, reviews=None, blocked_file=None):
        """推进一次有界工作；未完成明确返回下一步，重试不重交同步或覆盖旧输入。"""
        b, root = self.backend, self.directory
        at, owner, scope, current_window = b.context()
        state = (
            read(self.path)
            if self.path.exists()
            else {
                "owner_user_id": owner,
                "window": current_window,
                "created_at": at,
                "funds": {},
                "batch": None,
                "first_failures": [],
            }
        )
        if state["owner_user_id"] != owner:
            raise ValueError("DAILY_OWNER_BINDING_CHANGED")
        if blocked_file:
            blocked_file = Path(blocked_file).resolve()
            state["blocked_file"] = {
                "path": str(blocked_file),
                "sha256": hashlib.sha256(blocked_file.read_bytes()).hexdigest(),
            }
        elif state.get("blocked_file"):
            blocked_file = Path(state["blocked_file"]["path"])
            if hashlib.sha256(blocked_file.read_bytes()).hexdigest() != state["blocked_file"]["sha256"]:
                raise ValueError("EXTERNAL_BLOCK_RECEIPTS_CHANGED")
        if state["window"]["target_nav_date"] != current_window["target_nav_date"]:
            # 已完成结果可以读回；未完成运行必须保留错过窗口，绝不能偷换为下一期。
            if state.get("status") != "CLOSED":
                state["status"] = "MISSED_DEADLINE"
                versioned_save(self.path, state)
                return state
            return state
        state.update(checked_at=at, scope=scope, scope_hash=digest(scope), status="CHECKING")
        versioned_save(self.path, state)
        # 首次提交资料前仍须建立本人完成台账；等待同一作业时不重复展开全部原文。
        # 同步尚未终态时资料还会变化，重型守卫既不能据此生成结论，也会与采集争用内存。
        initial_guard = state["batch"] is None or not (root / "followthrough.json").exists()
        if initial_guard:
            b.guard(root / "followthrough.json", blocked_file)
        if state["batch"] is None:
            # 提交意图先落盘。超时或进程中断后只恢复原作业，不盲目重提。
            intent = root / "submission-intent.json"
            if intent.exists():
                batch = b.recover_submission(read(intent))
            else:
                previous = b.latest_batch()
                versioned_save(intent, {"at": at, "previous": previous, "materialsOnly": True})
                batch = b.start_or_reuse(previous)
            state["batch"] = batch
            versioned_save(self.path, state)
        terminal = b.batch_status(state["batch"])
        if terminal.get("existing_job_finished"):
            # 已运行的旧非纯资料/单项作业先等待；终态后另起本轮纯资料，绝不当成已同步。
            predecessor = state["batch"]
            state.setdefault("preceding_jobs", []).append(predecessor)
            intent = root / ("submission-after-" + predecessor["job_id"] + ".json")
            if intent.exists():
                state["batch"] = b.recover_submission(read(intent))
            else:
                previous = b.latest_batch()
                versioned_save(intent, {"at": at, "previous": previous, "materialsOnly": True})
                state["batch"] = b.start_or_reuse(previous)
            versioned_save(self.path, state)
            terminal = b.batch_status(state["batch"])
        versioned_save(root / "materials-status.json", terminal)
        if not terminal["terminal"]:
            state["status"] = "WAITING_MATERIALS"
            versioned_save(self.path, state)
            return state
        if not initial_guard:
            # 原作业及全部子任务终态后重新检查，绝不复用等待期间的旧资料就绪状态。
            b.guard(root / "followthrough.json", blocked_file)
        # outcomes 的缺失是明确失败，不能以父任务部分成功或默认零条替代。
        state["prediction_outcomes"] = terminal.get("prediction_outcomes")
        state["prediction_outcomes_recovery"] = terminal.get("prediction_outcomes_recovery")
        state["materials_terminal"] = terminal
        for code in scope:
            state["current_fund_code"] = code
            item = state["funds"].setdefault(code, {"fund_code": code})
            item.pop("failure", None)
            try:
                # 已保存仍须读回本人范围；本次有新增资料时会由来源摘要重新触发审阅。
                item.update(b.recover_materials(code, root / "funds" / code, state["window"]))
                review = (reviews or {}).get(code)
                if review:
                    if (
                        review.get("fund_code") != code
                        or review.get("target_nav_date") != state["window"]["target_nav_date"]
                    ):
                        raise ValueError("DAILY_REVIEW_IDENTITY_OR_TARGET_MISMATCH")
                    b.approve(code, review)
                periods = b.periods(code, state["window"])
                item["periods"] = []
                for period in periods:
                    horizon = period["horizon"]
                    folder = root / "funds" / code / horizon
                    if period["status"] != "READY":
                        item["periods"].append(period)
                        continue
                    if b.read_back(code, horizon, state["window"]):
                        item["periods"].append({**period, "status": "SAVED_AND_READ_BACK"})
                        continue
                    # 失败影响本周期，其他周期和其他基金仍继续执行。
                    try:
                        packet = b.freeze(code, horizon, folder)
                        folder = Path(packet.get("directory", folder))
                        review_path = folder / "review.json"
                        if not review_path.exists():
                            item["periods"].append(
                                {**period, "status": "READ_AND_JUDGE", "input": packet, "review_path": str(review_path)}
                            )
                            continue
                        receipt = b.import_and_archive(code, horizon, folder, owner, state["window"])
                        if not b.read_back(code, horizon, state["window"]):
                            raise ValueError("OWNER_ARCHIVE_READBACK_MISSING")
                        item["periods"].append({**period, "status": "SAVED_AND_READ_BACK", "receipt": receipt})
                    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as error:
                        # 归档接口的一次拒绝或超时只阻塞本周期；其余周期仍可
                        # 使用各自已冻结、已人工审阅的输入继续，不丢弃首个失败。
                        failure = {
                            "fund_code": code,
                            "horizon": horizon,
                            "type": type(error).__name__,
                            "reason": str(error)[:800],
                        }
                        if failure not in state["first_failures"]:
                            state["first_failures"].append(failure)
                        item["periods"].append({**period, "status": "ACTION_REQUIRED", "failure": str(error)})
            except Exception as error:
                # 保留首错；不能把代码/解析故障标为外部来源不存在。
                failure = {"fund_code": code, "type": type(error).__name__, "reason": str(error)[:800]}
                if failure not in state["first_failures"]:
                    state["first_failures"].append(failure)
                item.update(status="ACTION_REQUIRED", failure=failure)
            versioned_save(self.path, state)
        final = b.guard(root / "followthrough.json", blocked_file)
        if hasattr(b, "ratings"):
            try:
                state["ratings"] = b.ratings(terminal)
            except Exception as error:
                state["ratings"] = {"status": "FAILED", "reason": type(error).__name__ + ":" + str(error)[:300]}
            versioned_save(root / "ratings-status.json", state["ratings"])
        versioned_save(
            root / "work-items.json",
            {"owner_user_id": owner, "window": state["window"], "funds": final["funds"], "execution": state["funds"]},
        )
        # 守卫是唯一结束判据。同步摘要不完整时仍不能报告整轮完成。
        state.update(
            status=final["status"],
            all_predictions_generated=final["all_predictions_generated"],
            pending_funds=final["pending_funds"],
            current_fund_code=None,
        )
        if state["prediction_outcomes"] is None and state["status"] == "CLOSED":
            state["status"] = "OUTCOME_STEP_MISSING"
        if state.get("ratings", {}).get("status") != "SUCCEEDED" and "ratings" in state and state["status"] == "CLOSED":
            state["status"] = "RATINGS_PENDING"
        versioned_save(self.path, state)
        return state
