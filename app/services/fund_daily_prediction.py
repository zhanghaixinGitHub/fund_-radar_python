"""每天关注预测的持久续办器；同步、审阅、保存和本人读回各有独立完成条件。

本模块不计算方向，也不调用付费模型。聊天提供绑定真实输入的审阅后，才沿用
正式导入方法。运行目录只绑定一个目标窗口；没有外部证据的行动不能自动关闭。
"""

import hashlib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import httpx

from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_daily_budget import assert_daily_budget
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save

TERMINAL = {"SUCCEEDED", "PARTIAL_SUCCESS", "FAILED", "CANCELLED"}
FUND_INPUT_ERRORS = (ValueError, KeyError, TypeError, RuntimeError, FileNotFoundError, httpx.HTTPError)


def supplement_run_id(state):
    """旧台账绑定实际批次，新台账绑定本次逐基金运行。"""
    return state.get("supplement_run_id") or (state.get("batch") or {}).get("job_id")


class DailyPredictionRun:
    """用可替换业务适配器执行同一份台账，方便测试实际状态转换和异常边界。"""

    def __init__(self, directory: Path, backend):
        self.directory = directory.resolve()
        self.backend = backend
        self.path = self.directory / "run.json"

    def advance(self, *, reviews=None, blocked_file=None, publish_prepared=False, resume_only=False,
                initialize_only=False):
        """推进原窗口的逐基金工作；旧台账仍可查询已登记的同步作业。"""
        b, root = self.backend, self.directory
        assert_daily_budget(root, now())
        at, owner, scope, current_window = b.context()
        if resume_only and not self.path.is_file():
            raise ValueError("DAILY_RESUME_JOB_MISSING")
        state = (
            read(self.path)
            if self.path.exists()
            else {
                "owner_user_id": owner,
                "window": current_window,
                "created_at": at,
                "funds": {},
                "supplement_run_id": str(uuid4()),
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
        if any(state["window"].get(key) != current_window.get(key)
               for key in ("base_nav_date", "calendar_version", "deadline_at")):
            state["status"] = "ORIGINAL_WINDOW_CHANGED"
            versioned_save(self.path, state)
            return state
        if not supplement_run_id(state):
            if resume_only:
                raise ValueError("DAILY_RESUME_JOB_MISSING")
            state["supplement_run_id"] = str(uuid4())
        previous_scope = state.get("scope")
        state.update(checked_at=at, scope=scope, scope_hash=digest(scope), status="CHECKING")
        checklists = hasattr(b, "save_profiles")
        strict_supplements = checklists and hasattr(b, "supplements")
        supplements = None
        if checklists:
            from app.services.fund_daily_profiles import reset_fund_checklists, save_pending

            records = b.save_profiles(scope)
            checklist_day = datetime.fromisoformat(at).astimezone(ZONE).date().isoformat()
            # 北京时间当天首次运行清空对勾；当天续办按实际核对结果逐项更新，不在清单中累积流水。
            if state.get("checklist_day") != checklist_day:
                reset_fund_checklists(root, records, state["window"], checklist_day)
                state["checklist_day"] = checklist_day
            elif previous_scope != scope and not strict_supplements:
                save_pending(root, state, records=records, aggregate=False)
        state["workflow_version"] = "HOLDINGS_END_TO_END_V1"
        if state.get("batch") and hasattr(b, "repair_handoffs"):
            state["handoff_recovery"] = b.repair_handoffs(scope)
        versioned_save(self.path, state)
        if initialize_only:
            # 总指挥先取得完整真实名单和今日清单，再逐只派工；不提前启动全持仓长命令。
            if strict_supplements:
                from app.services.fund_daily_commander import initialize_material_ledger

                initialize_material_ledger(root, state, b.supplements(state, records))
            state.update(status="READY_FOR_DISPATCH", current_fund_code=None)
            versioned_save(self.path, state)
            return state
        batch = state.get("batch")
        terminal = b.batch_status(batch) if batch else None
        if terminal and terminal.get("existing_job_finished"):
            state.setdefault("preceding_jobs", []).append(state.pop("batch"))
            state["supplement_run_id"] = str(uuid4())
            terminal = None
        direct = terminal is None
        initial_guard = not direct and not (root / "followthrough.json").exists()
        if initial_guard:
            b.guard(root / "followthrough.json", blocked_file)
        if direct:
            terminal = {"mode": "FUND_CHECKLIST", "job_id": supplement_run_id(state), "terminal": False}
            state["materials_terminal"] = terminal
            versioned_save(self.path, state)
        if not direct and not terminal["terminal"]:
            versioned_save(root / "materials-status.json", terminal)
            state.update(status="WAITING_MATERIALS", materials_terminal=terminal)
            versioned_save(self.path, state)
            return state
        if strict_supplements:
            supplements = b.supplements(state, records)
        if direct and supplements is not None:
            state.update(status="SUPPLEMENTING", current_fund_code=None)
            versioned_save(self.path, state)
            state["supplements"] = supplements.process(
                state, recover_materials=b.recover_materials, publish_prepared=publish_prepared,
            )
            terminal = {**terminal, "terminal": True, "status": state["supplements"]["status"]}
            state["materials_terminal"] = terminal
            versioned_save(self.path, state)
            if hasattr(b, "repair_handoffs"):
                state["handoff_recovery"] = b.repair_handoffs(scope)
            supplements.check_storage()
            b.guard(root / "followthrough.json", blocked_file)
        if not direct and not initial_guard:
            # 原作业及全部子任务终态后重新检查，绝不复用等待期间的旧资料就绪状态。
            b.guard(root / "followthrough.json", blocked_file)
        # outcomes 的缺失是明确失败，不能以父任务部分成功或默认零条替代。
        state["prediction_outcomes"] = terminal.get("prediction_outcomes")
        state["prediction_outcomes_recovery"] = terminal.get("prediction_outcomes_recovery")
        state["materials_terminal"] = terminal
        versioned_save(root / "materials-status.json", terminal)
        for code in scope if not (direct and supplements is not None) else ():
            assert_daily_budget(root, now())
            state["current_fund_code"] = code
            item = state["funds"].setdefault(code, {"fund_code": code})
            item.pop("failure", None)
            try:
                # 已保存仍须读回本人范围；本次有新增资料时会由来源摘要重新触发审阅。
                if not publish_prepared or not item.get("materials_path"):
                    recovered = b.recover_materials(code, root / "funds" / code, state["window"])
                    item.update(recovered)
                    if checklists:
                        if recovered.get("materials_path"):
                            item["materials_checklist_day"] = checklist_day
                        else:
                            item.pop("materials_checklist_day", None)
                        if not strict_supplements:
                            save_pending(root, state, records=records, fund_code=code, aggregate=False)
                if hasattr(b, "refresh_nav"):
                    item["base_nav_refresh"] = b.refresh_nav(code, root / "funds" / code, state["window"])
                    if checklists:
                        item["nav_checklist_day"] = checklist_day
                        if not strict_supplements:
                            save_pending(root, state, records=records, fund_code=code, aggregate=False)
            except FUND_INPUT_ERRORS as error:
                record_fund_failure(state, item, code, error)
            versioned_save(self.path, state)
        if supplements is not None and not direct:
            state.update(status="SUPPLEMENTING", current_fund_code=None)
            versioned_save(self.path, state)
            state["supplements"] = supplements.process(state)
            versioned_save(self.path, state)
            supplements.check_storage()
        # 先补本轮资料与净值，再独立核对已到期的旧预测。
        try:
            state["one_day_outcomes"] = b.one_day_outcomes()
        except FUND_INPUT_ERRORS as error:
            state["one_day_outcomes"] = {"status": "FAILED", "reason": type(error).__name__ + ":" + str(error)[:300]}
        if direct:
            try:
                state["prediction_outcomes"] = b.prediction_outcomes(scope)
            except FUND_INPUT_ERRORS as error:
                state["prediction_outcomes"] = {
                    "status": "FAILED", "reason": type(error).__name__ + ":" + str(error)[:300],
                }
        versioned_save(self.path, state)
        if hasattr(b, "prepare_monthly"):
            assert_daily_budget(root, now())
            state["monthly_compilation"] = b.prepare_monthly(scope)
            versioned_save(self.path, state)
        for code in scope:
            assert_daily_budget(root, now())
            state["current_fund_code"] = code
            item = state["funds"][code]
            if item.get("failure"):
                continue
            try:
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
                    assert_daily_budget(root, now())
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
                        review_path = Path(packet.get("review_path", folder / "review.json"))
                        if not review_path.exists():
                            item["periods"].append(
                                {
                                    **period,
                                    "status": "WAITING_ANALYSIS"
                                    if packet.get("generationStatus") == "WAITING_ANALYSIS"
                                    else "READ_AND_JUDGE",
                                    "input": packet,
                                    "review_path": str(review_path),
                                }
                            )
                            continue
                        receipt = b.import_and_archive(code, horizon, folder, owner, state["window"])
                        if not b.read_back(code, horizon, state["window"]):
                            raise ValueError("OWNER_ARCHIVE_READBACK_MISSING")
                        item["periods"].append({**period, "status": "SAVED_AND_READ_BACK", "receipt": receipt})
                    except FUND_INPUT_ERRORS as error:
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
            except FUND_INPUT_ERRORS as error:
                record_fund_failure(state, item, code, error)
            versioned_save(self.path, state)
        final = b.guard(root / "followthrough.json", blocked_file)
        if supplements is not None:
            supplements.check_storage()
        if hasattr(b, "ratings"):
            assert_daily_budget(root, now())
            state.update(status="SCORING", current_fund_code=None)
            versioned_save(self.path, state)
            try:
                state["ratings"] = b.ratings(terminal, state) if direct else b.ratings(terminal)
            except (ValueError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as error:
                state["ratings"] = {"status": "FAILED", "reason": type(error).__name__ + ":" + str(error)[:300]}
            if supplements is not None:
                supplements.check_storage()
            versioned_save(root / "ratings-status.json", state["ratings"])
        # 建议必须在本轮评分、四周期处理后生成；原有结果不能代替本期依赖。
        if hasattr(b, "advice"):
            assert_daily_budget(root, now())
            state.update(status="GENERATING_ADVICE", current_fund_code=None)
            versioned_save(self.path, state)
            try:
                state["advice"] = b.advice(state, final)
            except (ValueError, KeyError, TypeError, RuntimeError, httpx.HTTPError) as error:
                state["advice"] = {"status": "FAILED", "failure": type(error).__name__}
            versioned_save(root / "advice-status.json", state["advice"])
        if supplements is not None:
            state["supplements"] = supplements.process(state, refresh_dependents=True)
            if direct:
                terminal["status"] = state["supplements"]["status"]
                versioned_save(root / "materials-status.json", terminal)
        work_items = {
            "owner_user_id": owner, "window": state["window"], "funds": final["funds"],
            "execution": state["funds"], "monthly_compilation": state.get("monthly_compilation"),
            "ratings": state.get("ratings"), "advice": state.get("advice"),
            "supplements": state.get("supplements"),
            "remaining_funds": sorted(set(final["pending_funds"]) | {
                row["fund_code"] for row in state.get("advice", {}).get("items", [])
                if row.get("status") in {"WAITING_INPUTS", "INSUFFICIENT", "FAILED"}
            } | set((state.get("supplements") or {}).get("pending_funds", []))),
        }
        versioned_save(root / "work-items.json", work_items)
        if checklists and not strict_supplements:
            save_pending(root, state, records=records)
        # 守卫是唯一结束判据。同步摘要不完整时仍不能报告整轮完成。
        state.update(
            status=final["status"],
            all_predictions_generated=final["all_predictions_generated"],
            pending_funds=final["pending_funds"],
            current_fund_code=None,
        )
        # 审阅完成、方向数量和旧区间的变化观察分别汇总，互不冒充完成。
        for key in (
            "all_reviews_saved",
            "reviewed_period_count",
            "unreviewed_period_count",
            "dated_prediction_count",
            "conditional_saved_count",
            "abstain_saved_count",
            "blocked_saved_count",
            "reassessment_watch",
            "reassessment_pending_count",
            "reassessment_check_failure_count",
        ):
            if key in final:
                state[key] = final[key]
        if not (state["prediction_outcomes"] or {}).get("summary") and state["status"] == "CLOSED":
            state["status"] = "OUTCOME_STEP_MISSING"
        if direct and (state["prediction_outcomes"] or {}).get("status") != "SUCCEEDED" and state["status"] == "CLOSED":
            state["status"] = "OUTCOME_STEP_PENDING"
        if state["one_day_outcomes"].get("status") != "SUCCEEDED" and state["status"] == "CLOSED":
            state["status"] = "ONE_DAY_OUTCOMES_PENDING"
        if state.get("ratings", {}).get("status") != "SUCCEEDED" and "ratings" in state and state["status"] == "CLOSED":
            state["status"] = "RATINGS_PENDING"
        if state.get("advice", {}).get("status") != "SUCCEEDED" and "advice" in state and state["status"] == "CLOSED":
            state["status"] = "ADVICE_PENDING"
        if supplements is not None:
            if state["supplements"].get("unattempted_item_count"):
                state["status"] = "SUPPLEMENT_EXECUTION_INCOMPLETE"
            elif state["supplements"]["pending_item_count"] and state["status"] == "CLOSED":
                state["status"] = "SUPPLEMENTS_PENDING"
        versioned_save(self.path, state)
        return state


def record_fund_failure(state, item, code, error):
    # 基金输入或服务失败单独留首错；磁盘 I/O 错误由入口停止写入。
    failure = {"fund_code": code, "type": type(error).__name__, "reason": str(error)[:800]}
    if failure not in state["first_failures"]:
        state["first_failures"].append(failure)
    item.update(status="ACTION_REQUIRED", failure=failure)
