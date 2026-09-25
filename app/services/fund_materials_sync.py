"""同步中心的 002112 持续资料任务；独立游标、逐阶段恢复和校验后发布。"""

from datetime import date, datetime, timedelta
from uuid import uuid4

from app.core.logging import get_logger
from app.integrations.cninfo_exposure import acquire_announcements, acquire_attachments
from app.integrations.dbfund_reports import acquire
from app.integrations.dbfund_supplement import acquire_documents
from app.services.fund_exposure_common import FUND, ROOT, now, read, save
from app.services.fund_exposure_field_enrichment import update_financials
from app.services.fund_exposure_quotes import acquire_quotes, reports, safe_error
from app.services.fund_exposure_runtime import execution_lock
from app.services.fund_materials_build import build
from app.services.fund_materials_store import LIVE, versioned_save
from app.services.fund_peer_materials import supplement as supplement_peers
from app.services.fund_training_package import supplement as prepare_training_materials

STAGES = (
    "基金报告与持仓",
    "关联股票行情",
    "公司财务与经营",
    "公司公告目录",
    "新增公司公告原文",
    "基金公告与公司新闻",
    "参照基金历史持仓与行情",
    "补齐并检查一日训练资料",
    "检查并更新页面资料",
)
logger = get_logger(__name__)


def failure_message(error: dict) -> str:
    """同步中心显示可理解的原因；技术错误码及公司、接口范围只保留在运行记录中。"""
    reason = error.get("reason", "")
    messages = {
        "EXPOSURE_BUSY": "已有后台维护正在执行",
        "MATERIAL_SOURCES_INCOMPLETE": "部分资料未齐，保留上一份页面资料",
        "PUBLIC_BODY_NOT_AVAILABLE": "正文暂未找到公开可读来源",
        "FileNotFoundError": "已保存的来源文件缺失",
        "PermissionError": "本地文件暂时无法写入",
        "REPORT_PERIODS_MISSING": "部分参照基金历史报告尚未补齐",
        "REPORT_CATALOG_EMPTY": "来源未返回参照基金历史报告，不能当作没有资料",
        "REPORT_CATALOG_DISAPPEARED": "来源目录发生变化，需要核对原报告状态",
        "REPORT_SOURCE_TEMPORARILY_UNAVAILABLE": "来源连续连接失败，本轮已停止重复请求，成功资料已保留",
        "REPORT_BODY_EMPTY_OR_TRUNCATED": "来源正文为空或内容不足，尚未补齐",
        "REPORT_REPRINT_PERIOD_NOT_FOUND": "公开来源暂未找到该期完整报告，相关日期未用于训练",
        "SOURCE_RETURNED_EMPTY": "来源未返回该段历史行情，相关日期已排除",
        "REPORT_PEER_QUOTES_INCOMPLETE": "部分参照股票的历史行情仍未核验，已保留具体缺口",
        "MORE_QUALIFIED_TRAINING_DATA_REQUIRED": "可用于一日研究的完整记录还不够，已保存合格部分和具体缺口",
        "EXPOSURE_PROVIDER_REJECTED": "来源拒绝本次请求，请核对现有权限或稍后重试",
    }
    if reason in messages:
        return messages[reason]
    if reason in {"ReadTimeout", "ConnectTimeout", "ConnectError", "HTTPStatusError", "ReadError"}:
        return "来源暂时无法访问，已保存成果可在重试时复用"
    if "LIMIT" in reason or "BUDGET" in reason:
        return "达到本轮检查上限，已保存进度"
    return "资料未通过完整性检查，请重试；持续失败时需检查来源"


def current_scope(cutoff: str, previous_cutoff: str) -> dict:
    """从游标前 30 日回查，逐份公开报告形成关联窗口；新持仓出现后自动纳入。

    保留原有已披露持仓最多 210 日的有效边界。旧公司退出后不再无限采集其未来消息。
    """
    from datetime import time

    from app.services.direction_1d_protocol import ZONE, digest
    from app.services.fund_exposure_features import select_report

    rs = reports()
    stop = datetime.strptime(cutoff, "%Y%m%d").date()
    day = max(date(2021, 1, 1), date.fromisoformat(previous_cutoff) - timedelta(days=30))
    windows = {}
    while day <= stop:
        selected = select_report(rs, datetime.combine(day, time(23, 59), ZONE))
        if (day - date.fromisoformat(selected["report_end"])).days <= 210:
            for holding in selected["holdings"]:
                code = holding["stock_code"]
                periods = windows.setdefault(code, [])
                left = max(date(2021, 1, 1), day - timedelta(days=30))
                if periods and str(left) <= periods[-1][1]:
                    periods[-1][1] = str(day)
                else:
                    periods.append([str(left), str(day)])
        day += timedelta(days=1)
    if not windows:
        raise ValueError("EXPOSURE_CURRENT_REPORT_UNAVAILABLE")
    return {
        "windows": windows,
        "cutoff": str(stop),
        "maximum_pages_per_window": 150,
        "maximum_new_requests": 4000,
        "report_hash": digest(rs),
        "training_eligible": False,
    }


class FundMaterialsSyncService:
    """所有入口都进入现有任务队列，真正采集再持有与旧维护相同的数据库锁。"""

    def sync(self, fund_code: str, *, progress_reporter=lambda *args: None) -> dict:
        if fund_code != FUND:
            raise ValueError("仅支持 002112，基金代码不能为空。")
        with execution_lock() as locked:
            if not locked:
                return {
                    "status": "FAILED",
                    "created": 0,
                    "updated": 0,
                    "skipped": 0,
                    "failed": 1,
                    "message": "已有持仓资料后台维护正在执行，请稍后重试。",
                    "errors": [{"stage": "并发检查", "reason": "EXPOSURE_BUSY"}],
                }
            return self._run(progress_reporter)

    def _run(self, reporter):
        state_path = LIVE / "state.json"
        previous = read(state_path) if state_path.exists() else {}
        cutoff = now().strftime("%Y%m%d")
        baseline = read(ROOT / "supplement/plan.json")["end_date"]
        baseline = datetime.strptime(baseline, "%Y%m%d").date().isoformat()
        # 中断或部分失败重试沿用同一天的完成阶段，新的自然日重新检查目录与来源。
        resume = previous.get("cutoff") == cutoff and previous.get("status") != "SUCCEEDED"
        stages = previous.get("stages", {}) if resume else {}
        check_id = previous.get("check_id") if resume else None
        check_id = check_id or now().isoformat()
        run_id = str(uuid4())
        state = {
            "run_id": run_id,
            "status": "RUNNING",
            "started_at": now().isoformat(),
            "cutoff": cutoff,
            "check_id": check_id,
            "stages": stages,
            "last_success_at": previous.get("last_success_at"),
            "last_complete_through": previous.get("last_complete_through", baseline),
            "resumed_from": previous.get("run_id"),
            "errors": [],
            "enabled": True,
        }
        save(state_path, state, replace=True)
        scope = None
        counts = {"created": 0, "updated": 0, "skipped": 0, "failed": 0}
        for index, title in enumerate(STAGES):
            if title in stages and stages[title].get("status") == "SUCCEEDED" and title != STAGES[-1]:
                completed = stages[title].get("result", {})
                counts["skipped"] += sum(completed.get(key, 0) for key in ("created", "updated", "skipped"))
                continue
            state["current_stage"] = title
            state["stage_progress"] = {"current": 0, "total": 0, "code": None}
            state["progress_current"], state["progress_total"] = index, len(STAGES)
            state["counts"] = counts
            save(state_path, state, replace=True)

            def progress(current, total, code, message, index=index, title=title):
                state["stage_progress"] = {"current": current, "total": total, "code": code}
                save(state_path, state, replace=True)
                reporter(index, len(STAGES), FUND, f"{title}：{current}/{total}，{message}")

            reporter(index, len(STAGES), FUND, title)
            try:
                if index == 0:
                    old_count = len(read(ROOT / "report-result.json")["report_files"])
                    result = acquire(incremental=True)
                    result["created"] = max(0, result["reports"] - old_count)
                    result["skipped"] = old_count
                    if result["created"]:
                        # 报告变化后必须重算公司范围，不能复用失败轮中基于旧报告的后续成果。
                        for dependent in STAGES[1:]:
                            stages.pop(dependent, None)
                elif index == 1:
                    result = acquire_quotes(incremental=True)
                elif index in (2, 3):
                    scope = scope or current_scope(cutoff, state["last_complete_through"])
                    scope["check_id"] = check_id
                    result = (
                        update_financials(sorted(scope["windows"]), cutoff, check_id=check_id, progress=progress)
                        if index == 2
                        else acquire_announcements(live_scope=scope, progress=progress)
                    )
                    if index == 3:
                        # 目录重试可能发现新附件；原文检查必须随目录再次执行（旧 PDF 会复用）。
                        stages.pop(STAGES[4], None)
                elif index == 4:
                    result = acquire_attachments(live=True, progress=progress)
                elif index == 5:
                    result = acquire_documents(live=True, progress=progress)
                elif index == 6:
                    result = supplement_peers(check_id=check_id, progress=progress)
                    # 参照资料变化后必须重新检查训练包，不沿用上一轮的就绪结论。
                    stages.pop(STAGES[7], None)
                elif index == 7:
                    result = prepare_training_materials(progress=progress)
                else:
                    # 参照基金独立保存，补充失败不能阻断已检查完整的 002112 详情更新。
                    # 主任务仍会如实返回部分成功，直到参照基金这一步也结束。
                    if any(error["stage"] not in STAGES[6:8] for error in state["errors"]):
                        raise ValueError("MATERIAL_SOURCES_INCOMPLETE")
                    result = build(checked_through=cutoff)
                errors = result.get("errors", [])
                for key in ("created", "updated", "skipped"):
                    counts[key] += result.get(key, 0)
                counts["failed"] += len(errors)
                stages[title] = {
                    "status": "PARTIAL_SUCCESS" if errors else "SUCCEEDED",
                    "result": result,
                    "finished_at": now().isoformat(),
                }
                state["errors"].extend({"stage": title, **e} for e in errors)
            except Exception as exc:
                logger.exception("fund_materials_sync._run >>> 阶段未完成, run_id=%s, stage=%s", run_id, title)
                counts["failed"] += 1
                error = {"stage": title, "reason": safe_error(exc)}
                state["errors"].append(error)
                stages[title] = {"status": "FAILED", "error": error, "finished_at": now().isoformat()}
            save(state_path, state, replace=True)
        success = not state["errors"]
        state.update(
            status="SUCCEEDED"
            if success
            else "PARTIAL_SUCCESS"
            if any(s["status"] == "SUCCEEDED" for s in stages.values())
            else "FAILED",
            finished_at=now().isoformat(),
            counts=counts,
            progress_current=len(STAGES),
        )
        page_updated = stages.get(STAGES[-1], {}).get("status") == "SUCCEEDED"
        if success:
            state["last_success_at"] = state["finished_at"]
        if page_updated:
            # 当前资料的游标不受独立历史补充失败影响；整体成功时间仍须所有步骤成功。
            state["last_page_success_at"] = state["finished_at"]
            state["last_complete_through"] = datetime.strptime(cutoff, "%Y%m%d").date().isoformat()
        state["next_attempt_at"] = (now() + timedelta(hours=12 if success else 2)).isoformat()
        versioned_save(state_path, state)
        save(LIVE / "runs" / (run_id + ".json"), state)
        message = (
            "资料更新完成，详情页已使用检查后的资料"
            if success
            else (
                "部分资料尚未完成；002112 详情已更新，参照基金成功成果已保存"
                if page_updated
                else "资料尚有未完成项；成功成果已保存，页面保留上一份完整资料"
            )
        )
        peer_result = stages.get(STAGES[6], {}).get("result", {})
        if peer_result.get("coverage"):
            count = peer_result["coverage"]["fit_504"]
            message += (
                f"；参照基金已配齐 {count['rows']} 条历史记录（持平 {count['directions']['FLAT']} 条），尚未用于训练"
            )
        readiness = stages.get(STAGES[7], {}).get("result", {}).get("readiness", {})
        if readiness.get("training_ready"):
            message += "；已达到 002112 一日离线训练的数据条件，尚未训练或启用"
        message += (
            f"；新增 {counts['created']}，更新 {counts['updated']}，复用 {counts['skipped']} 条记录或文件；"
            f"未完成 {counts['failed']} 项"
        )
        reporter(len(STAGES), len(STAGES), FUND, message)
        return {"status": state["status"], "message": message, "errors": state["errors"], **counts}


def schedule_if_due():
    """首次手动使用后启用持续检查；已有维护循环只登记任务，不能绕过队列直接采集。"""
    path = LIVE / "state.json"
    if not path.exists():
        return
    state = read(path)
    if not state.get("enabled") or (
        state.get("next_attempt_at") and now() < datetime.fromisoformat(state["next_attempt_at"])
    ):
        return
    from app.services.sync_jobs import SyncJobInProgressError, get_sync_job_manager

    try:
        get_sync_job_manager().start_fund_materials(FUND)
    except SyncJobInProgressError:
        pass
