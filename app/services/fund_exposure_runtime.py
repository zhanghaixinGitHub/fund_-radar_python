"""002112 分析资料更新与当时输入留存；由一键同步触发，不生成预测或训练模型。"""

from contextlib import contextmanager
from datetime import date, datetime, timedelta

from sqlalchemy import text

from app.db.session import get_engine
from app.integrations.dbfund_reports import acquire
from app.integrations.dbfund_supplement import acquire_nav, fill_live_nav_gap
from app.repositories import direction_1d as repo
from app.services.direction_1d_protocol import digest, features, input_days, window
from app.services.fund_exposure_common import FUND, ROOT, initialize, now, read, save
from app.services.fund_exposure_features import calculate, load_bundle, specification
from app.services.fund_exposure_quotes import acquire_quotes, safe_error


@contextmanager
def execution_lock():
    """同一数据库多进程共用单基金锁，CLI 和后台都不能重复执行采集。"""
    with get_engine().connect() as connection:
        locked = connection.execute(text("SELECT pg_try_advisory_lock(20260924, 2112)")).scalar_one()
        try:
            yield bool(locked)
        finally:
            if locked:
                connection.execute(text("SELECT pg_advisory_unlock(20260924, 2112)"))


def capture():
    """保存真实时刻的输入；目标日首次成功版本只创建一次，候选状态与答案明确分开。"""
    specification()
    current = now()
    w = window(current)
    target = w["target_nav_date"]
    path = ROOT / "forward-inputs" / (target + ".json")
    if path.exists():
        value = read(path)
        return {"status": "ALREADY_CAPTURED", "target": target, "input_hash": digest(value)}
    if w["status"] != "OPEN":
        return {"status": "DEADLINE_PASSED", "target": target}
    base = date.fromisoformat(w["base_nav_date"])
    required = input_days(base)
    with get_engine().connect() as c:
        source = repo.source(c)
        nav = repo.navs(c, FUND, source["source_id"], required[0], base)
    nav = fill_live_nav_gap(nav, required, current)
    if [r["nav_date"] for r in nav] != list(required):
        return {"status": "WAITING_NAV", "target": target, "base": str(base), "rows": len(nav), "required": 61}
    if any(r["ann_date"] and r["ann_date"] > current.date() for r in nav):
        raise ValueError("EXPOSURE_FUTURE_NAV_ANNOUNCEMENT")
    bundle = load_bundle()
    # 真实数据加载与核验完成后再取时间，不能把后来才收到的资料记成早已收到。
    recorded_at = now()
    if recorded_at >= datetime.fromisoformat(w["deadline_at"]):
        return {"status": "DEADLINE_PASSED", "target": target}
    exposure = calculate(bundle, base, recorded_at, live=True)
    if exposure["status"] != "AVAILABLE" or exposure["market_features"] is None:
        return {
            "status": "WAITING_QUOTES",
            "target": target,
            "missing_stocks": len(exposure["missing"]),
            "market_errors": exposure["market_errors"],
        }
    study = read(ROOT / read(ROOT / "study-current.json")["file"])
    # 本轮训练门槛未通过。记录输入是为以后审计来源，不等同于提前预测或训练样本。
    if study["status"] != "INSUFFICIENT_EVIDENCE_KEEP_EXISTING":
        raise ValueError("EXPOSURE_STUDY_STATE_UNEXPECTED")
    input_expiry = min(
        [recorded_at + timedelta(days=source["retention_days"])]
        + [datetime.fromisoformat(bundle["days"][d]["receipt"]["expires_at"]) for d in exposure["quote_hashes"]]
        + [datetime.fromisoformat(r["expires_at"]) for v in bundle["indices"].values() for r in v["receipts"]]
    )
    value = {
        "fund_code": FUND,
        "kind": "LIVE_INPUT_ONLY_NOT_A_PREDICTION",
        "generated_at": recorded_at.isoformat(),
        "window": w,
        "nav_features": features([r["unit_nav"] for r in nav]),
        "nav_rows": [
            {
                "date": str(r["nav_date"]),
                "unit_nav": str(r["unit_nav"]),
                "upstream_hash": r["content_hash"],
                "source_code": r.get("source_code", "TUSHARE_PRO_FUND"),
                "received_at": r.get("received_at"),
                "official_receipt": r.get("official_receipt"),
            }
            for r in nav
        ],
        "exposure": exposure,
        "study_hash": digest(study),
        "candidate_direction": None,
        "candidate_status": "BLOCKED_INSUFFICIENT_THREE_STATE_SAMPLES",
        "training_eligible": False,
        "expires_at": input_expiry.isoformat(),
    }
    save(path, value)
    return {"status": "INPUT_CAPTURED_CANDIDATE_BLOCKED", "target": target, "input_hash": digest(value)}


def tick(*, manual=False, progress_reporter=None):
    """执行一次有限维护；手动运行跳过旧自动开关和等待时间，仍遵守研究截止与基金锁。

    manual 只改变触发方式，不初始化研究、不延长允许日期，也不修改已有输入快照。
    进度共四步：基金报告、关联行情、官网净值、当时输入留存。
    """
    control_path = ROOT / "runtime-control.json"
    if not control_path.exists():
        return {"status": "NOT_ENABLED"}
    control = read(control_path)
    if (not manual and not control["enabled"]) or now() >= datetime.fromisoformat(control["until"]):
        return {"status": "DISABLED_OR_EXPIRED"}
    state_path = ROOT / "runtime-state.json"
    state = read(state_path) if state_path.exists() else {}
    current = now()
    if not manual and state.get("next_attempt_at") and current < datetime.fromisoformat(state["next_attempt_at"]):
        return {"status": "NOT_DUE"}
    # 清除本轮结果字段，失败时不能把上一次成功的官网净值或快照算成本次成果。
    for key in ("capture", "official_nav", "quote_errors", "reason", "next_attempt_at"):
        state.pop(key, None)
    state["last_attempt_at"] = current.isoformat()
    if not manual:
        state["next_attempt_at"] = (current + timedelta(minutes=30)).isoformat()
    report = progress_reporter or (lambda *_: None)
    save(state_path, state, replace=True)
    try:
        report(0, 4, FUND, "正在检查 002112 基金报告")
        if not state.get("reports_checked_at") or current - datetime.fromisoformat(
            state["reports_checked_at"]
        ) >= timedelta(days=7):
            result = acquire(incremental=True)
            if result["errors"]:
                raise ValueError("EXPOSURE_REPORT_REFRESH_INCOMPLETE")
            state["reports_checked_at"] = now().isoformat()
        report(1, 4, FUND, "正在更新 002112 关联股票行情")
        quote_result = acquire_quotes(incremental=True)
        state["quote_errors"] = quote_result["errors"]
        # 官网正式净值比供应商先更新时，只用于这个试点的真实输入保存，不写回原模型来源。
        report(2, 4, FUND, "正在检查 002112 官网净值")
        try:
            state["official_nav"] = acquire_nav(latest_only=True)
        except Exception as exc:
            state["official_nav"] = {"status": "UNAVAILABLE", "reason": safe_error(exc)}
        report(3, 4, FUND, "正在保存 002112 当时可得的分析资料")
        state["capture"] = capture()
        state["status"] = "COMPLETED_INPUT_MAINTENANCE"
        report(4, 4, FUND, "002112 分析资料检查结束")
    except Exception as exc:
        state["status"] = "RETRY_PENDING"
        state["reason"] = safe_error(exc)
        if not manual:
            state["next_attempt_at"] = (now() + timedelta(hours=2)).isoformat()
    state["finished_at"] = now().isoformat()
    save(state_path, state, replace=True)
    return state


def maintenance_once(*, manual=False, progress_reporter=None):
    """执行已准备好的 002112 维护；与资料补齐共用数据库锁，不影响其他基金。"""
    if not (ROOT / "runtime-control.json").exists():
        return {"status": "NOT_ENABLED"}
    with execution_lock() as locked:
        return tick(manual=manual, progress_reporter=progress_reporter) if locked else {"status": "BUSY"}


def synchronize_inputs(*, progress_reporter):
    """同步中心手动入口；将实际取数和留存结果映射为明确的成功、部分完成或失败。

    原维护循环的 COMPLETED 只表示流程走完，缺少净值、行情或错过留存时间均不算全成功。
    外部原始错误只留在维护记录，页面仅返回自然中文说明。
    """
    result = maintenance_once(manual=True, progress_reporter=progress_reporter)
    status = result["status"]
    if status != "COMPLETED_INPUT_MAINTENANCE":
        message = {
            "NOT_ENABLED": "002112 分析资料尚未准备好，本次未更新。",
            "DISABLED_OR_EXPIRED": "002112 分析资料已超出当前允许更新的日期范围。",
            "BUSY": "002112 资料正在被其他任务更新，请稍后重新执行一键同步。",
        }.get(status, "002112 分析资料更新未完成，请稍后重新执行一键同步。")
        return {"status": "FAILED", "message": message, "created": 0, "skipped": 0}
    gaps = []
    if result.get("quote_errors"):
        gaps.append("部分关联股票行情未更新")
    if result.get("official_nav", {}).get("status") == "UNAVAILABLE":
        gaps.append("官网净值暂时无法获取")
    capture_status = result.get("capture", {}).get("status")
    if capture_status not in {"INPUT_CAPTURED_CANDIDATE_BLOCKED", "ALREADY_CAPTURED"}:
        gaps.append({
            "WAITING_NAV": "所需净值尚未齐全",
            "WAITING_QUOTES": "所需行情尚未齐全",
            "DEADLINE_PASSED": "已错过当期资料留存时间",
        }.get(capture_status, "当期分析资料尚未保存"))
    return {
        "status": "PARTIAL_SUCCESS" if gaps else "SUCCEEDED",
        "message": "002112 分析资料更新未完整完成：" + "；".join(gaps) if gaps
        else "002112 分析资料已更新，当期记录已保存或复用。",
        "created": int(capture_status == "INPUT_CAPTURED_CANDIDATE_BLOCKED"),
        "skipped": int(capture_status == "ALREADY_CAPTURED"),
    }


def enable():
    """仅启用输入采集，不启用新模型；到现有日历末自动停止，避免超范围日期。"""
    initialize()
    study = read(ROOT / read(ROOT / "study-current.json")["file"])
    if study["fits"] != 0 or study["model_registry_changed"]:
        raise ValueError("EXPOSURE_ENABLE_PRECONDITION")
    value = {
        "enabled": True,
        "at": now().isoformat(),
        "until": "2026-12-31T15:00:00+08:00",
        "fund_code": FUND,
        "purpose": "INPUT_COLLECTION_ONLY",
        "automatic_model_adoption": False,
    }
    path = ROOT / "runtime-control.json"
    if not path.exists():
        save(path, value)
    elif not read(path)["enabled"]:
        save(path, value, replace=True)
    return read(path)


def disable():
    """仅关闭本轮输入维护，保留证据和原预测，不要求人工编辑带哈希的文件。"""
    path = ROOT / "runtime-control.json"
    if not path.exists():
        return {"status": "NOT_ENABLED"}
    control = read(path)
    control.update({"enabled": False, "disabled_at": now().isoformat()})
    save(path, control, replace=True)
    return {"status": "DISABLED", "existing_predictions_unchanged": True}
