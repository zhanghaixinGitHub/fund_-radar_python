"""复用净值维护循环的 002112 输入留存；候选不合格时绝不产生虚构预测。"""

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


def tick():
    """一次有限增量维护；持久化尝试时间控制重试，重启后不会重置频率。"""
    control_path = ROOT / "runtime-control.json"
    if not control_path.exists():
        return {"status": "NOT_ENABLED"}
    control = read(control_path)
    if not control["enabled"] or now() >= datetime.fromisoformat(control["until"]):
        return {"status": "DISABLED_OR_EXPIRED"}
    state_path = ROOT / "runtime-state.json"
    state = read(state_path) if state_path.exists() else {}
    current = now()
    if state.get("next_attempt_at") and current < datetime.fromisoformat(state["next_attempt_at"]):
        return {"status": "NOT_DUE"}
    state.update(
        {"last_attempt_at": current.isoformat(), "next_attempt_at": (current + timedelta(minutes=30)).isoformat()}
    )
    save(state_path, state, replace=True)
    try:
        if not state.get("reports_checked_at") or current - datetime.fromisoformat(
            state["reports_checked_at"]
        ) >= timedelta(days=7):
            result = acquire(incremental=True)
            if result["errors"]:
                raise ValueError("EXPOSURE_REPORT_REFRESH_INCOMPLETE")
            state["reports_checked_at"] = now().isoformat()
        quote_result = acquire_quotes(incremental=True)
        state["quote_errors"] = quote_result["errors"]
        # 官网正式净值比供应商先更新时，只用于这个试点的真实输入保存，不写回原模型来源。
        try:
            state["official_nav"] = acquire_nav(latest_only=True)
        except Exception as exc:
            state["official_nav"] = {"status": "UNAVAILABLE", "reason": safe_error(exc)}
        state["capture"] = capture()
        state["status"] = "COMPLETED_INPUT_MAINTENANCE"
    except Exception as exc:
        state["status"] = "RETRY_PENDING"
        state["reason"] = safe_error(exc)
        state["next_attempt_at"] = (now() + timedelta(hours=2)).isoformat()
    state["finished_at"] = now().isoformat()
    save(state_path, state, replace=True)
    return state


def maintenance_once():
    """后台入口默认关闭；只在本轮本地控制文件启用时生效，不影响其他基金。"""
    if not (ROOT / "runtime-control.json").exists():
        return {"status": "NOT_ENABLED"}
    with execution_lock() as locked:
        return tick() if locked else {"status": "BUSY"}


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
