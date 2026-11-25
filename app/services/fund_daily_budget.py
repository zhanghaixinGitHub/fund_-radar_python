"""用户授权的整轮预算：到达配置的运行分钟或复核轮数就暂停，不能冒充业务完成。"""

import json
from datetime import datetime, timedelta
from pathlib import Path

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import PROJECT, read

TICKET_PATH = PROJECT / ".local-runs/daily-recovery/armed.json"
LIMITS_PATH = PROJECT / "app/data/daily_commander_limits.json"


class DailyBudgetReached(Exception):
    """独立于业务输入异常，防止被单基金错误处理吞掉后继续派工。"""


def limits():
    # 工程配置是普通JSON；只有运行证据才采用hash/payload封装。
    result = json.loads(LIMITS_PATH.read_text(encoding="utf-8"))
    if any(type(result.get(key)) is not int or result[key] <= 0
           for key in ("max_runtime_minutes", "max_rounds")):
        raise ValueError("DAILY_BUDGET_LIMITS_INVALID")
    return result


def budget_status(ticket, at):
    configured = ticket.get("limits") or limits()
    if any(type(configured.get(key)) is not int or configured[key] <= 0
           for key in ("max_runtime_minutes", "max_rounds")):
        raise ValueError("DAILY_BUDGET_TICKET_LIMITS_INVALID")
    start = datetime.fromisoformat(ticket["armed_at"])
    if start.tzinfo is None:
        raise ValueError("DAILY_BUDGET_START_TIMEZONE_REQUIRED")
    completed = ticket.get("completed_rounds", 0)
    if type(completed) is not int or completed < 0:
        raise ValueError("DAILY_BUDGET_ROUND_COUNT_INVALID")
    deadline = start + timedelta(minutes=configured["max_runtime_minutes"])
    reason = "TIME_LIMIT" if at >= deadline else "ROUND_LIMIT" if completed >= configured["max_rounds"] else None
    return {**configured, "started_at": start.isoformat(), "deadline_at": deadline.isoformat(),
            "elapsed_minutes": round(max(0, (at - start).total_seconds()) / 60, 3),
            "completed_rounds": completed, "current_round": min(completed + 1, configured["max_rounds"]),
            "limit_reached": reason is not None, "pause_reason": reason}


def run_budget(directory, at):
    if not TICKET_PATH.is_file():
        return None
    ticket = read(TICKET_PATH)
    if Path(ticket.get("run_dir", ".")).resolve() != Path(directory).resolve() or ticket.get("accepted_at"):
        return None
    return budget_status(ticket, at)


def assert_daily_budget(directory, at):
    status = run_budget(directory, at)
    if status and status["limit_reached"]:
        raise DailyBudgetReached("DAILY_" + status["pause_reason"] + "_PAUSED_NOT_ACCEPTED")


def complete_round(ticket, commander, at):
    """一轮必须覆盖全持仓实际验收；同一批回执重复提交不能累计轮数。"""
    status = budget_status(ticket, at)
    if status["limit_reached"]:
        raise DailyBudgetReached("DAILY_" + status["pause_reason"] + "_PAUSED_NOT_ACCEPTED")
    if commander.control.get("assignments"):
        raise ValueError("DAILY_ROUND_WORKERS_STILL_ACTIVE")
    snapshot = commander.snapshot()
    if commander.audit.get("status") == "RECEIPTS_INCOMPLETE":
        raise ValueError("DAILY_ROUND_RECEIPTS_INCOMPLETE")
    reviewed = commander.control.get("acceptances", {})
    checkpoint = {}
    missing = []
    round_start = datetime.fromisoformat(ticket.get("round_started_at") or ticket["armed_at"])
    for row in snapshot["funds"]:
        code = row["fund_code"]
        proof = reviewed.get(code, {})
        current = commander.fingerprint(code)
        checked = proof.get("checked_at")
        if proof.get("fingerprint") != current or not checked:
            missing.append(code)
            continue
        checkpoint[code] = {"fingerprint": current, "checked_at": checked}
    token = digest(checkpoint)
    if ticket.get("last_round_checkpoint") == token:
        return ticket, "ALREADY_RECORDED"
    for row in snapshot["funds"]:
        code = row["fund_code"]
        if code not in checkpoint:
            continue
        stamp = datetime.fromisoformat(checkpoint[code]["checked_at"])
        path = commander.directory / "supplement-funds" / (code + ".json")
        ledger = read(path) if path.is_file() else {}
        if (stamp.tzinfo is None or stamp < round_start or stamp > at
                or not ledger.get("items") or any(item.get("status") == "NOT_ATTEMPTED"
                                                 for item in ledger["items"])
                or row["action"] in {"REPAIR_RECEIPTS", "DISPATCH", "VERIFY_RETURN", "COLLECT_WORKER"}
                or row["action"] == "RUN_FUND_BUSINESS" and not row.get("reviews_saved")):
            missing.append(code)
    if missing or set(checkpoint) != set(commander.state["scope"]):
        raise ValueError("DAILY_ROUND_NOT_REVIEWED:" + ",".join(sorted(set(missing))))
    count = status["completed_rounds"] + 1
    result = {**ticket, "completed_rounds": count, "round_started_at": at.isoformat(),
              "last_round_checkpoint": token, "round_history": [*ticket.get("round_history", []), {
                  "round": count, "completed_at": at.isoformat(), "scope": commander.state["scope"],
                  "checkpoint": checkpoint}]}
    return result, "RECORDED"
