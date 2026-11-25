"""开启、查看每日总指挥的原生结束检查，或接受Codex Stop事件。"""

import argparse
import json
import sys
from pathlib import Path

from app.services.fund_exposure_common import PROJECT, now, read
from app.services.fund_materials_store import versioned_save

GATE_ROOT = PROJECT / ".local-runs/daily-recovery"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("arm", "status", "round-complete", "hook"), required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    from app.services.fund_daily_budget import budget_status, complete_round
    from app.services.fund_daily_stop_gate import armed_ticket, block, evaluate_stop
    from app.services.prediction_trial_v2 import storage_check

    at, ticket_path = now(), GATE_ROOT / "armed.json"
    ticket = read(ticket_path) if ticket_path.is_file() else {}
    if args.action == "status":
        print(json.dumps({**ticket, "budget": budget_status(ticket, at)} if ticket else {"enabled": False},
                         ensure_ascii=False))
        return
    if args.action == "round-complete":
        from app.commands.fund_daily_prediction import run_lock
        from app.services.fund_daily_commander import Commander

        with run_lock():
            storage_check(at.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
            ticket = read(ticket_path)
            if ticket.get("thread_id") != args.thread_id or not ticket.get("enabled"):
                raise ValueError("DAILY_ROUND_TICKET_NOT_ACTIVE")
            directory = Path(ticket["run_dir"])
            state = read(directory / "run.json")
            commander = Commander(directory, state, at)
            if ticket.get("identity") != commander.binding:
                raise ValueError("DAILY_ROUND_ORIGINAL_IDENTITY_CHANGED")
            ticket, result = complete_round(ticket, commander, at)
            if result == "RECORDED":
                versioned_save(ticket_path, ticket)
        print(json.dumps({"status": result, "budget": budget_status(ticket, at)}, ensure_ascii=False))
        return
    if args.action == "arm":
        storage_check(at.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
        directory = (args.run_dir or PROJECT / ".local-runs" / (
            "daily-prediction-" + at.strftime("%Y%m%d"))).resolve()
        if directory.parent != (PROJECT / ".local-runs").resolve():
            parser.error("结束检查只绑定本项目原每日目录")
        state = read(directory / "run.json") if (directory / "run.json").is_file() else {}
        current = armed_ticket(directory, state, args.thread_id, at)
        if ticket and ticket.get("day") == current["day"]:
            if any(ticket.get(key) != current.get(key) for key in ("thread_id", "run_dir")):
                raise ValueError("STOP_GATE_ORIGINAL_TICKET_CHANGED")
            # 同日重入不清空原绑定；首次初始化前的空绑定可在初始化后补记。
            if ticket.get("identity") and ticket["identity"] != current["identity"]:
                raise ValueError("STOP_GATE_ORIGINAL_IDENTITY_CHANGED")
            # 同日重新开启、上下文压缩或进程重启都不能清零时间与轮数。
            current = {**current, **ticket, "identity": ticket.get("identity") or current["identity"],
                       "limits": current["limits"], "mode": current["mode"]}
        versioned_save(ticket_path, current)
        print(json.dumps(current, ensure_ascii=False))
        return
    payload = json.load(sys.stdin)
    applicable = bool(payload.get("session_id") == args.thread_id and ticket.get("thread_id") == args.thread_id
                      and payload.get("hook_event_name") == "Stop" and ticket.get("enabled"))
    try:
        storage_check(at.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
        # 先留下入口回执，区分未被调用与被调用后因身份/票据不匹配而跳过；不保存对话正文。
        versioned_save(GATE_ROOT / "last-hook-invocation.json", {
            "received_at": at.isoformat(), "event_name": payload.get("hook_event_name"),
            "session_id": payload.get("session_id"), "turn_id": payload.get("turn_id"),
            "expected_thread_id": args.thread_id, "ticket_thread_id": ticket.get("thread_id"),
            "ticket_enabled": bool(ticket.get("enabled")), "applicable": applicable,
            "payload_keys": sorted(payload),
        })
        if not applicable:
            print("{}")
            return
        decision, record = evaluate_stop(payload, ticket, at)
        record.update(checked_at=at.isoformat(), turn_id=payload.get("turn_id"), thread_id=args.thread_id)
        versioned_save(GATE_ROOT / "last-stop-check.json", record)
        if record.get("accepted"):
            versioned_save(ticket_path, {**ticket, "enabled": False, "accepted_at": at.isoformat()})
        elif record["status"] in {"PAUSED_TIME_LIMIT", "PAUSED_ROUND_LIMIT", "PAUSED_WINDOW_EXPIRED",
                                 "PAUSED_DATE_CHANGED"}:
            versioned_save(ticket_path, {**ticket, "enabled": False, "paused_at": at.isoformat(),
                                        "pause_status": record["status"]})
    except OSError:
        # 发现存储故障后不再写项目记录；暂停不是验收成功。
        decision = ({"continue": False, "stopReason": "存储检查或读写失败，停止写入；每日任务仍未验收完成。"}
                    if applicable else {})
    except Exception as error:
        # 检查器故障必须显式拦住正常收尾，不能让hook进程异常造成默认放行。
        decision = (block(f"结束检查未能完成（{type(error).__name__}），不能登记任务完成。"
                          "先修复检查入口，再恢复原总指挥队列；不要关闭或修改hook来绕过验收。")
                    if applicable else {})
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
